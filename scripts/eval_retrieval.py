"""
Evaluate retrieval effectiveness of the deduplication agent.

This script evaluates how well the dedup agent retrieves relevant cached reasoning traces
and measures the reuse rate, which indicates cost savings from avoiding redundant web searches.

Metrics:
- Reuse Rate: Percentage of questions that can be answered from cached knowledge
- Retrieval Accuracy: Quality of similarity matching (top-k)
- Average Similarity Score: Mean cosine similarity of retrieved entries
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from q2k.dedup_agent import run_dedup_agent
from q2k.reasoning_agent import run_reasoning_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Thread-safe metrics collector for concurrent evaluation."""

    def __init__(self):
        self.total_pairs = 0
        self.total_questions = 0
        self.total_reusable = 0
        self.total_fresh_needed = 0
        self.retrieval_counts = []
        self.similarity_scores = []
        self.reuse_rates = []
        self.lock = asyncio.Lock()

    async def add_result(
        self,
        num_questions: int,
        num_reusable: int,
        num_fresh: int,
        num_retrieved: int,
        similarities: list[float],
    ):
        """Add results from a single pair evaluation."""
        async with self.lock:
            self.total_pairs += 1
            self.total_questions += num_questions
            self.total_reusable += num_reusable
            self.total_fresh_needed += num_fresh

            if num_retrieved > 0:
                self.retrieval_counts.append(num_retrieved)
            if similarities:
                self.similarity_scores.extend(similarities)

            reuse_rate = (
                (num_reusable / num_questions * 100) if num_questions > 0 else 0
            )
            self.reuse_rates.append(reuse_rate)


async def evaluate_pair(
    pair: dict,
    idx: int,
    total_pairs: int,
    qdb_file: str,
    top_k: int,
    similarity_threshold: float,
    semaphore: asyncio.Semaphore,
    metrics: MetricsCollector,
):
    """
    Evaluate a single product pair.

    Args:
        pair: Product pair dictionary
        idx: Index of this pair (1-indexed)
        total_pairs: Total number of pairs
        qdb_file: Path to QDB file
        top_k: Number of similar entries to retrieve
        similarity_threshold: Minimum similarity threshold
        semaphore: Semaphore to limit concurrent API calls
        metrics: Shared metrics collector
    """
    base = pair["base"]
    candidate = pair["candidate"]
    label = pair.get("label", "unknown")

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating Pair {idx}/{total_pairs} (Label: {label})")
    logger.info(f"Base: {base[:80]}...")
    logger.info(f"Candidate: {candidate[:80]}...")

    try:
        # Generate questions using reasoning agent
        async with semaphore:
            reasoning_result = await run_reasoning_agent(base, candidate)

        questions = reasoning_result["questions"]

        if not questions:
            logger.info(f"Pair {idx}: No questions generated, skipping")
            return

        logger.info(f"Pair {idx}: Generated {len(questions)} questions:")
        for i, q in enumerate(questions, 1):
            logger.info(f"  {i}. {q[:80]}...")

        # Run dedup agent
        async with semaphore:
            result = await run_dedup_agent(
                questions=questions,
                qdb_path=qdb_file,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

        # Update metrics
        await metrics.add_result(
            num_questions=len(questions),
            num_reusable=len(result.reusable_answers),
            num_fresh=len(result.questions_needing_fresh_search),
            num_retrieved=len(result.retrieved_entries),
            similarities=result.similarities,
        )

        # Calculate reuse rate for this pair
        pair_reuse_rate = (
            (len(result.reusable_answers) / len(questions) * 100) if questions else 0
        )

        # Log results
        logger.info(f"\nPair {idx} - Retrieval Results:")
        logger.info(f"  Retrieved entries: {len(result.retrieved_entries)}")
        if result.similarities:
            logger.info(
                f"  Similarity scores: {[f'{s:.3f}' for s in result.similarities]}"
            )
        logger.info(f"  Reusable answers: {len(result.reusable_answers)}")
        logger.info(
            f"  Fresh searches needed: {len(result.questions_needing_fresh_search)}"
        )
        logger.info(f"  Reuse rate: {pair_reuse_rate:.1f}%")

        if result.reusable_answers:
            logger.info("\n  Reusable answers:")
            for answer in result.reusable_answers:
                logger.info(
                    f"    - Q{answer['new_question_idx']}: {answer['new_question'][:60]}..."
                )

        if result.questions_needing_fresh_search:
            logger.info("\n  Questions needing fresh search:")
            for q in result.questions_needing_fresh_search:
                logger.info(f"    - Q{q['question_idx']}: {q['question'][:60]}...")

    except Exception as e:
        logger.error(f"Pair {idx}: Error evaluating - {e}")


async def evaluate_retrieval(
    pairs_file: str,
    qdb_file: str,
    top_k: int = 5,
    similarity_threshold: float = 0.7,
    max_pairs: int = None,
    concurrency: int = 10,
):
    """
    Evaluate retrieval effectiveness on a dataset of product pairs.

    Args:
        pairs_file: Path to product pairs JSONL file
        qdb_file: Path to QDB JSONL file
        top_k: Number of similar entries to retrieve
        similarity_threshold: Minimum similarity for retrieval
        max_pairs: Maximum number of pairs to evaluate (None = all)
        concurrency: Maximum number of concurrent API calls (default: 10)
    """
    logger.info("\n" + "=" * 80)
    logger.info("Deduplication Agent - Retrieval Evaluation")
    logger.info("=" * 80)
    logger.info(f"Pairs file: {pairs_file}")
    logger.info(f"QDB file: {qdb_file}")
    logger.info(f"Top-k: {top_k}")
    logger.info(f"Similarity threshold: {similarity_threshold}")
    logger.info(f"Concurrency: {concurrency}")
    logger.info("=" * 80 + "\n")

    # Load product pairs
    pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    if max_pairs:
        pairs = pairs[:max_pairs]

    logger.info(f"Loaded {len(pairs)} product pairs for evaluation")

    # Check QDB exists
    qdb_path = Path(qdb_file)
    if not qdb_path.exists():
        logger.error(f"QDB file not found: {qdb_file}")
        logger.error(
            "Please run 'python -m scripts.generate_knowledge' first to create QDB"
        )
        return

    # Count QDB entries
    qdb_count = sum(1 for _ in open(qdb_path))
    logger.info(f"QDB contains {qdb_count} reasoning traces\n")

    # Create semaphore and metrics collector
    semaphore = asyncio.Semaphore(concurrency)
    metrics = MetricsCollector()

    # Evaluate all pairs concurrently
    logger.info(f"{'='*80}")
    logger.info("Starting concurrent evaluation...")
    logger.info(f"{'='*80}\n")

    tasks = [
        evaluate_pair(
            pair,
            idx,
            len(pairs),
            qdb_file,
            top_k,
            similarity_threshold,
            semaphore,
            metrics,
        )
        for idx, pair in enumerate(pairs, 1)
    ]

    await asyncio.gather(*tasks)

    # Calculate overall metrics
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*80}")

    logger.info("\nDataset Statistics:")
    logger.info(f"  Total pairs evaluated: {metrics.total_pairs}")
    logger.info(f"  Total questions generated: {metrics.total_questions}")
    logger.info(f"  QDB entries: {qdb_count}")

    logger.info("\nRetrieval Statistics:")
    if metrics.retrieval_counts:
        avg_retrieved = sum(metrics.retrieval_counts) / len(metrics.retrieval_counts)
        logger.info(
            f"  Average entries retrieved per pair: {avg_retrieved:.2f} (top-k={top_k})"
        )
    else:
        logger.info("  Average entries retrieved per pair: 0.00")

    if metrics.similarity_scores:
        avg_similarity = sum(metrics.similarity_scores) / len(metrics.similarity_scores)
        max_similarity = max(metrics.similarity_scores)
        min_similarity = min(metrics.similarity_scores)
        logger.info(f"  Average similarity score: {avg_similarity:.3f}")
        logger.info(f"  Max similarity score: {max_similarity:.3f}")
        logger.info(f"  Min similarity score: {min_similarity:.3f}")

    logger.info("\nReuse Statistics:")
    logger.info(f"  Total reusable answers: {metrics.total_reusable}")
    logger.info(f"  Total fresh searches needed: {metrics.total_fresh_needed}")

    if metrics.total_questions > 0:
        overall_reuse_rate = (metrics.total_reusable / metrics.total_questions) * 100
        logger.info(
            f"  Overall reuse rate: {overall_reuse_rate:.1f}% ({metrics.total_reusable}/{metrics.total_questions} questions)"
        )

        if metrics.reuse_rates:
            avg_pair_reuse = sum(metrics.reuse_rates) / len(metrics.reuse_rates)
            logger.info(f"  Average per-pair reuse rate: {avg_pair_reuse:.1f}%")

    logger.info("\nCost Savings:")
    if metrics.total_questions > 0:
        savings_pct = (metrics.total_reusable / metrics.total_questions) * 100
        logger.info(
            f"  Avoided {metrics.total_reusable} web searches out of {metrics.total_questions} questions"
        )
        logger.info(f"  Cost reduction: ~{savings_pct:.1f}%")
        logger.info(
            "  (Assuming each web search has equal cost, reuse rate directly translates to cost savings)"
        )

    logger.info(f"\n{'='*80}")
    logger.info(
        "Note: Higher reuse rate indicates more effective deduplication and cost savings."
    )
    if metrics.total_questions > 0:
        overall_reuse_rate = (metrics.total_reusable / metrics.total_questions) * 100
    else:
        overall_reuse_rate = 0

    logger.info(f"Final Overall Reuse Rate: {overall_reuse_rate:.1f}%")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate deduplication agent retrieval effectiveness"
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="data/pairs-sample.jsonl",
        help="Path to product pairs JSONL file (default: data/pairs-sample.jsonl)",
    )
    parser.add_argument(
        "--qdb",
        type=str,
        default="data/qdb.jsonl",
        help="Path to QDB JSONL file (default: data/qdb.jsonl)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar entries to retrieve (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum cosine similarity threshold (default: 0.7)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to evaluate (default: all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent API calls (default: 10)",
    )

    args = parser.parse_args()

    asyncio.run(
        evaluate_retrieval(
            args.pairs,
            args.qdb,
            args.top_k,
            args.threshold,
            args.max_pairs,
            args.concurrency,
        )
    )
