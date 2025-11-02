"""
Evaluate product matching with and without knowledge augmentation.

This script compares the effectiveness of knowledge augmentation by evaluating
product matching accuracy in two modes:
1. No Knowledge: Baseline decision using only product descriptions and questions
2. With Knowledge: Enhanced decision using cached knowledge from QDB via dedup agent

Metrics:
- Accuracy: % of correct predictions
- Precision/Recall/F1: For "same SKU" predictions
- Comparison: Improvement from knowledge augmentation
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from q2k.dedup_agent import run_dedup_agent
from q2k.reasoning_agent import run_reasoning_agent
from q2k.utils.llm_helper import invoke_async_with_backoff
from q2k.utils.openai import get_openai_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecisionOutput(BaseModel):
    """Structured output from decision agent."""

    prediction: int  # 0 or 1
    reasoning: str


class EvaluationMetrics:
    """Metrics for a single evaluation mode."""

    def __init__(self):
        self.total = 0
        self.correct = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    def add_result(self, prediction: int, label: int):
        """Add a single prediction result."""
        self.total += 1
        if prediction == label:
            self.correct += 1

        # For "same SKU" (label=1) predictions
        if prediction == 1 and label == 1:
            self.true_positives += 1
        elif prediction == 1 and label == 0:
            self.false_positives += 1
        elif prediction == 0 and label == 0:
            self.true_negatives += 1
        elif prediction == 0 and label == 1:
            self.false_negatives += 1

    def get_accuracy(self) -> float:
        """Calculate accuracy."""
        return (self.correct / self.total * 100) if self.total > 0 else 0.0

    def get_precision(self) -> float:
        """Calculate precision for 'same SKU' predictions."""
        total_predicted_same = self.true_positives + self.false_positives
        return (
            (self.true_positives / total_predicted_same * 100)
            if total_predicted_same > 0
            else 0.0
        )

    def get_recall(self) -> float:
        """Calculate recall for 'same SKU' predictions."""
        total_actual_same = self.true_positives + self.false_negatives
        return (
            (self.true_positives / total_actual_same * 100)
            if total_actual_same > 0
            else 0.0
        )

    def get_f1(self) -> float:
        """Calculate F1 score."""
        precision = self.get_precision()
        recall = self.get_recall()
        return (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )


class MetricsCollector:
    """Thread-safe metrics collector for concurrent evaluation."""

    def __init__(self):
        self.no_knowledge = EvaluationMetrics()
        self.with_knowledge = EvaluationMetrics()
        self.detailed_results = []  # Store detailed results per pair
        self.lock = asyncio.Lock()

    async def add_no_knowledge_result(self, prediction: int, label: int):
        """Add result for no-knowledge mode."""
        async with self.lock:
            self.no_knowledge.add_result(prediction, label)

    async def add_with_knowledge_result(self, prediction: int, label: int):
        """Add result for with-knowledge mode."""
        async with self.lock:
            self.with_knowledge.add_result(prediction, label)

    async def add_detailed_result(self, result: dict[str, Any]):
        """Add detailed result for a pair."""
        async with self.lock:
            self.detailed_results.append(result)


class DecisionAgent:
    """Agent that makes final product matching decision."""

    def __init__(self):
        self.model_name = "gpt-5-mini"
        self.llm = get_openai_model(self.model_name)
        self.decision_model = self.llm.with_structured_output(
            DecisionOutput, include_raw=True
        )

        # Load decision prompt
        prompt_path = Path("prompts/decision.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    async def decide(
        self,
        base_product: str,
        candidate_product: str,
        questions: list[str],
        knowledge: list[dict[str, Any]] = None,
    ) -> tuple[int, str]:
        """
        Make product matching decision.

        Args:
            base_product: Base product description
            candidate_product: Candidate product description
            questions: Generated disambiguation questions
            knowledge: Optional list of Q&A pairs with knowledge

        Returns:
            Tuple of (prediction, reasoning)
        """
        # Format questions
        questions_text = "\n".join([f"{i}. {q}" for i, q in enumerate(questions, 1)])

        # Format knowledge if provided
        knowledge_text = ""
        if knowledge:
            knowledge_text = "\n\n**Retrieved Knowledge:**\n"
            for item in knowledge:
                knowledge_text += f"\n**Q**: {item['new_question']}\n"
                knowledge_text += f"**A**: {item['retrieved_knowledge']}\n"

        # Create human prompt
        human_prompt = f"""**Base Product:**
{base_product}

**Candidate Product:**
{candidate_product}

**Disambiguation Questions:**
{questions_text}
{knowledge_text}

Please analyze whether these two products are the same SKU or different SKUs."""

        messages = [self.system_prompt, human_prompt]

        response = await invoke_async_with_backoff(
            self.decision_model.ainvoke, messages
        )
        decision = response["parsed"]

        return decision.prediction, decision.reasoning


async def evaluate_pair(
    pair: dict[str, Any],
    idx: int,
    total_pairs: int,
    qdb_path: str,
    mode: str,
    semaphore: asyncio.Semaphore,
    metrics: MetricsCollector,
    decision_agent: DecisionAgent,
):
    """
    Evaluate a single product pair in specified mode(s).

    Args:
        pair: Product pair dictionary
        idx: Index of this pair (1-indexed)
        total_pairs: Total number of pairs
        qdb_path: Path to QDB file
        mode: Evaluation mode ("both", "no_knowledge", "with_knowledge")
        semaphore: Semaphore to limit concurrent API calls
        metrics: Shared metrics collector
        decision_agent: Shared decision agent instance
    """
    base = pair["base"]
    candidate = pair["candidate"]
    label = pair.get("label", None)

    if label is None:
        logger.warning(f"Pair {idx}: No label found, skipping")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating Pair {idx}/{total_pairs} (Ground Truth: {label})")
    logger.info(f"Base: {base[:80]}...")
    logger.info(f"Candidate: {candidate[:80]}...")

    # Initialize detailed result dict
    detailed_result = {
        "pair_idx": idx,
        "base_product": base,
        "candidate_product": candidate,
        "ground_truth": label,
        "questions": None,
        "no_knowledge": None,
        "with_knowledge": None,
    }

    try:
        # Step 1: Generate questions using reasoning agent
        async with semaphore:
            reasoning_result = await run_reasoning_agent(base, candidate)

        questions = reasoning_result["questions"]

        if not questions:
            logger.warning(f"Pair {idx}: No questions generated, skipping")
            return

        logger.info(f"Pair {idx}: Generated {len(questions)} questions")
        detailed_result["questions"] = questions

        # Step 2a: No Knowledge Mode
        if mode in ["both", "no_knowledge"]:
            logger.info(f"\nPair {idx}: Running NO KNOWLEDGE mode...")
            async with semaphore:
                prediction_nk, reasoning_nk = await decision_agent.decide(
                    base, candidate, questions, knowledge=None
                )

            await metrics.add_no_knowledge_result(prediction_nk, label)

            detailed_result["no_knowledge"] = {
                "prediction": prediction_nk,
                "reasoning": reasoning_nk,
                "correct": prediction_nk == label,
            }

            logger.info(f"Pair {idx} [No Knowledge]:")
            logger.info(f"  Prediction: {prediction_nk} (Ground Truth: {label})")
            logger.info(f"  Correct: {'' if prediction_nk == label else ''}")
            logger.info(f"  Reasoning: {reasoning_nk[:100]}...")

        # Step 2b: With Knowledge Mode
        if mode in ["both", "with_knowledge"]:
            logger.info(f"\nPair {idx}: Running WITH KNOWLEDGE mode...")

            # Run dedup agent to get cached knowledge
            async with semaphore:
                dedup_result = await run_dedup_agent(
                    questions=questions,
                    qdb_path=qdb_path,
                    top_k=5,
                    similarity_threshold=0.7,
                )

            reusable_answers = dedup_result.reusable_answers
            logger.info(
                f"Pair {idx}: Dedup agent returned {len(reusable_answers)} reusable answers"
            )

            # Make decision with available knowledge (skip questions needing fresh search)
            async with semaphore:
                prediction_wk, reasoning_wk = await decision_agent.decide(
                    base, candidate, questions, knowledge=reusable_answers
                )

            await metrics.add_with_knowledge_result(prediction_wk, label)

            detailed_result["with_knowledge"] = {
                "prediction": prediction_wk,
                "reasoning": reasoning_wk,
                "correct": prediction_wk == label,
                "reusable_answers_count": len(reusable_answers),
                "reusable_answers": reusable_answers,
            }

            logger.info(f"Pair {idx} [With Knowledge]:")
            logger.info(f"  Prediction: {prediction_wk} (Ground Truth: {label})")
            logger.info(f"  Correct: {'' if prediction_wk == label else ''}")
            logger.info(f"  Reasoning: {reasoning_wk[:100]}...")

        # Add detailed result to metrics
        await metrics.add_detailed_result(detailed_result)

    except Exception as e:
        logger.error(f"Pair {idx}: Error evaluating - {e}")


async def evaluate_mapping(
    pairs_file: str,
    qdb_file: str = None,
    max_pairs: int = None,
    concurrency: int = 10,
    mode: str = "both",
    output_file: str = None,
):
    """
    Evaluate product mapping with and without knowledge augmentation.

    Args:
        pairs_file: Path to product pairs JSONL file
        qdb_file: Path to QDB JSONL file (required for with_knowledge mode)
        max_pairs: Maximum number of pairs to evaluate (None = all)
        concurrency: Maximum number of concurrent API calls
        mode: Evaluation mode ("both", "no_knowledge", "with_knowledge")
        output_file: Path to save detailed results as JSONL (None = no output)
    """
    logger.info("\n" + "=" * 80)
    logger.info("Product Mapping Evaluation - Knowledge Augmentation Comparison")
    logger.info("=" * 80)
    logger.info(f"Pairs file: {pairs_file}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Concurrency: {concurrency}")
    if mode in ["both", "with_knowledge"]:
        logger.info(f"QDB file: {qdb_file}")
    logger.info("=" * 80 + "\n")

    # Validate mode
    if mode not in ["both", "no_knowledge", "with_knowledge"]:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'both', 'no_knowledge', or 'with_knowledge'"
        )

    # Check QDB for with_knowledge mode
    if mode in ["both", "with_knowledge"]:
        if not qdb_file:
            raise ValueError("--qdb is required for 'with_knowledge' or 'both' mode")
        qdb_path = Path(qdb_file)
        if not qdb_path.exists():
            logger.error(f"QDB file not found: {qdb_file}")
            logger.error(
                "Please run 'python -m scripts.generate_knowledge' first to create QDB"
            )
            return

        qdb_count = sum(1 for _ in open(qdb_path))
        logger.info(f"QDB contains {qdb_count} reasoning traces\n")

    # Load product pairs
    pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    if max_pairs:
        pairs = pairs[:max_pairs]

    logger.info(f"Loaded {len(pairs)} product pairs for evaluation\n")

    # Create semaphore, metrics collector, and decision agent
    semaphore = asyncio.Semaphore(concurrency)
    metrics = MetricsCollector()
    decision_agent = DecisionAgent()

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
            mode,
            semaphore,
            metrics,
            decision_agent,
        )
        for idx, pair in enumerate(pairs, 1)
    ]

    await asyncio.gather(*tasks)

    # Print results
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*80}")

    logger.info("\nDataset Statistics:")
    logger.info(f"  Total pairs evaluated: {len(pairs)}")

    if mode in ["both", "no_knowledge"]:
        nk_metrics = metrics.no_knowledge
        logger.info(f"\n{'='*80}")
        logger.info("NO KNOWLEDGE MODE (Baseline)")
        logger.info(f"{'='*80}")
        logger.info(f"  Accuracy: {nk_metrics.get_accuracy():.2f}%")
        logger.info(f"  Precision (same SKU): {nk_metrics.get_precision():.2f}%")
        logger.info(f"  Recall (same SKU): {nk_metrics.get_recall():.2f}%")
        logger.info(f"  F1 Score: {nk_metrics.get_f1():.2f}%")
        logger.info("\n  Confusion Matrix:")
        logger.info(f"    True Positives: {nk_metrics.true_positives}")
        logger.info(f"    False Positives: {nk_metrics.false_positives}")
        logger.info(f"    True Negatives: {nk_metrics.true_negatives}")
        logger.info(f"    False Negatives: {nk_metrics.false_negatives}")

    if mode in ["both", "with_knowledge"]:
        wk_metrics = metrics.with_knowledge
        logger.info(f"\n{'='*80}")
        logger.info("WITH KNOWLEDGE MODE (Knowledge Augmentation)")
        logger.info(f"{'='*80}")
        logger.info(f"  Accuracy: {wk_metrics.get_accuracy():.2f}%")
        logger.info(f"  Precision (same SKU): {wk_metrics.get_precision():.2f}%")
        logger.info(f"  Recall (same SKU): {wk_metrics.get_recall():.2f}%")
        logger.info(f"  F1 Score: {wk_metrics.get_f1():.2f}%")
        logger.info("\n  Confusion Matrix:")
        logger.info(f"    True Positives: {wk_metrics.true_positives}")
        logger.info(f"    False Positives: {wk_metrics.false_positives}")
        logger.info(f"    True Negatives: {wk_metrics.true_negatives}")
        logger.info(f"    False Negatives: {wk_metrics.false_negatives}")

    if mode == "both":
        logger.info(f"\n{'='*80}")
        logger.info("COMPARISON")
        logger.info(f"{'='*80}")
        accuracy_improvement = (
            metrics.with_knowledge.get_accuracy() - metrics.no_knowledge.get_accuracy()
        )
        f1_improvement = metrics.with_knowledge.get_f1() - metrics.no_knowledge.get_f1()
        logger.info(f"  Accuracy Improvement: {accuracy_improvement:+.2f}%")
        logger.info(f"  F1 Score Improvement: {f1_improvement:+.2f}%")
        logger.info(
            f"\n  Knowledge augmentation {'improved' if accuracy_improvement > 0 else 'decreased' if accuracy_improvement < 0 else 'maintained'} accuracy by {abs(accuracy_improvement):.2f}%"
        )

    # Save detailed results to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort results by pair_idx
        sorted_results = sorted(metrics.detailed_results, key=lambda x: x["pair_idx"])

        with open(output_path, "w", encoding="utf-8") as f:
            for result in sorted_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info(f"\n{'='*80}")
        logger.info(f"DETAILED RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Saved detailed results to: {output_file}")
        logger.info(f"Total pairs: {len(sorted_results)}")

    # Print summary of incorrect predictions
    if metrics.detailed_results:
        logger.info(f"\n{'='*80}")
        logger.info("PREDICTION ANALYSIS")
        logger.info(f"{'='*80}")

        # Find incorrect predictions
        nk_incorrect = []
        wk_incorrect = []

        for result in metrics.detailed_results:
            if result.get("no_knowledge") and not result["no_knowledge"]["correct"]:
                nk_incorrect.append(result)
            if result.get("with_knowledge") and not result["with_knowledge"]["correct"]:
                wk_incorrect.append(result)

        if mode in ["both", "no_knowledge"]:
            logger.info(f"\nNo Knowledge Mode - Incorrect Predictions: {len(nk_incorrect)}")
            if nk_incorrect:
                for result in nk_incorrect[:10]:  # Show up to 10
                    logger.info(f"\n  Pair {result['pair_idx']}:")
                    logger.info(f"    Ground Truth: {result['ground_truth']}")
                    logger.info(f"    Prediction: {result['no_knowledge']['prediction']}")
                    logger.info(f"    Base: {result['base_product'][:60]}...")
                    logger.info(f"    Candidate: {result['candidate_product'][:60]}...")
                if len(nk_incorrect) > 10:
                    logger.info(f"\n  ... and {len(nk_incorrect) - 10} more (see output file)")

        if mode in ["both", "with_knowledge"]:
            logger.info(f"\nWith Knowledge Mode - Incorrect Predictions: {len(wk_incorrect)}")
            if wk_incorrect:
                for result in wk_incorrect[:10]:  # Show up to 10
                    logger.info(f"\n  Pair {result['pair_idx']}:")
                    logger.info(f"    Ground Truth: {result['ground_truth']}")
                    logger.info(f"    Prediction: {result['with_knowledge']['prediction']}")
                    logger.info(f"    Base: {result['base_product'][:60]}...")
                    logger.info(f"    Candidate: {result['candidate_product'][:60]}...")
                    logger.info(f"    Reusable Answers: {result['with_knowledge']['reusable_answers_count']}")
                if len(wk_incorrect) > 10:
                    logger.info(f"\n  ... and {len(wk_incorrect) - 10} more (see output file)")

    logger.info(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate product mapping with and without knowledge augmentation"
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
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "no_knowledge", "with_knowledge"],
        help="Evaluation mode (default: both)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed results as JSONL (default: None, no output file)",
    )

    args = parser.parse_args()

    asyncio.run(
        evaluate_mapping(
            args.pairs,
            args.qdb,
            args.max_pairs,
            args.concurrency,
            args.mode,
            args.output,
        )
    )
