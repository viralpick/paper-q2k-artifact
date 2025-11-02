"""
Generate Question Database (QDB) for product matching.

This script:
1. Reads product pairs from data/pairs-sample.jsonl
2. For each pair, runs the reasoning agent to generate disambiguation questions
3. For each question, runs the knowledge agent to retrieve answers
4. Concatenates questions per pair and generates embeddings for retrieval
5. Saves all reasoning traces to data/qdb.jsonl (one entry per product pair)

The resulting QDB will be used by the deduplication agent to avoid redundant
web searches by retrieving and reusing similar past reasoning traces.

QDB Structure (from paper):
Q_DB = {(Q_1, A_1), (Q_2, A_2), ..., (Q_j, A_j)}
where Q_i is concatenated questions and A_i is the answer set.

Supports concurrent processing with --concurrency flag for faster execution.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from q2k.knowledge_agent import run_knowledge_agent
from q2k.reasoning_agent import run_reasoning_agent
from q2k.utils.openai import create_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenCounter:
    """Thread-safe token counter for concurrent operations."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.lock = asyncio.Lock()

    async def add(self, input_tokens: int, output_tokens: int):
        async with self.lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def get_totals(self) -> tuple[int, int]:
        return self.input_tokens, self.output_tokens


async def process_pair(
    pair: dict[str, Any],
    idx: int,
    total_pairs: int,
    semaphore: asyncio.Semaphore,
    token_counter: TokenCounter,
) -> dict[str, Any] | None:
    """
    Process a single product pair to generate a QDB entry.

    Args:
        pair: Product pair dictionary with 'base', 'candidate', and 'label'
        idx: Index of this pair (1-indexed for display)
        total_pairs: Total number of pairs being processed
        semaphore: Semaphore to limit concurrent API calls
        token_counter: Shared token counter

    Returns:
        Single QDB entry containing concatenated questions and answer set,
        or None if processing failed
    """
    base_product = pair["base"]
    candidate_product = pair["candidate"]
    pair_label = pair.get("label", None)

    logger.info(f"\n{'='*80}")
    logger.info(f"Processing pair {idx}/{total_pairs}")
    logger.info(f"Base: {base_product[:100]}...")
    logger.info(f"Candidate: {candidate_product[:100]}...")

    try:
        # Step 1: Run reasoning agent to get questions
        async with semaphore:
            reasoning_result = await run_reasoning_agent(
                base_product, candidate_product
            )

        questions = reasoning_result["questions"]
        thinking = reasoning_result["thinking"]

        await token_counter.add(
            reasoning_result["input_tokens"], reasoning_result["output_tokens"]
        )

        logger.info(f"Pair {idx}: Reasoning agent generated {len(questions)} questions")
        if thinking:
            logger.info(f"Pair {idx}: Thinking: {thinking[:200]}...")

        if not questions:
            logger.info(f"Pair {idx}: No questions generated, skipping")
            return None

        # Step 2: Process all questions for this pair concurrently
        model_input = f"base product: {base_product} / candidate: {candidate_product}"
        task_description = "Product matching: Determine if two product descriptions refer to the same SKU"

        async def process_question(q_idx: int, question: str) -> dict[str, Any] | None:
            """Process a single question with knowledge agent."""
            try:
                logger.info(
                    f"Pair {idx}, Question {q_idx}/{len(questions)}: {question[:80]}..."
                )

                # Run knowledge agent
                async with semaphore:
                    knowledge_item, k_input_tokens, k_output_tokens = (
                        await run_knowledge_agent(
                            model_input=model_input,
                            question=question,
                            task_description=task_description,
                        )
                    )

                await token_counter.add(k_input_tokens, k_output_tokens)

                logger.info(
                    f"Pair {idx}, Q{q_idx}: Knowledge retrieved ({len(knowledge_item.knowledge)} chars)"
                )

                # Return Q&A pair (no individual embedding needed now)
                return {
                    "question": question,
                    "knowledge": knowledge_item.knowledge,
                    "question_idx": q_idx,
                    "annotations": knowledge_item.additional_info.get(
                        "annotations", []
                    ),
                }

            except Exception as e:
                logger.error(f"Pair {idx}, Q{q_idx}: Error processing question: {e}")
                return None

        # Process all questions concurrently
        question_tasks = [
            process_question(q_idx, question)
            for q_idx, question in enumerate(questions, 1)
        ]
        results = await asyncio.gather(*question_tasks)

        # Filter out None results
        answers = [entry for entry in results if entry is not None]

        if not answers:
            logger.info(f"Pair {idx}: No answers retrieved, skipping")
            return None

        # Step 3: Concatenate questions in paper format
        concatenated_questions = "; ".join(
            [f"Question{i}: {q}" for i, q in enumerate(questions, 1)]
        )

        logger.info(
            f"Pair {idx}: Concatenated questions ({len(concatenated_questions)} chars)"
        )

        # Step 4: Generate embedding for concatenated questions
        logger.info(f"Pair {idx}: Generating embedding for concatenated questions...")
        async with semaphore:
            embedding = await create_embeddings(concatenated_questions)

        # Step 5: Create single QDB entry per pair
        qdb_entry = {
            "id": str(uuid4()),
            "pair_idx": idx,
            "pair_label": pair_label,
            "base_product": base_product,
            "candidate_product": candidate_product,
            "questions": questions,
            "concatenated_questions": concatenated_questions,
            "embedding": embedding,
            "answers": answers,
            "reasoning_thinking": thinking,
            "model_name": "gpt-5-mini",
            "created_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Pair {idx}: Generated QDB entry with {len(questions)} questions and {len(answers)} answers"
        )

        return qdb_entry

    except Exception as e:
        logger.error(f"Pair {idx}: Error processing pair: {e}")
        return None


async def generate_qdb(input_file: str, output_file: str, concurrency: int = 20):
    """
    Generate the Question Database from product pairs with concurrent processing.

    Args:
        input_file: Path to the input JSONL file containing product pairs
        output_file: Path to the output JSONL file for the QDB
        concurrency: Maximum number of concurrent API calls (default: 20)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read all product pairs
    pairs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    logger.info(f"Loaded {len(pairs)} product pairs from {input_file}")
    logger.info(f"Using concurrency: {concurrency}")

    # Create semaphore and token counter
    semaphore = asyncio.Semaphore(concurrency)
    token_counter = TokenCounter()

    # Process all pairs concurrently
    logger.info(f"\n{'='*80}")
    logger.info("Starting concurrent processing...")
    logger.info(f"{'='*80}\n")

    tasks = [
        process_pair(pair, idx, len(pairs), semaphore, token_counter)
        for idx, pair in enumerate(pairs, 1)
    ]

    results = await asyncio.gather(*tasks)

    # Filter out None results
    qdb_entries = [entry for entry in results if entry is not None]

    # Write all entries to output file
    logger.info(f"\n{'='*80}")
    logger.info(f"Writing {len(qdb_entries)} QDB entries to {output_file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in qdb_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Get token totals
    total_input_tokens, total_output_tokens = token_counter.get_totals()

    # Calculate statistics
    total_questions = sum(len(entry["questions"]) for entry in qdb_entries)
    total_answers = sum(len(entry["answers"]) for entry in qdb_entries)

    logger.info(f"\n{'='*80}")
    logger.info("QDB Generation Complete!")
    logger.info(f"Total QDB entries (pairs): {len(qdb_entries)}")
    logger.info(f"Total questions: {total_questions}")
    logger.info(f"Total answers: {total_answers}")
    logger.info(f"Total input tokens: {total_input_tokens:,}")
    logger.info(f"Total output tokens: {total_output_tokens:,}")
    logger.info(f"Total tokens: {total_input_tokens + total_output_tokens:,}")
    logger.info(f"Output saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Question Database (QDB) for product matching"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/pairs-sample.jsonl",
        help="Path to input JSONL file with product pairs (default: data/pairs-sample.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/qdb.jsonl",
        help="Path to output JSONL file for QDB (default: data/qdb.jsonl)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum number of concurrent API calls (default: 20)",
    )

    args = parser.parse_args()

    asyncio.run(generate_qdb(args.input, args.output, args.concurrency))
