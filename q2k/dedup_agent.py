"""
Deduplication Agent for Q2K product matching.

This agent retrieves and reuses previously solved reasoning traces to avoid redundant
web searches. It uses semantic similarity search to find relevant cached Q&A pairs,
then uses an LLM to evaluate whether the retrieved knowledge is sufficient to answer
new questions.

From paper:
"The Deduplication Agent coordinates two key operations: (1) retrieving potentially
relevant reasoning traces, and (2) deciding whether these traces provide sufficient
information gain to resolve the current case."
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from q2k.utils.llm_helper import invoke_async_with_backoff
from q2k.utils.openai import create_embeddings, get_openai_model

logger = logging.getLogger(__name__)


class ReusableAnswer(BaseModel):
    """A reusable answer from the knowledge database."""

    new_question_idx: int
    new_question: str
    retrieved_knowledge: str
    source_question: str


class QuestionNeedingSearch(BaseModel):
    """A question that needs fresh knowledge agent search."""

    question_idx: int
    question: str


class DedupEvaluation(BaseModel):
    """LLM evaluation of knowledge sufficiency."""

    reusable_answers: list[ReusableAnswer]
    questions_needing_fresh_search: list[QuestionNeedingSearch]


class DedupResult:
    """Result from deduplication agent."""

    def __init__(
        self,
        reusable_answers: list[dict[str, Any]],
        questions_needing_fresh_search: list[dict[str, Any]],
        retrieved_entries: list[dict[str, Any]],
        similarities: list[float],
    ):
        self.reusable_answers = reusable_answers
        self.questions_needing_fresh_search = questions_needing_fresh_search
        self.retrieved_entries = retrieved_entries
        self.similarities = similarities

    def has_reusable_knowledge(self) -> bool:
        """Check if any knowledge can be reused."""
        return len(self.reusable_answers) > 0

    def needs_fresh_search(self) -> bool:
        """Check if any questions need fresh knowledge search."""
        return len(self.questions_needing_fresh_search) > 0


class DedupAgent:
    """
    Deduplication Agent that retrieves and reuses cached reasoning traces.

    Uses semantic similarity to find relevant Q&A pairs from the QDB, then evaluates
    whether retrieved knowledge is sufficient using an LLM.
    """

    def __init__(
        self, qdb_path: str, top_k: int = 5, similarity_threshold: float = 0.7
    ):
        """
        Initialize the Deduplication Agent.

        Args:
            qdb_path: Path to the Question Database (qdb.jsonl)
            top_k: Number of similar entries to retrieve (default: 5)
            similarity_threshold: Minimum cosine similarity for retrieval (default: 0.7)
        """
        self.qdb_path = Path(qdb_path)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.qdb_entries = []
        self.qdb_embeddings = None

        # Load QDB
        self._load_qdb()

        # Setup LLM for sufficiency evaluation
        self.model_name = "gpt-5-mini"
        self.llm = get_openai_model(self.model_name)
        self.evaluator_model = self.llm.with_structured_output(
            DedupEvaluation, include_raw=True
        )

        # Load dedup prompt
        prompt_path = Path("prompts/dedup.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def _load_qdb(self):
        """Load the Question Database from disk."""
        if not self.qdb_path.exists():
            logger.warning(f"QDB file not found: {self.qdb_path}")
            return

        logger.info(f"Loading QDB from {self.qdb_path}")

        with open(self.qdb_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.qdb_entries.append(entry)

        if not self.qdb_entries:
            logger.warning("QDB is empty")
            return

        # Extract embeddings into numpy array for fast similarity computation
        self.qdb_embeddings = np.array(
            [entry["embedding"] for entry in self.qdb_entries], dtype=np.float32
        )

        logger.info(
            f"Loaded {len(self.qdb_entries)} entries from QDB "
            f"(embedding shape: {self.qdb_embeddings.shape})"
        )

    def _compute_cosine_similarity(self, query_embedding: list[float]) -> np.ndarray:
        """
        Compute cosine similarity between query and all QDB entries.

        Args:
            query_embedding: Query embedding vector

        Returns:
            Array of cosine similarity scores
        """
        query_vec = np.array(query_embedding, dtype=np.float32)

        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        qdb_norms = self.qdb_embeddings / np.linalg.norm(
            self.qdb_embeddings, axis=1, keepdims=True
        )

        # Compute cosine similarity
        similarities = np.dot(qdb_norms, query_norm)

        return similarities

    def _retrieve_top_k(
        self, query_embedding: list[float]
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """
        Retrieve top-k most similar entries from QDB.

        Args:
            query_embedding: Query embedding vector

        Returns:
            Tuple of (top-k entries, similarity scores)
        """
        if not self.qdb_entries:
            return [], []

        # Compute similarities
        similarities = self._compute_cosine_similarity(query_embedding)

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][: self.top_k]

        # Filter by threshold
        filtered_indices = [
            idx
            for idx in top_k_indices
            if similarities[idx] >= self.similarity_threshold
        ]

        if not filtered_indices:
            logger.info(
                f"No entries above similarity threshold {self.similarity_threshold}"
            )
            return [], []

        # Get entries and scores
        top_entries = [self.qdb_entries[idx] for idx in filtered_indices]
        top_scores = [float(similarities[idx]) for idx in filtered_indices]

        logger.info(
            f"Retrieved {len(top_entries)} entries above threshold "
            f"(similarities: {[f'{s:.3f}' for s in top_scores]})"
        )

        return top_entries, top_scores

    async def _evaluate_sufficiency(
        self, new_questions: list[str], retrieved_entries: list[dict[str, Any]]
    ) -> DedupEvaluation:
        """
        Use LLM to evaluate whether retrieved knowledge is sufficient for new questions.

        Args:
            new_questions: List of new questions to answer
            retrieved_entries: List of retrieved QDB entries

        Returns:
            DedupEvaluation with reusable answers and questions needing fresh search
        """
        # Format retrieved knowledge
        retrieved_knowledge_text = ""
        for idx, entry in enumerate(retrieved_entries, 1):
            retrieved_knowledge_text += f"\n### Retrieved Entry {idx}:\n"
            for answer in entry["answers"]:
                retrieved_knowledge_text += f"- **Question**: {answer['question']}\n"
                retrieved_knowledge_text += f"  **Knowledge**: {answer['knowledge']}\n"

        # Format new questions
        new_questions_text = "\n".join(
            [f"{i}. {q}" for i, q in enumerate(new_questions, 1)]
        )

        # Create human prompt
        human_prompt = f"""**New Questions:**
{new_questions_text}

**Retrieved Knowledge from Database:**
{retrieved_knowledge_text}

Please evaluate which new questions can be answered using the retrieved knowledge, and which need fresh searches."""

        messages = [self.system_prompt, human_prompt]

        logger.info("Evaluating knowledge sufficiency with LLM...")
        response = await invoke_async_with_backoff(
            self.evaluator_model.ainvoke, messages
        )

        evaluation = response["parsed"]
        logger.info(
            f"LLM evaluation: {len(evaluation.reusable_answers)} reusable, "
            f"{len(evaluation.questions_needing_fresh_search)} need fresh search"
        )

        return evaluation

    async def deduplicate(self, questions: list[str]) -> DedupResult:
        """
        Main deduplication logic: retrieve and evaluate cached reasoning traces.

        Args:
            questions: List of new disambiguation questions

        Returns:
            DedupResult with reusable knowledge and questions needing fresh search
        """
        if not questions:
            logger.info("No questions provided")
            return DedupResult([], [], [], [])

        if not self.qdb_entries:
            logger.info("QDB is empty, all questions need fresh search")
            return DedupResult(
                [],
                [
                    {"question_idx": i, "question": q}
                    for i, q in enumerate(questions, 1)
                ],
                [],
                [],
            )

        logger.info(f"\n{'='*80}")
        logger.info(f"Dedup Agent: Processing {len(questions)} questions")
        logger.info(f"{'='*80}")

        # Step 1: Concatenate questions in paper format
        concatenated_questions = "; ".join(
            [f"Question{i}: {q}" for i, q in enumerate(questions, 1)]
        )
        logger.info(f"Concatenated questions ({len(concatenated_questions)} chars)")

        # Step 2: Embed concatenated questions
        logger.info("Generating embedding for concatenated questions...")
        query_embedding = await create_embeddings(concatenated_questions)

        # Step 3: Retrieve top-k similar entries
        logger.info(f"Retrieving top-{self.top_k} similar entries from QDB...")
        retrieved_entries, similarities = self._retrieve_top_k(query_embedding)

        if not retrieved_entries:
            logger.info("No similar entries found, all questions need fresh search")
            return DedupResult(
                [],
                [
                    {"question_idx": i, "question": q}
                    for i, q in enumerate(questions, 1)
                ],
                [],
                [],
            )

        # Step 4: Evaluate sufficiency with LLM
        evaluation = await self._evaluate_sufficiency(questions, retrieved_entries)

        # Convert to dict format
        reusable_answers = [
            answer.model_dump() for answer in evaluation.reusable_answers
        ]
        questions_needing_search = [
            q.model_dump() for q in evaluation.questions_needing_fresh_search
        ]

        logger.info(f"\n{'='*80}")
        logger.info("Deduplication Complete!")
        logger.info(f"Reusable answers: {len(reusable_answers)}")
        logger.info(f"Questions needing fresh search: {len(questions_needing_search)}")
        logger.info(f"{'='*80}\n")

        return DedupResult(
            reusable_answers, questions_needing_search, retrieved_entries, similarities
        )


async def run_dedup_agent(
    questions: list[str],
    qdb_path: str = "data/qdb.jsonl",
    top_k: int = 5,
    similarity_threshold: float = 0.7,
) -> DedupResult:
    """
    Run the deduplication agent on a list of questions.

    Args:
        questions: List of disambiguation questions
        qdb_path: Path to the Question Database (default: data/qdb.jsonl)
        top_k: Number of similar entries to retrieve (default: 5)
        similarity_threshold: Minimum cosine similarity (default: 0.7)

    Returns:
        DedupResult with reusable knowledge and questions needing fresh search
    """
    agent = DedupAgent(qdb_path, top_k, similarity_threshold)
    return await agent.deduplicate(questions)
