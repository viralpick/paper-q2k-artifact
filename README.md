# (Q2K) Question-to-Knowledge: Multi-Agent Generation of Inspectable Facts for Product Mapping

This repository provides a reproducible implementation of Q2K, a multi-agent system for product matching that uses question decomposition and knowledge augmentation. The system determines whether two product descriptions refer to the same SKU by generating targeted disambiguation questions and retrieving authoritative information from cached knowledge or web searches.

## Overview

Q2K uses a three-agent architecture:

1. **Reasoning Agent**: Generates disambiguation questions based on the Basic Matching Rule (Brand, Core name, Variant, Specification, Quantity)
2. **Deduplication Agent**: Retrieves and reuses cached reasoning traces to avoid redundant work (~22% reuse rate)
3. **Knowledge Agent**: Retrieves authoritative information via web search when cached knowledge is insufficient

## Setup

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env
```

**Requirements:**

- Python 3.10+
- OpenAI API key (uses `gpt-5-mini` and `text-embedding-3-large`)

## Quick Start

**IMPORTANT**: All scripts must be run using `python -m scripts.<script_name>` from the repository root.

### 1. Generate Question Database (QDB)

Create cached knowledge traces from product pairs:

```bash
# Generate QDB with default settings (20 concurrent API calls)
python -m scripts.generate_knowledge

# Use higher concurrency for faster processing
python -m scripts.generate_knowledge --concurrency 40

# Process specific input/output files
python -m scripts.generate_knowledge --input data/pairs-sample.jsonl --output data/qdb.jsonl
```

**Arguments:**

- `--input`: Path to product pairs JSONL file (default: `data/pairs-sample.jsonl`)
- `--output`: Path to output QDB JSONL file (default: `data/qdb.jsonl`)
- `--concurrency`: Max concurrent API calls (default: `20`)

**What it does**: For each product pair, generates disambiguation questions using the reasoning agent, retrieves answers using the knowledge agent, and stores the results as cached reasoning traces with embeddings.

---

### 2. Evaluate Product Mapping Accuracy

Compare product matching with and without knowledge augmentation:

```bash
# Default: Compare both modes (no knowledge vs with knowledge)
python -m scripts.eval_mapping

# Evaluate specific number of pairs with higher concurrency
python -m scripts.eval_mapping --max-pairs 20 --concurrency 15

# Run only baseline (no knowledge)
python -m scripts.eval_mapping --mode no_knowledge --max-pairs 10

# Run only with knowledge augmentation
python -m scripts.eval_mapping --mode with_knowledge --concurrency 20

# Save detailed results to examine incorrect predictions
python -m scripts.eval_mapping --output results/eval_results.jsonl
```

**Arguments:**

- `--pairs`: Path to product pairs JSONL file (default: `data/pairs-sample.jsonl`)
- `--qdb`: Path to QDB JSONL file (default: `data/qdb.jsonl`, required for `with_knowledge` mode)
- `--max-pairs`: Maximum number of pairs to evaluate (default: all)
- `--concurrency`: Max concurrent API calls (default: `10`)
- `--mode`: Evaluation mode - `both`, `no_knowledge`, or `with_knowledge` (default: `both`)
- `--output`: Path to save detailed results as JSONL (default: None, prints to console only)

**What it does**: Evaluates product matching accuracy by comparing predictions against ground truth labels. Measures accuracy, precision, recall, F1 score, and shows the improvement from knowledge augmentation.

**Metrics:**

- **Accuracy**: % of correct predictions
- **Precision/Recall/F1**: For "same SKU" predictions
- **Confusion Matrix**: TP, FP, TN, FN counts
- **Improvement**: Accuracy gain from knowledge augmentation (when mode=`both`)
- **Prediction Analysis**: Summary of incorrect predictions showing which pairs were wrong

**Detailed Output File** (when `--output` is specified):

Each line in the output JSONL contains:

- `pair_idx`: Pair number
- `base_product`, `candidate_product`: Product descriptions
- `ground_truth`: Ground truth label (0 or 1)
- `questions`: Generated disambiguation questions
- `no_knowledge`: Prediction, reasoning, and correctness for no-knowledge mode
- `with_knowledge`: Prediction, reasoning, correctness, reusable answers count, and full retrieved knowledge

This allows you to examine which pairs each mode predicted incorrectly and understand why.

---

### 3. Evaluate Retrieval Effectiveness

Measure deduplication agent's retrieval quality and cost savings:

```bash
# Default: Evaluate retrieval with 10 concurrent calls
python -m scripts.eval_retrieval

# Use higher concurrency for faster evaluation
python -m scripts.eval_retrieval --concurrency 30

# Evaluate with custom parameters
python -m scripts.eval_retrieval --top-k 10 --threshold 0.75 --max-pairs 20
```

**Arguments:**

- `--pairs`: Path to product pairs JSONL file (default: `data/pairs-sample.jsonl`)
- `--qdb`: Path to QDB JSONL file (default: `data/qdb.jsonl`)
- `--top-k`: Number of similar entries to retrieve (default: `5`)
- `--threshold`: Minimum cosine similarity threshold (default: `0.7`)
- `--max-pairs`: Maximum number of pairs to evaluate (default: all)
- `--concurrency`: Max concurrent API calls (default: `10`)

**What it does**: Evaluates how effectively the deduplication agent retrieves relevant cached knowledge and measures potential cost savings from avoiding redundant web searches.

**Metrics:**

- **Reuse Rate**: % of questions answered from cached knowledge (paper reports ~22%)
- **Retrieval Accuracy**: Quality of similarity matching
- **Average Similarity Score**: Mean cosine similarity of retrieved entries
- **Cost Savings**: Estimated reduction in web search API calls

---

## Directory Structure

```
q2k/
├── README.md
├── CLAUDE.md                    # Documentation for Claude Code
├── LICENSE
├── requirements.txt
├── .env                         # OpenAI API key (create this)
├── data/
│   ├── pairs-sample.jsonl      # Product pairs with labels
│   └── qdb.jsonl               # Question Database (generated)
├── prompts/
│   ├── reasoning.txt           # Reasoning agent prompt
│   ├── knowledge.txt           # Knowledge agent prompt
│   ├── dedup.txt              # Deduplication agent prompt
│   └── decision.txt           # Decision agent prompt
├── q2k/                        # Main package
│   ├── __init__.py
│   ├── reasoning_agent.py     # Generates disambiguation questions
│   ├── knowledge_agent.py     # Retrieves knowledge via web search
│   ├── dedup_agent.py         # Retrieves & reuses cached traces
│   ├── config/
│   │   ├── __init__.py
│   │   └── openai_config.py
│   └── utils/
│       ├── __init__.py
│       ├── llm_helper.py
│       ├── openai.py
│       └── parser.py
└── scripts/
    ├── generate_knowledge.py   # Generate QDB
    ├── eval_mapping.py         # Evaluate product matching
    └── eval_retrieval.py       # Evaluate retrieval effectiveness
```

## Data Formats

### Product Pairs (`data/pairs-sample.jsonl`)

```json
{
  "base": "Product A description in Korean",
  "candidate": "Product B description in Korean",
  "label": 1
}
```

- `label`: `1` = same SKU, `0` = different SKU

### Question Database (`data/qdb.jsonl`)

Generated by `generate_knowledge.py`. One entry per product pair:

```json
{
  "id": "uuid",
  "pair_idx": 1,
  "pair_label": 1,
  "base_product": "...",
  "candidate_product": "...",
  "questions": ["Q1", "Q2", ...],
  "concatenated_questions": "Question1: Q1; Question2: Q2; ...",
  "embedding": [1536-dim vector],
  "answers": [
    {
      "question": "Q1",
      "knowledge": "Answer with citations",
      "question_idx": 1,
      "annotations": [...]
    }
  ],
  "reasoning_thinking": "...",
  "model_name": "gpt-5-mini",
  "created_at": "2025-01-15T..."
}
```

## Architecture Details

### Agent Pipeline

```
Product Pair → Reasoning Agent → Questions → Dedup Agent → Reusable Knowledge
                                                ↓                      ↓
                                        (cache miss)                   ↓
                                                ↓                      ↓
                                      Knowledge Agent → New Evidence   ↓
                                                                       ↓
Product Pair + Questions + Knowledge → Decision Agent → Prediction (0/1)
```

### Key Features

- **Semantic Similarity Search**: Uses cosine similarity on concatenated question embeddings (1536-dim)
- **Partial Knowledge Reuse**: Returns both reusable answers and questions needing fresh search
- **LLM-based Sufficiency Evaluation**: GPT-5-mini evaluates whether retrieved knowledge is sufficient
- **Concurrent Processing**: All scripts support concurrent API calls with semaphore-based rate limiting
- **Structured Outputs**: Pydantic models for type-safe LLM responses

## Model Configuration

- **LLM**: `gpt-5-mini` (OpenAI)
- **Embeddings**: `text-embedding-3-large` (1536 dimensions)
- **Web Search**: Uses OpenAI's `web_search_preview` tool

## Citation

# TODO: Add citation details here.
