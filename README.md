# Automated Fact-Checking LLM Agent

An intelligent multi-step reasoning pipeline that verifies claims by decomposing them into atomic sub-claims, retrieving evidence, and returning confidence-graded verdicts with cited sources.

## Overview

This agent accepts text claims — headlines, quotes, or statistics — and returns a confidence-graded verdict (**Supported**, **Refuted**, or **Not Enough Info**) with cited evidence passages. Rather than relying on a single LLM call, it operates as a multi-step reasoning pipeline where the orchestrating model decides which tool to invoke at each step, accumulates evidence, and determines when enough information exists to commit to a verdict.

**Phase 1 Focus:** Political claims

## Motivation

Misinformation spreads faster than human fact-checkers can respond. Existing automated tools suffer from two failure modes:
- **Over-aggressive classifiers** that produce too many false positives to be actionable
- **LLM-based systems** that generate fluent, confident-sounding verdicts while hallucinating citations

A credible automated fact-checker must be **accurate**, **traceable**, and **calibrated** — expressing appropriate uncertainty when evidence is thin rather than confabulating support.

> **Design Principle:** Abstention over fabrication. The agent is trained to prefer returning "Not Enough Info" when evidence is insufficient, rather than committing to unsupported verdicts.

## Architecture

The system operates as a tool-augmented LLM pipeline with five discrete tools:

| Tool | Description |
|------|-------------|
| **Claim Decomposer** | Breaks compound claims into atomic sub-claims |
| **Evidence Retriever** | Performs RAG over an indexed corpus |
| **Source Credibility Scorer** | Rates the reliability of each source |
| **Stance Classifier** | Labels each passage as supporting, refuting, or neutral |
| **Verdict Synthesizer** | Aggregates signals into a final output with confidence score |

## Goals

- ✅ Decompose and verify compound claims against an indexed evidence corpus (FEVER + PolitiFact)
- ✅ Return confidence-graded verdicts with source attribution for each sub-claim
- ✅ Keep hallucination rate below 5% — citations must correspond to retrieved passages
- ✅ Fine-tune a small language model using LoRA on claim–evidence–verdict triples
- ✅ Apply preference learning (DPO or similar) to reward well-cited, conservative verdicts
- ✅ Establish a reproducible evaluation baseline on FEVER (accuracy, macro F1, calibration)
- ✅ Keep the system simple enough for a small team to iterate on within an 8-week window

## Datasets

- **[FEVER](https://fever.ai/)** - Fact Extraction and VERification dataset (~185k claims)
- **[PolitiFact/LIAR](https://www.politifact.com/)** - Political fact-checking database (~5k claims)

## Evaluation Metrics

- **Accuracy** - Overall correctness of verdicts
- **Macro F1** - Balanced performance across verdict classes
- **Calibration** - Confidence scores reflect actual accuracy
- **Hallucination Rate** - Target: < 5%

---

## Task A: Data & Ingestion Module

### Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download datasets and build index
python -m src.scripts.download_data
python -m src.scripts.build_index

# Validate the corpus
python -m src.scripts.validate_corpus
```

### Using the Evidence Retriever

```python
from src.data_ingestion import EvidenceRetriever

# Initialize retriever (lazy loads on first query)
retriever = EvidenceRetriever()

# Retrieve evidence for a claim
results = retriever.retrieve("The Earth is round", top_k=5)
for r in results:
    print(f"{r.rank}. [{r.score:.3f}] {r.passage.text[:100]}...")

# Filter by dataset
fever_results = retriever.retrieve("claim text", dataset_filter="fever")

# Batch retrieval for efficiency
batch_results = retriever.retrieve_batch(["claim1", "claim2", "claim3"])
```

### Project Structure

```
src/
├── data_ingestion/
│   ├── datasets/          # FEVER & PolitiFact loaders
│   ├── preprocessing/     # Text cleaning & chunking
│   ├── indexing/          # ChromaDB indexing pipeline
│   ├── retriever/         # Evidence Retriever (RAG)
│   └── triples/           # Ground-truth triple generation
├── config/                # Configuration management
└── scripts/               # CLI tools
```

### Configuration

Configure via environment variables (prefix `FACTCHECK_`) or `.env` file:

```bash
FACTCHECK_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FACTCHECK_CHROMA_PERSIST_DIR=data/index/chroma
FACTCHECK_DEFAULT_TOP_K=10
```

