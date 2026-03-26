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
- ⬜ Keep hallucination rate below 5% — citations must correspond to retrieved passages
- ⬜ Fine-tune a small language model using LoRA on claim–evidence–verdict triples
- ⬜ Apply preference learning (DPO or similar) to reward well-cited, conservative verdicts
- ⬜ Establish a reproducible evaluation baseline on FEVER (accuracy, macro F1, calibration)
- ✅ Keep the system simple enough for a small team to iterate on within an 8-week window

## Datasets

- **[FEVER](https://fever.ai/)** - Fact Extraction and VERification dataset (~185k claims, loaded via `fever/fever` on HuggingFace)
- **[LIAR/PolitiFact](https://huggingface.co/datasets/liar)** - Political fact-checking database (12,836 claims across train/val/test)

## Evaluation Metrics

- **Accuracy** - Overall correctness of verdicts
- **Macro F1** - Balanced performance across verdict classes
- **Calibration** - Confidence scores reflect actual accuracy
- **Hallucination Rate** - Target: < 5%

---

## Task A: Data & Ingestion — Status

> **Done.** Corpus indexed, retriever operational, training triples exported.
> One RFC deliverable outstanding: the evaluation baseline script (FEVER dev-set accuracy/F1/calibration) has not yet been written.

### Corpus at a glance

| Dataset | Claims | Passages in index |
|---------|--------|-------------------|
| FEVER (train) | 145,449 | 32,535 Wikipedia sentences |
| LIAR/PolitiFact | 12,836 | 12,836 claim texts |
| **Total** | **158,285** | **45,371** |

Training triples for fine-tuning are in `data/processed/`:

| File | Triples |
|------|---------|
| `train.jsonl` | 126,628 |
| `val.jsonl` | 15,828 |
| `test.jsonl` | 15,829 |

### Quick Start (first-time setup)

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Download FEVER + LIAR datasets (HuggingFace caches automatically)
python -m src.scripts.download_data --fever-split train --load-wiki --wiki-limit 5000

# Build ChromaDB index (drop --wiki-limit for the full corpus)
python -m src.scripts.build_index --fever-full-wiki --wiki-limit 5000 --clear

# Sanity-check the index
python -m src.scripts.validate_corpus --sample-queries 20
```

### Using the Evidence Retriever (Person B interface)

```python
from src.data_ingestion import EvidenceRetriever

retriever = EvidenceRetriever()  # lazy-loads embedding model on first call

# Retrieve evidence for a single claim
results = retriever.retrieve("Nikolaj Coster-Waldau worked with Fox.", top_k=5)
for r in results:
    print(f"{r.rank}. [{r.score:.3f}] ({r.passage.source}) {r.passage.text[:120]}")

# Filter to FEVER Wikipedia passages only
fever_only = retriever.retrieve("claim text", top_k=5, dataset_filter="fever")

# Batch mode — more efficient for multiple claims
batch = retriever.retrieve_batch(["claim A", "claim B", "claim C"], top_k=5)
# batch[i] is List[RetrievalResult] for claims[i]
```

Each `RetrievalResult` exposes:
- `r.score` — cosine similarity (0–1)
- `r.rank` — 1-indexed position
- `r.passage.text` — evidence sentence
- `r.passage.source` — Wikipedia page title or `"politifact"`
- `r.passage.dataset` — `"fever"` or `"politifact"`

### Project Structure

```
src/
├── data_ingestion/
│   ├── datasets/          # FEVER & PolitiFact loaders
│   ├── preprocessing/     # Text cleaning & chunking
│   ├── indexing/          # ChromaDB indexing pipeline
│   ├── retriever/         # Evidence Retriever (RAG) — main interface
│   └── triples/           # Ground-truth triple generation
├── config/                # Configuration management
└── scripts/               # CLI tools (download_data, build_index, validate_corpus)
data/
├── index/chroma/          # Persisted ChromaDB vector index
└── processed/             # train/val/test JSONL triples
```

### Configuration

Via environment variables (prefix `FACTCHECK_`) or `.env`:

```bash
FACTCHECK_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FACTCHECK_CHROMA_PERSIST_DIR=data/index/chroma
FACTCHECK_DEFAULT_TOP_K=10
```

