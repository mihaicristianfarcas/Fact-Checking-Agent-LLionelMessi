# Task A: Data & Ingestion Module

Owns the evidence corpus, retriever, and ground-truth triples. This document is the
primary reference for anyone consuming Task A outputs — especially **Person B (Claim
Processing)**, who needs the retriever and the triple files.

---

## What is ready

| Deliverable | Location | Notes |
|-------------|----------|-------|
| ChromaDB index | `data/index/chroma/` | 45,371 passages (32,535 FEVER + 12,836 PolitiFact) |
| Training triples | `data/processed/train.jsonl` | 126,628 claim-evidence-verdict triples |
| Val triples | `data/processed/val.jsonl` | 15,828 triples |
| Test triples | `data/processed/test.jsonl` | 15,829 triples |
| EvidenceRetriever | `src/data_ingestion/retriever/` | RAG interface over ChromaDB |

---

### Quick Start (first-time setup)

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Download FEVER + LIAR datasets (HuggingFace caches automatically)
python -m src.scripts.download_data --fever-split train --load-wiki --wiki-limit 10000

# Build ChromaDB index (drop --wiki-limit for the full corpus)
python -m src.scripts.build_index --fever-full-wiki --wiki-limit 10000 --clear

# Sanity-check the index
python -m src.scripts.validate_corpus --sample-queries 20
```

### Retrieve evidence for a claim

```python
from src.data_ingestion import EvidenceRetriever

retriever = EvidenceRetriever()  # embedding model loads lazily on first call

results = retriever.retrieve("Nikolaj Coster-Waldau worked with Fox.", top_k=5)

for r in results:
    print(r.rank, r.score, r.passage.source, r.passage.text)
```

`RetrievalResult` fields:

| Field | Type | Description |
|-------|------|-------------|
| `r.score` | `float` | Cosine similarity, 0–1 |
| `r.rank` | `int` | 1-indexed position in result list |
| `r.passage.text` | `str` | Evidence sentence |
| `r.passage.source` | `str` | Wikipedia page title or `"politifact"` |
| `r.passage.dataset` | `str` | `"fever"` or `"politifact"` |
| `r.passage.id` | `str` | Unique passage ID |
| `r.passage.metadata` | `dict` | `{"sentence_id": int}` for FEVER |

### Batch retrieval (preferred for multiple claims)

```python
claims = ["claim A", "claim B", "claim C"]
batch = retriever.retrieve_batch(claims, top_k=5)

for claim_text, claim_results in zip(claims, batch):
    top = claim_results[0]
    print(f"{claim_text[:60]} -> [{top.score:.3f}] {top.passage.text[:80]}")
```

### Filter by source corpus

```python
# Wikipedia evidence only (higher precision for FEVER-style claims)
results = retriever.retrieve("claim", top_k=5, dataset_filter="fever")

# PolitiFact only
results = retriever.retrieve("claim", top_k=5, dataset_filter="politifact")
```

### What the Stance Classifier receives

Each `EvidencePassage` returned by the retriever maps directly to one item the
Stance Classifier needs to label (supporting / refuting / neutral):

```python
results = retriever.retrieve(claim_text, top_k=10)
passages_to_classify = [(r.passage.text, r.score) for r in results]
# pass passages_to_classify to your stance classifier
```

---

## Training triples (for fine-tuning / Person C)

Each line in `data/processed/train.jsonl` is a JSON object:

```json
{
  "claim_id": "fever_75397",
  "claim_text": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
  "evidence_passages": [
    {
      "id": "fever_Nikolaj_Coster-Waldau_7",
      "text": "He had a recurring role on the short-lived Fox ...",
      "source": "Nikolaj_Coster-Waldau",
      "dataset": "fever",
      "metadata": {"sentence_id": 7}
    }
  ],
  "verdict": "SUPPORTED",
  "confidence": 1.0,
  "metadata": {"dataset": "fever"}
}
```

Verdict values: `"SUPPORTED"`, `"REFUTED"`, `"NOT_ENOUGH_INFO"`.

`confidence` is `1.0` for FEVER (manually verified) and `0.8` for PolitiFact
(editorial judgement, inherently more subjective).

~51% of triples have at least one gold evidence passage. The remaining ~49%
(NOT_ENOUGH_INFO claims and all PolitiFact claims) have `evidence_passages: []`.

---

## Module structure

```
src/data_ingestion/
├── datasets/
│   ├── base.py             # Claim, EvidencePassage, Verdict, ClaimEvidenceTriple
│   ├── fever.py            # FeverDataset — loads fever/fever from HuggingFace
│   └── politifact.py       # PolitifactDataset — loads liar from HuggingFace
├── preprocessing/
│   └── text_cleaner.py     # clean_text(), chunk_text(), split_sentences()
├── indexing/
│   ├── embedder.py         # Embedder (sentence-transformers/all-MiniLM-L6-v2)
│   └── chroma_index.py     # ChromaIndex — add/search/clear/stats
├── retriever/
│   └── evidence_retriever.py  # EvidenceRetriever — the public RAG interface
└── triples/
    └── triple_generator.py    # TripleGenerator — export claim-evidence-verdict files
```

---

## Rebuilding the index from scratch

Only needed if the index is lost or you want a larger corpus:

```bash
# Smoke-test (fast, ~10k wiki pages)
python -m src.scripts.download_data --fever-split train --load-wiki --wiki-limit 10000
python -m src.scripts.build_index --fever-full-wiki --wiki-limit 10000 --clear
python -m src.scripts.validate_corpus --sample-queries 20

# Full corpus (slow, ~5.4M wiki pages)
python -m src.scripts.download_data --fever-split train --load-wiki
python -m src.scripts.build_index --fever-full-wiki --clear
```

## Configuration

Via environment variables (prefix `FACTCHECK_`) or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTCHECK_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `FACTCHECK_EMBEDDING_BATCH_SIZE` | `32` | Embedding batch size |
| `FACTCHECK_CHROMA_PERSIST_DIR` | `data/index/chroma` | ChromaDB storage path |
| `FACTCHECK_CHROMA_COLLECTION_NAME` | `evidence_corpus` | Collection name |
| `FACTCHECK_DEFAULT_TOP_K` | `10` | Default number of results |
| `FACTCHECK_CHUNK_SIZE` | `512` | Text chunk size (chars) |
| `FACTCHECK_CHUNK_OVERLAP` | `50` | Chunk overlap (chars) |
