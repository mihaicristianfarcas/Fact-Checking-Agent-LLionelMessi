# Task A: Data & Ingestion Module

This module provides the evidence corpus infrastructure for the Fact-Checking LLM Agent.

## Overview

Task A is responsible for:
- **FEVER dataset** (~185k claims) - fact verification with Wikipedia evidence
- **PolitiFact/LIAR dataset** (~5k claims) - political fact-checking
- **Evidence Retriever** - RAG pipeline using ChromaDB + sentence-transformers
- **Ground-truth triples** - claim–evidence–verdict data for training

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets (HuggingFace will cache them)
python -m src.scripts.download_data

# Build the ChromaDB index
python -m src.scripts.build_index

# Validate the corpus
python -m src.scripts.validate_corpus
```

## Module Structure

```
src/
├── data_ingestion/
│   ├── datasets/           # Dataset loaders
│   │   ├── base.py         # Data models (Claim, Evidence, Verdict)
│   │   ├── fever.py        # FEVER dataset loader
│   │   └── politifact.py   # PolitiFact/LIAR loader
│   ├── preprocessing/      # Text processing
│   │   └── text_cleaner.py # Cleaning, chunking, sentence splitting
│   ├── indexing/           # Vector indexing
│   │   ├── embedder.py     # Sentence transformer embeddings
│   │   └── chroma_index.py # ChromaDB operations
│   ├── retriever/          # RAG interface
│   │   └── evidence_retriever.py  # Main API for downstream
│   └── triples/            # Training data generation
│       └── triple_generator.py    # Export claim-evidence-verdict triples
└── config/
    └── settings.py         # Configuration management
```

## API Reference

### EvidenceRetriever

The main interface for downstream components.

```python
from src.data_ingestion import EvidenceRetriever

# Initialize (lazy loads on first query)
retriever = EvidenceRetriever()

# Retrieve evidence for a claim
results = retriever.retrieve("The Earth is approximately 4.5 billion years old.", top_k=5)

for r in results:
    print(f"[{r.score:.3f}] {r.passage.text}")
    print(f"  Source: {r.passage.source}, Dataset: {r.passage.dataset}")

# Filter by dataset
fever_results = retriever.retrieve(query, dataset_filter="fever")

# Batch retrieval (more efficient for multiple queries)
batch_results = retriever.retrieve_batch(["claim1", "claim2", "claim3"])
```

### Data Models

```python
from src.data_ingestion.datasets import Claim, EvidencePassage, Verdict

# Verdict enum
Verdict.SUPPORTED      # Evidence supports the claim
Verdict.REFUTED        # Evidence contradicts the claim  
Verdict.NOT_ENOUGH_INFO  # Insufficient evidence

# Evidence passage
passage = EvidencePassage(
    id="fever_Earth_0",
    text="Earth is the third planet from the Sun.",
    source="Wikipedia:Earth",
    dataset="fever",
    metadata={"sentence_id": 0}
)

# Claim with evidence
claim = Claim(
    id="fever_12345",
    text="The Earth orbits the Sun.",
    verdict=Verdict.SUPPORTED,
    evidence=[passage],
    dataset="fever"
)
```

### Dataset Loaders

```python
from src.data_ingestion.datasets import FeverDataset, PolitifactDataset

# Load FEVER
fever = FeverDataset(split="train")
fever.load()
fever.load_wiki_pages()  # For evidence text

for claim in fever.iter_claims():
    print(f"{claim.verdict}: {claim.text}")

# Load PolitiFact
politifact = PolitifactDataset(split="train", max_samples=1000)
politifact.load()
print(politifact.get_statistics())
```

### Triple Generation

```python
from src.data_ingestion.triples import TripleGenerator

generator = TripleGenerator()
generator.load_fever(split="train", max_samples=10000)
generator.load_politifact(max_samples=5000)

# Export for fine-tuning
generator.export_splits("data/processed/triples/", format="jsonl")
generator.export_parquet("data/processed/triples/all.parquet")

print(generator.get_statistics())
```

### Text Preprocessing

```python
from src.data_ingestion.preprocessing import clean_text, chunk_text, split_sentences

# Clean text
clean = clean_text("<p>Messy  HTML</p>")  # "Messy HTML"

# Chunk for indexing
chunks = chunk_text(long_document, chunk_size=512, chunk_overlap=50)

# Split into sentences
sentences = split_sentences(paragraph, min_length=10)
```

## Configuration

Set via environment variables (prefix `FACTCHECK_`) or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `FACTCHECK_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `FACTCHECK_EMBEDDING_BATCH_SIZE` | `32` | Batch size for embeddings |
| `FACTCHECK_CHROMA_PERSIST_DIR` | `data/index/chroma` | ChromaDB storage |
| `FACTCHECK_CHROMA_COLLECTION_NAME` | `evidence_corpus` | Collection name |
| `FACTCHECK_DEFAULT_TOP_K` | `10` | Default retrieval count |
| `FACTCHECK_CHUNK_SIZE` | `512` | Text chunk size |
| `FACTCHECK_CHUNK_OVERLAP` | `50` | Chunk overlap |

## Output Formats

### Triple JSONL Format (for fine-tuning)

```json
{
  "claim_id": "fever_12345",
  "claim_text": "The Earth is round.",
  "evidence_passages": [
    {
      "id": "fever_Earth_0",
      "text": "Earth is an oblate spheroid...",
      "source": "Wikipedia:Earth",
      "dataset": "fever"
    }
  ],
  "verdict": "SUPPORTED",
  "confidence": 1.0,
  "metadata": {"dataset": "fever", "split": "train"}
}
```

### Retrieval Result Format

```python
RetrievalResult(
    passage=EvidencePassage(...),
    score=0.87,  # Cosine similarity
    rank=1       # 1-indexed position
)
```

## Evaluation Metrics

The validation script measures:
- **Recall@k**: Proportion of relevant evidence in top-k results
- **MRR**: Mean Reciprocal Rank of first relevant result
- **Latency**: Query response time

Target: Recall@10 > 0.8 on FEVER dev set

## Dependencies

Core:
- `sentence-transformers` - Embedding generation
- `chromadb` - Vector database
- `datasets` (HuggingFace) - Dataset loading
- `pandas`, `pyarrow` - Data manipulation

Processing:
- `beautifulsoup4`, `lxml` - HTML cleaning
- `ftfy` - Unicode fixing

See `requirements.txt` for full list.
