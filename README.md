# Automated Fact-Checking LLM Agent

An intelligent multi-step reasoning pipeline that verifies claims by decomposing them into atomic sub-claims, retrieving evidence, and returning confidence-graded verdicts with cited sources.

## Overview

This agent accepts text claims — headlines, quotes, or statistics — and returns a confidence-graded verdict (**Supported**, **Refuted**, or **Not Enough Info**) with cited evidence passages. Rather than relying on a single LLM call, it operates as a multi-step reasoning pipeline where the orchestrating model decides which tool to invoke at each step, accumulates evidence, and determines when enough information exists to commit to a verdict.

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
- ✅ Keep hallucination rate below 5% — 0.0% on the 500-claim decontaminated FEVER eval
- ✅ Fine-tune a small language model using LoRA on claim–evidence–verdict triples
- ✅ Apply preference learning (DPO or similar) to reward well-cited, conservative verdicts
- ✅ Establish a reproducible evaluation baseline on FEVER (accuracy, macro F1, calibration)
- ✅ Keep the system simple enough for a small team to iterate on within an 8-week window

## RFC Track Status

| Track | Owner Scope | Status | What is implemented | Remaining proof point |
|-------|-------------|--------|---------------------|-----------------------|
| **A — Data & Ingestion** | FEVER + PolitiFact corpus, indexing, RAG retriever, triples | ✅ Done | Dataset loaders, text cleaning, Chroma index builder, retriever, JSONL triples | Full index is local-only and must be built by each user |
| **B — Claim Processing** | Claim decomposer + stance classifier | ✅ Done | Ollama-based decomposer with fallback, DeBERTa NLI stance classifier, mocked and regression tests | Live Ollama improves compound-claim handling; fallback keeps eval running |
| **C — Model Training** | LoRA SFT + DPO preference alignment | ✅ Done | SFT/DPO scripts, prompt formatting, HF adapter inference wrapper with citation guardrail | Raw adapter alone is not trusted without guardrail |
| **D — Scoring, Synthesis & Evaluation** | Credibility scorer, verdict synthesis, agent loop, eval harness | ✅ Done | Credibility scorer, synthesizer with citation tracing, orchestrator, FEVER pipeline eval, offline prediction eval | Verdict quality needs improvement, especially SUPPORTED recall |

Bottom line: all four RFC tracks are implemented and testable. The current system is citation-safe on the measured eval, but verdict classification is still modest and biased toward `REFUTED`.

## Latest Results

### Trained FEVER Verifier Result (Best Pipeline)

The final best pipeline keeps the existing RAG stack and citation guardrails,
then overrides the final verdict with a locally trained DeBERTa verifier:

```text
claim + retrieved evidence passages -> SUPPORTED / REFUTED / NOT_ENOUGH_INFO
```

The verifier is trained on FEVER `train` only. FEVER `labelled_dev` is reserved
for evaluation, and evaluation still uses `--exclude-train-overlap`.

Best calibrated 500-claim result:

| Pipeline | Accuracy | Macro F1 | ECE | Hallucinated citations | Missing citations |
|----------|----------|----------|-----|------------------------|-------------------|
| Baseline RAG + NLI synthesis | 0.618 | 0.597 | 0.130 | 0.0% | 0.0% |
| Trained DeBERTa verifier, 20k fast dataset | 0.698 | 0.692 | 0.183 | 0.0% | 0.0% |
| Calibrated trained verifier, 20k fast dataset | **0.704** | **0.701** | 0.185 | **0.0%** | **0.0%** |

Final model directory:

```text
models/fever_verifier_deberta_base_20k_fast
```

Final result files:

```text
data/processed/pipeline_eval_500_baseline.json
data/processed/pipeline_eval_500_baseline_errors.json
data/processed/pipeline_eval_500_verifier_20k_fast.json
data/processed/pipeline_eval_500_verifier_20k_fast_errors.json
data/processed/pipeline_eval_500_verifier_20k_calibrated.json
data/processed/pipeline_eval_500_verifier_20k_calibrated_errors.json
```

Interpretation: the trained verifier improves the same 500-claim evaluation by
`+8.6` accuracy points and `+10.4` macro-F1 points over the RAG/NLI baseline,
while preserving `0.0%` citation hallucination because citations are still
restricted to retrieved passage IDs.

### Reproducing The Trained Verifier

Use a CUDA machine when possible. The dataset build is retrieval-bound and took
about 31 minutes for 20k claims with the fast settings below; training took
about 5.5 minutes on the RTX 4070 setup used for the final run.

Check CUDA:

```bat
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Build the FEVER title index once:

```bat
python -m src.scripts.build_fever_title_index ^
  --output data/index/fever_titles.sqlite
```

Build the verifier dataset from FEVER train only:

```bat
python -m src.scripts.build_fever_verifier_dataset ^
  --max-claims 20000 ^
  --title-index-path data/index/fever_titles.sqlite ^
  --output data/processed/fever_verifier_train_20k_fast.jsonl ^
  --dev-output data/processed/fever_verifier_dev_2k_fast.jsonl ^
  --dev-size 2000 ^
  --top-k 3 ^
  --candidate-k 10 ^
  --title-candidate-pages 3 ^
  --title-candidate-k 10 ^
  --retrieval-batch-size 256 ^
  --max-passages 3
```

Train the verifier:

```bat
python -m src.model_training.train_fever_verifier ^
  --train-file data/processed/fever_verifier_train_20k_fast.jsonl ^
  --eval-file data/processed/fever_verifier_dev_2k_fast.jsonl ^
  --model-name microsoft/deberta-v3-base ^
  --output-dir models/fever_verifier_deberta_base_20k_fast ^
  --max-length 384 ^
  --batch-size 8 ^
  --gradient-accumulation-steps 2 ^
  --epochs 2 ^
  --learning-rate 2e-5 ^
  --class-weighting ^
  --fp16
```

Expected verifier dev result from the final run:

```text
eval_accuracy = 0.819
eval_macro_f1 = 0.795
```

Run the no-verifier 500-claim baseline:

```bat
python -m src.scripts.evaluate_pipeline ^
  --max-claims 500 ^
  --top-k 5 ^
  --exclude-train-overlap ^
  --skip-decomposition ^
  --enable-title-retrieval ^
  --enable-reranker ^
  --candidate-k 50 ^
  --title-index-path data/index/fever_titles.sqlite ^
  --include-source-title-in-stance ^
  --combine-same-source-evidence ^
  --trace-errors-output data/processed/pipeline_eval_500_baseline_errors.json ^
  --output data/processed/pipeline_eval_500_baseline.json
```

Run the calibrated trained-verifier 100-claim check:

```bat
python -m src.scripts.evaluate_pipeline ^
  --max-claims 100 ^
  --top-k 5 ^
  --exclude-train-overlap ^
  --skip-decomposition ^
  --enable-title-retrieval ^
  --enable-reranker ^
  --candidate-k 50 ^
  --title-index-path data/index/fever_titles.sqlite ^
  --include-source-title-in-stance ^
  --combine-same-source-evidence ^
  --use-trained-verifier ^
  --verifier-model-path models/fever_verifier_deberta_base_20k_fast ^
  --verifier-max-passages 3 ^
  --verifier-supported-threshold 0.70 ^
  --verifier-refuted-threshold 0.55 ^
  --verifier-min-margin 0.08 ^
  --ensemble-baseline-refute-fallback ^
  --ensemble-baseline-refute-threshold 0.82 ^
  --ensemble-verifier-refute-prob-threshold 0.30 ^
  --trace-errors-output data/processed/pipeline_eval_100_verifier_20k_calibrated_errors.json ^
  --output data/processed/pipeline_eval_100_verifier_20k_calibrated.json
```

Run the final calibrated 500-claim evaluation:

```bat
python -m src.scripts.evaluate_pipeline ^
  --max-claims 500 ^
  --top-k 5 ^
  --exclude-train-overlap ^
  --skip-decomposition ^
  --enable-title-retrieval ^
  --enable-reranker ^
  --candidate-k 50 ^
  --title-index-path data/index/fever_titles.sqlite ^
  --include-source-title-in-stance ^
  --combine-same-source-evidence ^
  --use-trained-verifier ^
  --verifier-model-path models/fever_verifier_deberta_base_20k_fast ^
  --verifier-max-passages 3 ^
  --verifier-supported-threshold 0.70 ^
  --verifier-refuted-threshold 0.55 ^
  --verifier-min-margin 0.08 ^
  --ensemble-baseline-refute-fallback ^
  --ensemble-baseline-refute-threshold 0.82 ^
  --ensemble-verifier-refute-prob-threshold 0.30 ^
  --trace-errors-output data/processed/pipeline_eval_500_verifier_20k_calibrated_errors.json ^
  --output data/processed/pipeline_eval_500_verifier_20k_calibrated.json
```

These results were run locally in the `llms` conda environment.

### Test Results

| Check | Command | Result |
|-------|---------|--------|
| Core component tests | `pytest tests/test_evaluation_metrics.py tests/test_model_prompting.py tests/test_synthesis.py tests/test_orchestrator.py tests/test_claim_processing.py tests/test_scoring.py -v` | `62 passed` |
| Prompt + guardrail tests | `pytest tests/test_model_prompting.py -v` | `7 passed` |
| Guarded HF adapter inference | `RUN_MODEL_INFERENCE_TESTS=1 pytest tests/test_model_inference.py -v -s` | `5 passed` |
| Decomposer regression tests | targeted `TestClaimDecomposer` tests | `3 passed` |
| Retrieval diagnostics tests | `pytest tests/test_fever_title_retriever.py tests/test_hybrid_retriever.py tests/test_pipeline_diagnostics.py -q` | `9 passed` |
| Source-aware retrieval/synthesis tests | `pytest tests/test_fever_title_retriever.py tests/test_hybrid_retriever.py tests/test_scoring.py tests/test_synthesis.py tests/test_pipeline_diagnostics.py -q` | `33 passed` |
| Expanded local suite | `pytest tests/test_evaluation_metrics.py tests/test_model_prompting.py tests/test_synthesis.py tests/test_orchestrator.py tests/test_claim_processing.py tests/test_scoring.py tests/test_retriever.py tests/test_fever_title_retriever.py tests/test_hybrid_retriever.py tests/test_pipeline_diagnostics.py -q` | `77 passed, 2 skipped` |

Note: the raw published DPO adapter initially failed the irrelevant-evidence abstention smoke test by returning `SUPPORTED` for an Everest claim with Mariana Trench evidence. Production inference therefore uses `generate_verdict()`, which adds a deterministic citation/evidence guardrail. Use `generate_text()` only when intentionally inspecting the raw model completion.

### Corpus Validation

Command:

```bash
python -m src.scripts.validate_corpus --sample-queries 20
```

Result:

```text
Validation PASSED
Total passages: 662,806
Queries with results: 20/20
Average top score: 0.539
```

This is the FEVER Wikipedia chunk index. The documented full corpus also includes the PolitiFact/LIAR claim passages when that indexing step is included.

### Retrieval Baseline

Command:

```bash
python -m src.scripts.evaluate_baseline \
  --max-claims 500 \
  --top-k 10 \
  --output data/processed/retrieval_baseline_500_full_index.json
```

Result:

| Metric | Value |
|--------|-------|
| Recall@1 | 0.015 |
| Recall@5 | 0.023 |
| Recall@10 | 0.023 |
| Baseline accuracy | 0.344 |
| Baseline macro F1 | 0.264 |

This is the main bottleneck. Page-level retrieval recall is very low: only `8/343` claims with gold evidence had the correct Wikipedia page in the top 10. If the retriever does not surface the right evidence, the stance classifier and synthesizer cannot recover the correct verdict reliably.

Additional page-presence diagnostic: in the first 500 decontaminated FEVER dev claims, only `14/343` claims with gold evidence had any gold Wikipedia page present in the current Chroma index (`4.1%`). This means the first accuracy lever is full-FEVER title/page retrieval, not just reranking the existing vector results.

Hybrid retrieval after building `data/index/fever_titles.sqlite`:

```bash
python -m src.scripts.evaluate_baseline \
  --max-claims 500 \
  --top-k 10 \
  --candidate-k 100 \
  --recall-ks 1,5,10,50,100 \
  --retrieval-mode hybrid \
  --title-index-path data/index/fever_titles.sqlite \
  --output data/processed/retrieval_baseline_500_hybrid.json
```

| Metric | Value |
|--------|-------|
| Recall@1 | 0.694 |
| Recall@5 | 0.848 |
| Recall@10 | 0.857 |
| Recall@50 | 0.866 |
| Recall@100 | 0.869 |

### Full Pipeline Eval

Command:

```bash
python -m src.scripts.evaluate_pipeline \
  --max-claims 500 \
  --top-k 5 \
  --exclude-train-overlap \
  --output data/processed/pipeline_eval_500_full_index.json
```

Result on FEVER `labelled_dev`:

| Metric | Value |
|--------|-------|
| Claims evaluated | 500 |
| Exact train overlaps excluded | 2 |
| Accuracy | 0.380 |
| Macro F1 | 0.318 |
| ECE | 0.243 |
| Hallucinated citations | 0.0% |
| Missing citations | 0.0% |
| Hallucination target `<5%` | Met |

Per-class results:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| `SUPPORTED` | 0.750 | 0.075 | 0.136 | 161 |
| `REFUTED` | 0.428 | 0.714 | 0.535 | 182 |
| `NOT_ENOUGH_INFO` | 0.267 | 0.306 | 0.285 | 157 |

Top-k sensitivity:

| Pipeline setting | Accuracy | Macro F1 | ECE | Hallucination rate |
|------------------|----------|----------|-----|--------------------|
| Full index, `top_k=5` | 0.380 | 0.318 | 0.243 | 0.0% |
| Full index, `top_k=10` | 0.386 | 0.305 | 0.300 | 0.0% |
| Smaller 36k index, `top_k=5` | 0.418 | 0.333 | 0.243 | 0.0% |
| Full index + FEVER title retrieval, `top_k=5`, `candidate_k=50` | 0.448 | 0.388 | 0.234 | 0.0% |
| Full index + FEVER title retrieval + `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, `top_k=5`, `candidate_k=50`, decomposition skipped | 0.510 | 0.488 | 0.217 | 0.0% |
| Source relevance + source-title stance context + same-source evidence windows, `top_k=5`, `candidate_k=50`, decomposition skipped, 100-claim smoke | 0.690 | 0.672 | 0.128 | 0.0% |
| Conservative refute tuning, title retrieval, decomposition skipped | 0.342 | 0.221 | 0.321 | 0.0% |

Best current run:

```bash
python -m src.scripts.evaluate_pipeline \
  --max-claims 500 \
  --top-k 5 \
  --exclude-train-overlap \
  --skip-decomposition \
  --enable-title-retrieval \
  --enable-reranker \
  --candidate-k 50 \
  --title-index-path data/index/fever_titles.sqlite \
  --output data/processed/pipeline_eval_500_title_rerank.json
```

Per-class results for the best run:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| `SUPPORTED` | 0.837 | 0.255 | 0.390 | 161 |
| `REFUTED` | 0.575 | 0.736 | 0.646 | 182 |
| `NOT_ENOUGH_INFO` | 0.367 | 0.510 | 0.427 | 157 |

Interpretation: citation faithfulness remains strong, and hybrid title retrieval plus reranking raises retrieval Recall@10 from `0.023` to `0.857` and pipeline macro F1 from `0.318` to `0.488`. The remaining bottleneck is no longer page retrieval; it is stance/synthesis behavior, especially low `SUPPORTED` recall and over-predicting `REFUTED` for `NOT_ENOUGH_INFO` claims. A very conservative refutation configuration overcorrected into `NOT_ENOUGH_INFO` and should not be promoted.

Source-aware follow-up: retrieval candidates now carry `source_relevance`, derived from title match quality, title score, reranker score, retrieval method, and page-title penalties for low-signal pages such as lists, indexes, discographies, and disambiguation pages. Credibility scoring downweights weak-source refutations more strongly than support, and synthesis blocks fuzzy-title refutations from overriding stronger-source support unless the refutation has high source/rerank quality. A 25-claim smoke run with title retrieval + reranker + source relevance reached `0.680` accuracy, `0.687` macro F1, and `0.0%` hallucinated/missing citations; this is only a fast smoke sample and should not replace the 500-claim result.

Stance-context follow-up: the pipeline can now prepend page titles to stance-classifier premises and classify compact same-source evidence windows while keeping citations restricted to original retrieved passage IDs. On the first 100 decontaminated FEVER dev claims, this reached `0.690` accuracy, `0.672` macro F1, `0.128` ECE, and `0.0%` hallucinated/missing citations. The remaining weakness is `NOT_ENOUGH_INFO` recall: 14/26 NEI claims still became `REFUTED`.

### Retrieval Accuracy Iteration

Build the full FEVER title lookup index. This stores page titles and FEVER wiki row offsets only; it does not use dev labels or evidence annotations:

```bash
python -m src.scripts.build_fever_title_index \
  --output data/index/fever_titles.sqlite
```

Run deeper dense retrieval diagnostics:

```bash
python -m src.scripts.evaluate_baseline \
  --max-claims 500 \
  --top-k 10 \
  --candidate-k 100 \
  --recall-ks 1,5,10,50,100 \
  --output data/processed/retrieval_baseline_500_candidates.json \
  --trace-output data/processed/retrieval_baseline_500_trace.json
```

Run title-only or hybrid retrieval diagnostics:

```bash
python -m src.scripts.evaluate_baseline \
  --max-claims 500 \
  --top-k 10 \
  --candidate-k 100 \
  --recall-ks 1,5,10,50,100 \
  --retrieval-mode hybrid \
  --title-index-path data/index/fever_titles.sqlite \
  --output data/processed/retrieval_baseline_500_hybrid.json
```

Run the full pipeline with title candidates and cross-encoder reranking:

```bash
python -m src.scripts.evaluate_pipeline \
  --max-claims 500 \
  --top-k 5 \
  --exclude-train-overlap \
  --enable-title-retrieval \
  --enable-reranker \
  --candidate-k 50 \
  --title-index-path data/index/fever_titles.sqlite \
  --include-source-title-in-stance \
  --combine-same-source-evidence \
  --trace-errors-output data/processed/pipeline_eval_500_hybrid_errors.json \
  --output data/processed/pipeline_eval_500_hybrid_rerank.json
```

Each run now records the retrieval mode, Chroma gold-page presence rate, embedding model defaults, reranker model, candidate depth, title candidate settings, train-overlap exclusion, and any stance/synthesis threshold overrides in the output JSON.

## Datasets

- **[FEVER](https://fever.ai/)** - Fact Extraction and VERification dataset (~185k claims, loaded via `fever/fever` on HuggingFace)
- **[LIAR/PolitiFact](https://huggingface.co/datasets/liar)** - Political fact-checking database (12,836 claims across train/val/test)

## Evaluation Metrics

- **Accuracy** - Overall correctness of verdicts
- **Macro F1** - Balanced performance across verdict classes
- **Calibration** - Confidence scores reflect actual accuracy
- **Hallucination Rate** - Target: < 5%

---

## Using the Published DPO Model

The trained model is published as a PEFT/LoRA adapter:

```text
andreiungureanu/Fact-Checking-Agent-LLionelMessi
```

It attaches to:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Install the model dependencies first:

```bash
pip install -r requirements.txt
```

Then load it directly from Hugging Face:

```python
from src.model_training.inference import load_fact_checker

model = load_fact_checker()

evidence = [
    {
        "id": "chem_1",
        "text": "A water molecule has the chemical formula H2O, meaning it contains two hydrogen atoms and one oxygen atom.",
    }
]

result = model.generate_verdict(
    "Water is composed of two hydrogen atoms and one oxygen atom.",
    evidence,
)

print(result.verdict)
print(result.raw_text)
print(result.hallucinated_citations)
```

For quick non-download tests of prompt parsing:

```bash
pytest tests/test_model_prompting.py -v
```

For the real model download/inference test:

```bash
RUN_MODEL_INFERENCE_TESTS=1 pytest tests/test_model_inference.py -v -s
```

## Evaluation Commands

Run the core tests:

```bash
pytest tests/test_evaluation_metrics.py tests/test_model_prompting.py tests/test_synthesis.py tests/test_orchestrator.py tests/test_claim_processing.py tests/test_scoring.py -v
```

Run the guarded Hugging Face adapter smoke tests:

```bash
RUN_MODEL_INFERENCE_TESTS=1 pytest tests/test_model_inference.py -v -s
```

Validate the local Chroma index:

```bash
python -m src.scripts.validate_corpus --sample-queries 20
```

Run the full pipeline evaluation after building the local Chroma index. Use decontamination to exclude exact training-claim overlaps:

```bash
python -m src.scripts.evaluate_pipeline \
  --max-claims 500 \
  --top-k 5 \
  --exclude-train-overlap \
  --output data/processed/pipeline_eval_500_decontam.json
```

Run the offline evaluator on saved predictions:

```bash
python -m src.scripts.evaluate_predictions predictions.jsonl --output data/processed/offline_eval_results.json
```

Each offline prediction row should look like:

```json
{"claim_id":"dev_1","gold_verdict":"SUPPORTED","predicted_verdict":"SUPPORTED","confidence":0.82,"cited_passage_ids":["p1"],"retrieved_passage_ids":["p1","p2"]}
```

The offline evaluator reports accuracy, macro F1, ECE, hallucinated citations, missing citations, and whether the hallucination target is met.

## Task A: Data & Ingestion — Status

> **Done.** Corpus indexed, retriever operational, training triples exported, evaluation baseline script written.
> **Important:** `data/index/` is gitignored. Clone users must run the full-wiki download + build steps below before the retriever will return meaningful results.

### Corpus at a glance

| Dataset | Claims | Passages in index |
|---------|--------|-------------------|
| FEVER (train) | 145,449 | 662,806 Wikipedia chunks (262,993 political pages) |
| LIAR/PolitiFact | 12,836 | 12,836 claim texts |
| **Total** | **158,285** | **675,642** |

The index uses a political keyword filter over the full FEVER Wikipedia dump (~5.4M pages). Only pages with politically relevant titles (politician roles, government institutions, elections, policy topics, etc.) are retained (~263k pages, ~4.9% of the dump), then chunked into ~500-character overlapping passages. This gives broad evidence coverage for political claims without the impractical cost of indexing all 25M Wikipedia sentences.

Training triples for fine-tuning are in `data/processed/`:

| File | Triples |
|------|---------|
| `train.jsonl` | 126,628 |
| `val.jsonl` | 15,828 |
| `test.jsonl` | 15,829 |

### Quick Start (first-time setup)

> **Note:** `data/index/` is gitignored — the ChromaDB index is not stored in the repository. Every team member must build it locally.

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Download FEVER + LIAR datasets and the full Wikipedia corpus
# (~30–60 min depending on bandwidth; HuggingFace caches on disk)
python -m src.scripts.download_data --fever-split train --load-wiki

# Build the ChromaDB index with political keyword filter (~20–40 min)
# Streams all Wikipedia pages, keeps politically relevant ones,
# chunks full page content into ~500-char passages.
python -m src.scripts.build_index --political-filter --clear

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
