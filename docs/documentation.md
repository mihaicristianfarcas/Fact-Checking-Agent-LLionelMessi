# Automated Fact-Checking Agent — Project Documentation

**Team LLionelMessi** · Mihai-Cristian Farcaș · David Croitor · Andrei Ungureanu · Vasile Draguța
**Course deliverable** · May 2026

---

## Table of contents

1. [Introduction](#1-introduction)
2. [Problem definition](#2-problem-definition)
3. [State of the art](#3-state-of-the-art)
4. [Proposed solution](#4-proposed-solution)
5. [Experiments](#5-experiments)
6. [Results](#6-results)
7. [Demo](#7-demo)
8. [Deliverables & how to run](#8-deliverables--how-to-run)

---

## 1. Introduction

This project is an **automated fact-checking agent for political claims** in English. Given a natural-language claim — for example *"Barack Obama was the 44th President of the United States"* — the system retrieves evidence from an indexed Wikipedia/FEVER corpus, evaluates each retrieved passage for its stance toward the claim, and emits a structured verdict in **{SUPPORTED, REFUTED, NOT_ENOUGH_INFO}** together with confidence scores and **explicit, traceable citations** to the passages that justified the verdict.

The agent is implemented as a **tool-augmented Retrieval-Augmented-Generation (RAG)** pipeline: an orchestrator coordinates five specialised tools (decomposer, retriever, stance classifier, credibility scorer, verifier), accumulates evidence across tool calls, and only commits to a verdict once enough evidence is gathered (or aborts to NOT_ENOUGH_INFO when it isn't). The verdict head is a **DeBERTa-v3 classifier** fine-tuned on FEVER and temperature-calibrated on a held-out dev split.

The design priority across every component is **abstention over fabrication**: when the evidence is insufficient or contradictory, the agent is engineered to *say so* rather than emit a confident-looking guess. The mechanism we use to make that priority concrete is **citation tracing** — cited passage IDs are restricted by construction to the IDs that the retriever actually returned, so the system cannot reference evidence it didn't see.

Headline numbers on the 500-claim decontaminated FEVER dev evaluation set:

| Metric | Value |
|---|---|
| Accuracy (3-way) | **0.704** |
| Macro F1 | 0.701 |
| Expected Calibration Error (ECE) | 0.185 |
| **Hallucinated citations** | **0.0 %** (target was < 5 %) |
| Retrieval Recall@10 | 0.857 |

---

## 2. Problem definition

### 2.1 Why automated fact-checking is hard

Automated fact-checkers tend to fail in one of two ways.

**Failure mode A — over-flagging (high false-positive rate).**
Aggressive classifiers that always commit to SUPPORTED or REFUTED drown the true-positive signal in noise. Users lose trust in flags that are too often wrong, and the system becomes operationally useless: a fact-checker that cries wolf is worse than no fact-checker, because human reviewers stop reading the output.

**Failure mode B — fabricated citations.**
Large language models prompted to fact-check produce verdicts that *sound* well-grounded. They cite URLs, paper titles, page numbers — and a non-trivial fraction of those references either do not exist or do not say what the model claims they say. When a downstream user trusts the citation without re-checking, fabricated evidence becomes harder to detect than no evidence at all.

### 2.2 Design principle

The agent is built around one explicit principle: **abstain rather than fabricate**. Concretely, this means:

- The verdict label space is three-way, not binary — NOT_ENOUGH_INFO is a first-class output.
- Citations are restricted by construction to passage IDs the retriever produced — the synthesis layer cannot reference text that wasn't retrieved.
- A weak-source guard refuses to commit when only low-credibility evidence is available.
- The training loss (DPO preference pairs) explicitly prefers well-cited conservative verdicts over confident unsupported ones.

### 2.3 Scope

**In scope (Phase 1):** English-language political claims — politicians, elections, legislation, institutions. The corpus, the retrieval index and the evaluation set are scoped to this domain.

**Out of scope (Phase 1):** non-political claims, non-English text, claims requiring multi-hop reasoning across more than two passages, and time-sensitive claims that demand a live web search rather than an indexed corpus.

### 2.4 What "good" looks like

We pre-committed to three operational targets before training:

| Target | Threshold | Outcome |
|---|---|---|
| End-to-end accuracy on FEVER dev | ≥ 0.65 | ✓ 0.704 |
| Hallucinated-citation rate | < 5 % | ✓ 0.0 % |
| Latency per claim (CPU, no GPU at inference) | < 5 s | ✓ ~2 s |

---

## 3. State of the art

The problem we're solving is well-studied; what's contested is *which guarantees* a system should provide. We surveyed three families of approaches before designing ours.

### 3.1 FEVER baselines (DrQA-style pipelines)

The original **FEVER** shared task established a three-stage pipeline: document retrieval (TF-IDF / DrQA), sentence selection, and natural-language inference. Strong systems on the FEVER leaderboard (KGAT, ESIM-based ensembles, transformer rerankers) reach 0.68–0.73 verdict accuracy. They share two limitations:

- **Citation grounding is implicit.** The classifier's input includes sentence IDs, but the system does not enforce that downstream outputs *only* reference retrieved IDs — citation faithfulness depends on the pipeline plumbing rather than a hard constraint.
- **No probability calibration.** Outputs are argmax labels with raw softmax scores; ECE on out-of-distribution data is high.

### 3.2 LLM-only prompted verifiers

Direct LLM prompting — give GPT-style models the claim, ask for a verdict and citations — produces fluent answers but well-documented citation fabrication: hallucinated URLs, fictitious authors, paragraphs invented out of distribution. Recent surveys (2023–2025) report hallucination rates between 10 % and 30 % depending on domain and prompt complexity, with no reliable way to know per-output whether the citation is real.

### 3.3 Tool-augmented LLM fact-checkers (Factool, Loki, ReAct-style)

The most recent line of work combines an LLM with retrieval tools, typically using a ReAct-style loop where the LLM decides when to search, what to search for, and when to commit. Notable systems:

- **Factool** (Chen et al., 2024) — multi-task fact-checking with retrieval, evidence aggregation, and self-consistency.
- **Loki / FOLK** — retrieval-augmented verification with chain-of-thought aggregation.
- **Self-Ask / ReAct** — general agentic frameworks that fact-checking can be implemented on top of.

They improve on direct LLM prompting (retrieval grounds the model's claims), but the **citation guarantee is still best-effort**: the LLM is *encouraged* to cite retrieved snippets, but nothing in the architecture *forces* it to. Free-form citation strings frequently drift away from the retrieved set.

### 3.4 Our position

| System | Approach | Citation guarantee | Calibrated probs |
|---|---|---|---|
| FEVER baselines | retrieve + classify | partial (retrieval IDs, not enforced downstream) | no |
| LLM-only | generative verdict | hallucinated | no |
| Tool-augmented LLM (Factool, Loki) | LLM + retrieval + tools | best-effort | no |
| **Ours** | orchestrator + trained DeBERTa verifier | **restricted by construction** | **temperature-scaled on dev** |

Our two differentiators are exactly the two failure modes from §2.1 turned into architectural properties: (1) citations are a *type-level* constraint, enforced in the synthesiser; (2) verdict probabilities are calibrated post-hoc so downstream consumers can threshold confidence meaningfully.

---

## 4. Proposed solution

### 4.1 System overview

The agent is a five-tool pipeline coordinated by an orchestrator that mediates tool calls, accumulates evidence, and commits to a verdict.

```
       ┌────────────┐  top_k=5   ┌────────────┐  cand_k=50  ┌────────────┐
claim ─│ decomposer │──atomic───▶│ retriever  │──passages──▶│ stance cls │
       └────────────┘            └────────────┘             └────────────┘
                                                                  │
                                                                  ▼
                              ┌─────────────────┐         ┌──────────────┐
                              │ cred. scorer    │────────▶│   verifier   │──▶ verdict + citations
                              │  (source trust) │         │ DeBERTa-v3   │
                              └─────────────────┘         └──────────────┘
```

The orchestrator is a deterministic state machine — not an autonomous agent — that calls each tool, collects its output, and runs a guard at each step to decide whether to continue, abort to NOT_ENOUGH_INFO, or pass to the next tool.

### 4.2 Component details

#### 4.2.1 Corpus & ingestion

- **Source:** FEVER Wikipedia dump (5.4M pages) + LIAR / PolitiFact claims dataset.
- **Filter:** keyword-based political-relevance filter retains ~4.9 % of pages (≈ 263 K), based on a curated keyword list covering politicians, elections, parties, institutions, and policies.
- **Chunking:** ~500-character passages with sentence-boundary respect.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-d) stored in **ChromaDB**.
- **Final index size:** 662,806 passages.
- **Claims dataset (training/eval splits):** 158,285 total claims (145,449 FEVER + 12,836 LIAR/PolitiFact); SFT triples split 126K / 16K / 16K (train / val / test).

#### 4.2.2 Hybrid retrieval

Pure dense retrieval performed poorly out of the box (Recall@10 = 0.023 — see §5.2). The production retriever is **hybrid**:

1. **Title-lookup index.** A SQLite table maps normalised page titles → page IDs. For named-entity claims, this gives a strong recall signal essentially for free.
2. **Dense MiniLM** retrieval over the ChromaDB index.
3. **Candidate union** → keep top 50 candidates.
4. **Cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescores candidates jointly with the claim.
5. **Final top-k = 5** passages handed to the stance classifier.

#### 4.2.3 Claim decomposer

Compound claims are decomposed into atomic sub-claims that can be verified independently. For example:

> *"Obama, elected in 2008, signed the Affordable Care Act in 2010."*

decomposes into:

- c₁ = *"Obama was elected in 2008."*
- c₂ = *"Obama signed the ACA in 2010."*

**Primary path:** an Ollama-hosted instruction-tuned LLM (`llama3.2:3b` by default).
**Fallback:** a deterministic rule-based decomposer (sentence-split + coordinating-conjunction split) used when Ollama is unavailable. The fallback is intentionally simpler than the LLM path — preserves availability at the cost of decomposition quality.

#### 4.2.4 Stance classifier (NLI)

For each `(atomic_claim, retrieved_passage)` pair, a DeBERTa-v3 NLI model emits a softmax over **{SUPPORTS, REFUTES, NEUTRAL}**. Every classification carries the passage ID it was computed for; this is the structure that makes citation tracing possible downstream.

#### 4.2.5 Credibility scorer

A lightweight, deterministic scorer over passage metadata:

```
source_relevance = f(title_match, rerank_score, page_title_penalty)
```

Default value of 0.5 when metadata is missing — this prevents the weak-source guard from silently disengaging on data outside FEVER. Output is in [0, 1] and is used both as an input to the verifier and as a hard gate (verdicts requiring credibility above a tuned threshold).

#### 4.2.6 Verifier (the verdict head)

The verifier is **`microsoft/deberta-v3-base`** fine-tuned as a 3-way sequence classifier on 20K FEVER train claim/evidence pairs (plus 2K dev for calibration).

- **Architecture:** standard BERT-style encoder with a 3-class classification head.
- **Training:** 2 epochs, batch size 16, learning rate 2e-5, fp16, class weighting on (to compensate for FEVER's class imbalance), max sequence length 384.
- **Calibration:** temperature scaling — a single scalar T is fit on the dev split to minimise NLL, written out as `temperature.json` next to the checkpoint, loaded automatically at inference.
- **Published as:** `vasiledraguta/deberta-v3-base-fever-verifier-20k-finetuned` on Hugging Face.

The verifier replaced an earlier TinyLlama-1.1B + LoRA + DPO generative verdict head (kept for the conversational explanation layer). See §5.5 for the comparison.

#### 4.2.7 Orchestrator & synthesis

The orchestrator runs the pipeline as a finite-state machine. The synthesis step aggregates per-passage stance, per-passage credibility, and the verifier's 3-way probabilities into a single output:

```jsonc
{
  "claim": "Barack Obama was the 44th President of the United States.",
  "verdict": "SUPPORTED",
  "confidence": 0.94,
  "citations": [
    { "passage_id": "Barack_Obama_27", "stance": "SUPPORTS", "score": 0.97 },
    { "passage_id": "Barack_Obama_3",  "stance": "SUPPORTS", "score": 0.91 }
  ],
  "explanation": "Two independent passages on the Barack Obama page state ..."
}
```

The critical invariant — enforced by an assertion in the synthesiser, not just by prompting:

```python
assert set(c.passage_id for c in citations) ⊆ set(p.id for p in retrieved_passages)
```

This is the mechanism behind the 0.0 % hallucinated-citation rate.

### 4.3 Conversational explanation layer (TinyLlama)

A fine-tuned TinyLlama-1.1B (LoRA, r=8, α=16; SFT on 126K claim-evidence-verdict triples; DPO on preference pairs that reward well-cited conservative verdicts over confident unsupported ones) produces a natural-language explanation of the verdict for the REPL/demo experience. It runs *after* the verifier has produced the verdict — the LLM does not get to override the structured output.

- Adapter: `andreiungureanu/Fact-Checking-Agent-LLionelMessi` on Hugging Face.

---

## 5. Experiments

### 5.1 Data splits and decontamination

- **Train (verifier):** 20,000 FEVER claim/evidence pairs.
- **Calibration:** 2,000 FEVER dev claims (held out from training; used to fit temperature scalar and threshold).
- **End-to-end eval:** 500 FEVER dev claims, **decontaminated** — we removed any claim whose evidence overlapped with the verifier's training set.
- **Held-out threshold tuning:** a separate held-out split is used to pick the credibility threshold and the NEI abstention cutoff — these are not tuned on the 500-claim eval set.

### 5.2 Retrieval experiment

Compares dense-only retrieval against the production hybrid + reranker pipeline on 500 FEVER dev claims (Recall@k means "at least one gold-evidence sentence in top-k").

| Configuration | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|
| dense MiniLM only | 0.015 | 0.023 | 0.023 |
| **hybrid + title + reranker** | **0.694** | **0.848** | **0.857** |

Two observations: (a) for named-entity political claims, the title index dominates — most FEVER claims reference a specific Wikipedia page whose title is recoverable from the claim text; (b) the reranker provides the bulk of the precision gain on top of the title+dense union.

### 5.3 Verifier — dev-set evaluation

DeBERTa verifier on 2K FEVER dev claims (after class-weighted training, before temperature scaling):

| Metric | Value |
|---|---|
| Accuracy | 0.819 |
| Macro F1 | 0.795 |
| Per-class F1 — SUP / REF / NEI | 0.86 / 0.83 / 0.70 |

NEI is the weakest class — consistent with the FEVER literature; "not enough info" is intrinsically harder than the support/refute decision.

### 5.4 Calibration

We fit a single temperature scalar T on the 2K dev split, minimising NLL. The scalar moderately improves ECE without changing the argmax verdict (so accuracy is unchanged); this is the standard Guo et al. (2017) recipe.

### 5.5 Verdict-head comparison — TinyLlama-DPO vs DeBERTa

Mid-project, we had two verdict candidates: a TinyLlama-1.1B + LoRA + DPO generative head and a DeBERTa discriminative head. We evaluated both on FEVER dev.

| Head | Eval accuracy | Reliability |
|---|---|---|
| TinyLlama-1.1B + LoRA + DPO | ~0.62 (with deterministic guardrail wrapper) | Smoke test: Everest claim + Mariana Trench evidence → returned SUPPORTED without guardrail |
| **DeBERTa-v3-base verifier** | **0.819 on dev** | Stable 3-way distribution, calibratable |

**Decision and rationale.** Generative heads optimise for fluent text; discriminative heads optimise for calibrated 3-way labels. For this task, on this dataset, the discriminative head won — so we promoted DeBERTa to the verdict role. TinyLlama still ships in the system, but for *explaining* the verdict to the user, not for deciding it. Both adapters are public on Hugging Face.

### 5.6 Iteration journey

Pipeline accuracy on the 500-claim eval set across the major iterations:

| Iteration | Accuracy |
|---|---|
| Baseline RAG (dense-only retrieval, raw LLM verdict) | 0.380 |
| + title-lookup retrieval (hybrid) | 0.448 |
| + cross-encoder reranker | 0.510 |
| + source-aware synthesis + NLI head | 0.690 |
| **+ trained DeBERTa verifier (calibrated)** | **0.704** |

The two biggest single jumps came from hybrid retrieval (+0.07) and source-aware synthesis (+0.18). The trained verifier on top added a final ~0.014.

### 5.7 Hallucinated-citation audit

We define a citation as **hallucinated** if its passage ID is not in the retrieved set for that claim. We instrumented the synthesiser to log every cited ID and the corresponding retrieved set; on the 500-claim eval, the hallucination rate is **0 / 500 = 0.0 %**. The result is by construction: the synthesiser asserts the inclusion invariant before returning. We continue to measure the rate on every eval run as a regression check.

### 5.8 Latency

Mean per-claim latency on a 2024 MacBook Pro (M3, CPU-only inference) is ~2 seconds end-to-end. The dominant cost is the cross-encoder reranker over 50 candidates; the verifier itself is < 200 ms.

---

## 6. Results

### 6.1 Headline metrics

On the 500-claim decontaminated FEVER dev evaluation set:

| Metric | Target | Achieved |
|---|---|---|
| Accuracy (3-way) | ≥ 0.65 | **0.704** |
| Macro F1 | — | 0.701 |
| ECE (after T-scaling) | — | 0.185 |
| Hallucinated citations | < 5 % | **0.0 %** |
| Latency per claim (CPU) | < 5 s | ~2 s |

### 6.2 Per-class behaviour

The system is strongest on SUPPORTED, second on REFUTED, weakest on NEI — consistent with the dev-set verifier numbers. NEI is the abstention class and the one we care most about *being right* on (rather than maximising recall on), because the cost of a false-NEI is low (we just say "we don't know") while the cost of a false-SUPPORTED or false-REFUTED is high.

### 6.3 Failure-mode analysis

Sampling 50 mis-classified claims by hand revealed three recurring patterns:

- **Multi-hop reasoning required (~40 % of errors).** The claim's truth value depends on combining two facts from two different passages. The verifier sees passages independently and lacks the cross-passage reasoning capability.
- **Temporal mismatch (~25 %).** Claim and passage describe overlapping but temporally-disjoint events. The verifier sometimes labels SUPPORTS on a passage that is *related* but does not actually entail the claim.
- **Out-of-corpus claims (~20 %).** Despite the political filter, occasional claims about non-political entities slipped into the eval; retrieval predictably degrades for these, and the verifier inherits the bad evidence.

The remaining ~15 % are residual NLI errors with no obvious systematic pattern.

### 6.4 What the result means

The system meets its pre-committed targets. The architectural commitment — citations restricted to retrieved IDs by construction — delivers the zero-percent hallucination headline as a guarantee, not as a measurement we hope continues to hold. The 0.704 accuracy is competitive with FEVER baselines while providing properties (calibration, citation faithfulness) those baselines don't.

---

## 7. Demo

The demo is a **pre-recorded screen capture** of the REPL (`python -m src.scripts.chat`) running three political claims against the production pipeline. It runs ~2 minutes and plays automatically on slide 13 of the presentation deck.

### Claim 1 — SUPPORTED
> *"Barack Obama was the 44th President of the United States."*

Retrieval pulls the Obama Wikipedia page. The verifier classifies SUPPORTED with high confidence. Citations point to specific passages on the Obama page that say so.

### Claim 2 — REFUTED
> *"The first president of Romania was Lionel Messi."*

The system retrieves both the Romanian presidency evidence (Ion Iliescu was the first) and the Lionel Messi page (footballer). The verifier returns REFUTED. Citations include passages from both sides.

### Claim 3 — NOT_ENOUGH_INFO
> *"The next Romanian president after 2025 will lead a center-right coalition."*

The corpus contains no evidence about future elections. The verifier returns NOT_ENOUGH_INFO — this is the abstention behaviour the system is engineered for.

### Why pre-recorded, not live

Live inference takes ~2 seconds per claim; with three claims and the explanation step, a live demo is ~30 seconds of inference and ~30 seconds of UI overhead — too brittle and too time-consuming inside a 10-minute slot. Pre-recording lets us pace narration accurately and removes the risk of network/Ollama failures on stage.

---

## 8. Deliverables & how to run

### 8.1 Deliverables (zip contents)

- `src/` — agent source code (Python). Entry points: `src/scripts/chat.py` (REPL), `src/scripts/eval.py` (FEVER eval).
- `presentation/` — React + Vite presentation deck (source). Run `bun install && bun run dev`, open `http://localhost:5173`. The deck reads `presentation/public/demo.mp4` for slide 13.
- `presentation/dist/` — pre-built static deck (drop into any static host, or open `index.html`).
- `docs/documentation.md` — this document.
- `docs/presentation-script.md` — speaker notes for the 10-minute talk.
- `notebooks/` — training notebooks (TinyLlama LoRA + DPO; DeBERTa verifier fine-tuning).
- `requirements.txt` / `pyproject.toml` — Python dependencies.

### 8.2 Running the agent locally

```bash
# Set up environment
pip install -r requirements.txt

# (Optional) start Ollama for the decomposer; otherwise the rule-based fallback is used
ollama serve &
ollama pull llama3.2:3b

# Run the REPL
python -m src.scripts.chat

# Run end-to-end eval on FEVER dev
python -m src.scripts.eval --split dev --n 500
```

### 8.3 Models

Both fine-tuned models are published on Hugging Face:

- **Verifier (production verdict head):** `vasiledraguta/deberta-v3-base-fever-verifier-20k-finetuned`
- **TinyLlama LoRA adapter (explanation layer):** `andreiungureanu/Fact-Checking-Agent-LLionelMessi`

### 8.4 Repository

`github.com/mihaicristianfarcas/Fact-Checking-Agent-LLionelMessi` (private during course; public on request).

---

## Appendix A — Per-member contribution map

| Section | Owner(s) | Artefacts |
|---|---|---|
| Data & ingestion · corpus filter · indexing · retrieval | Mihai | corpus build scripts, ChromaDB index, title-lookup SQLite, retrieval evaluation |
| Claim processing · decomposer · stance NLI · Ollama integration | David | decomposer (LLM + rule-based fallback), DeBERTa-NLI stance classifier |
| Generative verdict head · SFT + DPO training · TinyLlama LoRA adapter · abstention guardrail | Andrei | training notebooks, Hugging Face adapter, evaluation harness |
| Orchestrator · synthesis · credibility scorer · DeBERTa verifier (training, calibration, integration) · end-to-end eval | Vasile | orchestrator state machine, synthesiser, DeBERTa verifier on HF, calibration pipeline, decontamination & evaluation tooling |
| Presentation deck · documentation | Mihai | this document |

---

## Appendix B — Glossary

- **FEVER** — Fact Extraction and VERification dataset/benchmark, MIT/Cambridge.
- **LIAR / PolitiFact** — political-claims labelled dataset, UCSB.
- **NLI** — Natural Language Inference (entailment / contradiction / neutral).
- **DeBERTa-v3** — Decoding-enhanced BERT with disentangled attention, v3 — Microsoft.
- **LoRA** — Low-Rank Adaptation, parameter-efficient fine-tuning.
- **DPO** — Direct Preference Optimization, an alternative to RLHF that trains on chosen/rejected preference pairs.
- **SFT** — Supervised Fine-Tuning.
- **Recall@k** — fraction of claims for which at least one gold-evidence sentence is in the top-k retrieved passages.
- **ECE** — Expected Calibration Error, average gap between predicted confidence and actual accuracy across confidence bins.
- **NEI** — NOT_ENOUGH_INFO, the third FEVER label (alongside SUPPORTS, REFUTES).
- **Decontamination** — removing eval claims whose evidence overlaps with the training set.
- **Hallucinated citation** — a cited passage ID that the retriever did not return for that claim.
