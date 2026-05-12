# Presentation Design Spec — Fact-Checking Agent

**Date:** 2026-05-12
**Audience:** ~100 people, course presentation
**Total time:** 10 minutes (4 speakers × ~2 min + ~2 min demo video)
**Owners:** Mihai (A — Data & Ingestion), David (B — Claim Processing), Andrei (C — Model Training), Vasile (D — Scoring, Synthesis, Eval & Orchestration)

---

## 1. Goals

- Communicate the agent's architecture and the team's contributions at the level a non-specialist 100-person audience can follow, while staying credible to ML/NLP-literate viewers.
- Lead with the **engineering identity** of the work — this is a real multi-component system with measured results, not a hackathon mock.
- Land two headline numbers in the audience's memory: **0.704 accuracy / 0.701 macro F1** and **0.0% hallucinated citations**.
- Make the team's iteration on the verdict head (TinyLlama SFT+DPO → DeBERTa verifier) read as normal, data-driven engineering, not a substitution.
- Use a demo video as the closer — three political claims showing all three verdicts, voice-over live.

## 2. Non-goals

- Pitch-deck framing (no TAM/SAM, no "what you feel" sections).
- Live coding or live model inference during the talk.
- Full reproduction of evaluation methodology — the deck cites results, the spec/repo has the methods.
- Dark-themed slides (rejected on projector-lighting risk).

## 3. Style — "Engineering Notebook"

Light background, blue-toned, grid + figure-label chrome, restrained color, monospace numbers. Engineering-document feel, not marketing.

### Tokens

| Token | Value | Use |
|---|---|---|
| `surface` | `#f5f7fb` | slide background |
| `grid` | `#dbe4f3` @ 8% opacity, 24px square | background grid |
| `ink` | `#0b2447` | primary text, borders |
| `ink-2` | `#1e3a8a` | secondary text |
| `muted` | `#64748b` | captions, axis labels |
| `accent` | `#0ea5e9` | data highlights, used sparingly |
| `accent-strong` | `#0369a1` | accent on hover / hero numbers |
| `caveat` | `#d97706` | trade-offs, honest caveats |
| `ok` | `#0d9488` | success indicators (e.g. 0.0% hallucination badge) |

### Type

- Sans (headings/body): **Inter** (already web-safe; load via `@fontsource/inter`).
- Mono (numbers, metrics, code, axis labels, figure labels): **JetBrains Mono** (via `@fontsource/jetbrains-mono`).
- Italic small-caps mono for figure labels and section indicators.

### Slide chrome (every content slide)

- **Top-right:** figure label in italic mono small-caps — e.g. `FIG. 05 · RETRIEVAL RECALL`.
- **Top-left:** section indicator — e.g. `§ A — DATA & INGESTION`. **No speaker names on chrome.**
- **Bottom-right:** progress dots + slide number `05 / 14` in mono.
- All boxes use 1px `ink` hairline borders; dashed for conceptual flow, solid for data flow.
- Arrows in diagrams have small mono edge labels (e.g. `top_k=5`).

### Motion

- Default: 200ms `ease-out` fade + 12px vertical translation, no spring.
- **One signature animation per slide** for the data payoff (e.g. recall bars animate in sequence, big number counts up, journey line draws).
- Slide-to-slide transition: 250ms horizontal slide, no scale change.

## 4. Deck structure (14 slides)

Time column is target talk time over the slide; sum is ~10 min 5 s.

| # | Slide | Section | Time | Diagram |
|---|---|---|---|---|
| 1 | Title | — | 0:05 | Project name + 4 names + HF model handles |
| 2 | The Problem | Intro | 0:25 | Two-failure-mode schematic + abstention principle |
| 3 | Architecture | Intro | 0:30 | 5-tool horizontal pipeline w/ labeled I/O edges |
| 4 | Corpus & filter | A | 0:40 | Funnel `5.4M → 263K → 662K` with amber filter caveat |
| 5 | Retrieval breakthrough | A | 0:30 | Grouped bar chart Recall@{1,5,10} dense vs hybrid+title |
| 6 | Claim Decomposer | B | 0:50 | Tree: compound → atomics, Ollama+fallback path |
| 7 | Stance Classifier | B | 1:00 | NLI schematic `(claim, passage) → DeBERTa → {SUP/REF/NEU}` |
| 8 | SFT + DPO pipeline | C | 1:00 | Pipeline: triples → SFT → preference pairs → DPO → LoRA |
| 9 | Why two verdict heads | C | 0:50 | Side-by-side: TinyLlama+DPO vs DeBERTa verifier dev results |
| 10 | Synthesis & orchestrator | D | 0:40 | Orchestrator state diagram, citation tracing callout |
| 11 | Trained verifier | D | 0:35 | Verifier card + dev numbers (0.819 / 0.795) + calibration note |
| 12 | Headline result | D | 0:45 | Big mono `0.704` + `0.0%` badge + journey-line inset |
| 13 | Demo video | — | ~2:00 | Embedded MP4, voice-over live |
| 14 | Close + links | — | 0:15 | HF model URLs, repo URL, thanks |

**Speaker budget:** Mihai ≈ 2:10 (slides 1–5, includes title), David ≈ 1:50 (6–7), Andrei ≈ 1:50 (8–9), Vasile ≈ 2:00 (10–12). Vasile also voice-overs the demo video. Slot 14 (close) is handled by Mihai or whoever picks up after the demo. Buffer is built into the demo slot — the video itself can be trimmed to ~110 s leaving ~10 s of slack.

## 5. Per-slide content (what each slide must show)

### Slide 1 — Title
- Project name "Fact-Checking Agent — LLionelMessi" rendered with mono treatment on `LLionelMessi`.
- Names of the 4 team members, no roles on this slide (roles surface as the talk progresses).
- Two HF model handles in mono footer:
  - `vasiledraguta/deberta-v3-base-fever-verifier-20k-finetuned`
  - `andreiungureanu/Fact-Checking-Agent-LLionelMessi`
- Network-IP remote URL stays visible in faint mono at bottom right (already in template).

### Slide 2 — The Problem
- Headline: "Automated fact-checkers fail in two ways."
- Two-column schematic:
  - Left box: "Over-aggressive classifiers" → false positives.
  - Right box: "Confident LLMs" → hallucinated citations.
- Below: a banded statement — `Design principle: abstention > fabrication.` in mono.

### Slide 3 — Architecture
- The 5-tool pipeline, left to right:
  - Claim → **Decomposer** → atomic claims → **Retriever** → passages → **Stance Classifier** + **Credibility Scorer** → signals → **Verdict Synthesizer / Verifier** → verdict + citations.
- Each tool box labeled with the I/O contract in mono.
- Edge labels: `top_k=5`, `candidate_k=50`, `max_passages=3`.
- Foot caption: "Tool-augmented LLM pipeline. Orchestrator decides when to invoke each."

### Slide 4 — Corpus & filter (A)
- Funnel diagram with mono numbers:
  - `5.4M` FEVER wiki pages → (political-keyword filter) → `263K` pages → (chunking, ~500 char w/ overlap) → `662,806` passages.
- Side card: total claims `158,285` = FEVER train `145,449` + LIAR/PolitiFact `12,836`.
- **Amber caveat box** in `caveat` color: "Filter introduces selection bias. We keep only politically relevant pages to fit infrastructure constraints — claims outside that scope retrieve poorly."
- Foot: "Index: ChromaDB, MiniLM-L6-v2 embeddings."

### Slide 5 — Retrieval breakthrough (A)
- Grouped bar chart, `Recall@1`, `Recall@5`, `Recall@10`, on 500 decontaminated FEVER dev claims.
- Two series:
  - Dense-only: `0.015 / 0.023 / 0.023` (small navy bars at the bottom).
  - Hybrid + FEVER title index: `0.694 / 0.848 / 0.857` (tall cyan bars).
- Caption: "Hybrid retrieval (dense + title index) + cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) on `candidate_k=50`."
- One-line punch in mono: `Recall@10: 0.023 → 0.857`.

### Slide 6 — Claim Decomposer (B)
- Tree:
  - Root: `"Obama, elected in 2008, signed the ACA in 2010."`
  - Children: `c1: Obama was elected in 2008.` / `c2: Obama signed the ACA in 2010.`
- Implementation path:
  - Primary: Ollama-hosted instruction-tuned LLM (note: local; no API key required).
  - Fallback: deterministic rule-based splitter when Ollama unavailable.
- Caption: "Atomic sub-claims unlock per-claim retrieval + per-claim verdicts."

### Slide 7 — Stance Classifier (B)
- NLI schematic: `(claim, passage) → DeBERTa-v3-NLI → {SUPPORTS, REFUTES, NEUTRAL}`.
- Worked example:
  - Claim: "Obama was the 44th US President."
  - Passage: "Barack Obama (born 1961) served as the 44th president of the United States from 2009 to 2017."
  - Output: `SUPPORTS · 0.97`.
- Footnote: "We also classify same-source evidence windows when title context is available — keeps citations restricted to original passage IDs."

### Slide 8 — SFT + DPO pipeline (C)
- Pipeline:
  - `claim–evidence–verdict triples (126K)` → **SFT** on `TinyLlama-1.1B-Chat` with LoRA → SFT adapter
  - → preference pairs (well-cited > confident-unsupported) → **DPO** → DPO adapter
- Side card: "Prompt design encourages explicit citation IDs and `NOT_ENOUGH_INFO` when evidence is weak."
- HF: `andreiungureanu/Fact-Checking-Agent-LLionelMessi` (in mono).

### Slide 9 — Why two verdict heads (C)
- **Honest framing:** we trained TinyLlama with SFT+DPO; on the abstention smoke test (Everest claim with Mariana Trench evidence) it returned `SUPPORTED`. We added a deterministic citation/evidence guardrail to make it safe (`generate_verdict()`), and in evaluation explored a DeBERTa verifier as a second verdict head — which won on FEVER dev.
- Side-by-side comparison (FEVER labelled_dev):
  - TinyLlama + DPO + guardrail: results bounded by the guardrail; raw model not trusted as standalone verdict head.
  - DeBERTa verifier (20k fast): `eval_accuracy 0.819 · eval_macro_f1 0.795`.
- Lesson box in `ink-2`: "Generative heads optimise for fluent verdicts; discriminative heads optimise for label probabilities. For 3-way classification with calibrated confidence, discriminative wins on this dataset."

### Slide 10 — Synthesis & orchestrator (D)
- Orchestrator state diagram (compact, vertical):
  - `decompose → retrieve(per atomic) → score_credibility → classify_stance → synthesize → verify(verifier) → calibrate`.
- Callout on "citation tracing": all cited passage IDs are restricted to retrieved IDs — this is what produces the 0.0% hallucination figure.
- Foot caption: "Source-relevance defaulted to 0.5 (neutral) when metadata absent — prevents the weak-source guard from silently disengaging on non-FEVER data."

### Slide 11 — Trained verifier (D)
- Card: model = `microsoft/deberta-v3-base`, fine-tuned on `20K` FEVER train triples (+ `2K` dev), `2 epochs`, `lr=2e-5`, `fp16`, class-weighted.
- Dev metrics in mono: `eval_accuracy 0.819 · eval_macro_f1 0.795`.
- Calibration note: temperature scaling on dev split (sidecar `temperature.json`).
- HF: `vasiledraguta/deberta-v3-base-fever-verifier-20k-finetuned`.

### Slide 12 — Headline result (D)
- Hero number: `0.704` (accuracy, 8rem mono, ink), with `Macro F1 0.701` directly below in smaller mono.
- Right column: `0.0%` hallucinated citations as a green/ok badge — "Target was < 5%."
- Bottom inset: journey line — `0.380 → 0.448 → 0.510 → 0.690 → 0.704` over the iteration steps (baseline → +title retrieval → +reranker → +source-aware synthesis → +trained verifier).
- Caption: "500-claim FEVER labelled_dev, decontaminated (`--exclude-train-overlap`), held-out threshold tuning."

### Slide 13 — Demo video
- Embedded MP4 (`presentation/public/demo.mp4`), 16:9, autoplays on slide enter, no controls visible, ~2 min runtime.
- Voice-over: live by Vasile (or fallback Mihai). Script in §7.

### Slide 14 — Close + links
- One-line takeaway: "Tool-augmented RAG + trained discriminative verifier + citation tracing = 0.0% hallucination on political fact-checking."
- Links: GitHub repo, two HF model URLs (all mono).
- Thanks.

## 6. Demo video plan

**Format:** Screen recording of `python -m src.scripts.chat`, three political claims, ~35–40 s per claim, ~15 s opening shot, ~10 s closing shot. Final ~110 s total. Voice-over is live during the talk, not baked into the video — so the recording can be silent screen capture.

**Claims (locked):**
1. **SUPPORTED** — "Barack Obama was the 44th President of the United States."
2. **REFUTED** — "The first president of Romania was Lionel Messi."
   - **Risk:** depends on whether the political-keyword filter retained Romanian presidency pages (Ion Iliescu) AND/OR a strong Lionel Messi page that contradicts him being a politician. We must dry-run before recording.
   - **Fallback claim if #2 returns NEI:** "Lionel Messi served as Prime Minister of Spain." Same joke, Spain's PMs are well-covered.
3. **NEI** — "The next Romanian president after 2025 will lead a center-right coalition."

**What the video must show for each claim:**
- The user prompt in the REPL.
- Streaming retrieval indicator + the top retrieved passages with source titles.
- The structured `SynthesisResult` block: verdict, confidence, cited passage IDs.
- The Qwen responder's conversational reply.

**Production checklist (before recording):**
- [ ] Build the index locally (`build_index --political-filter --clear`).
- [ ] Run the 3 claims through `chat.py`; verify verdicts match expected. If #2 is wrong, switch to fallback.
- [ ] Record at 1920×1080, terminal at 16pt, dark-on-light terminal theme to match deck.
- [ ] No background noise, no notifications.
- [ ] Trim to ~110 s.
- [ ] Save as `presentation/public/demo.mp4`.

## 7. Speaker scripts

Each script is timed to the slot. Delivery is *natural* — these are talk-notes, not memorised text.

### Mihai (slides 1–5, ~2:10)

**Slide 1 (0:05):** "We built a fact-checking agent for political claims. I'm Mihai, with David, Andrei, and Vasile."

**Slide 2 (0:25):** "Automated fact-checkers fail in two ways. Aggressive classifiers flag too much — false positives bury the real signal. Confident LLMs do the opposite: they generate verdicts that sound right and cite sources that don't exist. We built this agent around one principle: abstain rather than fabricate."

**Slide 3 (0:30):** "Five tools, one pipeline. A claim comes in, we decompose it, retrieve evidence, score the source, classify per-passage stance, and synthesise a verdict with cited passages. The orchestrator decides when to invoke each tool — the audit trail is the whole point."

**Slide 4 (0:40):** "I own data and ingestion. We index FEVER and LIAR/PolitiFact — about 158 thousand claims. The catch: the full FEVER Wikipedia dump is 5.4 million pages, far too much to embed. So I filter on political-relevance keywords and keep about 263 thousand pages, chunked into 662 thousand passages. The honest trade-off is selection bias — claims outside political scope retrieve worse. We accepted that because political fact-checking was the Phase 1 target."

**Slide 5 (0:30):** "Dense embeddings alone gave Recall@10 of 0.023 — the right Wikipedia page was almost never in the top results. Hybrid retrieval — a FEVER title index plus dense embeddings, plus a cross-encoder reranker over fifty candidates — took Recall@10 to 0.857. That single change unlocked everything downstream."

### David (slides 6–7, ~1:50)

**Slide 6 (0:50):** "I own claim processing. The decomposer takes a compound claim — 'Obama, elected in 2008, signed the ACA in 2010' — and splits it into atomic sub-claims, each verifiable independently. Primary path is a local Ollama-hosted instruction-tuned LLM, with a deterministic rule-based fallback so the pipeline doesn't break when Ollama isn't running. Decomposition is what lets us assign per-sub-claim verdicts and per-sub-claim citations."

**Slide 7 (1:00):** "The stance classifier is a DeBERTa-v3 NLI model. For every retrieved passage, it labels the relationship between the claim and the passage as supports, refutes, or neutral. Example: claim 'Obama was the 44th US President', passage 'Barack Obama served as the 44th president from 2009 to 2017' — SUPPORTS with 0.97 confidence. Citations are always restricted to passage IDs we actually retrieved — no hallucinated sources."

### Andrei (slides 8–9, ~1:50)

**Slide 8 (1:00):** "I own model training. We started with TinyLlama-1.1B, fine-tuned with LoRA on 126 thousand claim–evidence–verdict triples — SFT first to teach the format, then DPO to prefer well-cited, conservative verdicts over confident but unsupported ones. The prompt explicitly asks the model to cite passage IDs and to output NOT_ENOUGH_INFO when evidence is thin. The adapter is on Hugging Face."

**Slide 9 (0:50):** "Then we evaluated. On an abstention smoke test — an Everest claim with Mariana Trench evidence — the raw model returned SUPPORTED. So we added a deterministic citation and evidence guardrail. In parallel, the team explored a DeBERTa verifier as a discriminative second verdict head. On FEVER dev it hit 0.819 accuracy — more reliable than what we could get from the generative head alone. The lesson: generative heads produce fluent verdicts; discriminative heads produce calibrated labels. For 3-way classification on this dataset, discriminative won."

### Vasile (slides 10–12, ~2:00)

**Slide 10 (0:40):** "I own scoring, synthesis, evaluation, and the agent loop. The orchestrator runs the tools in sequence, accumulating evidence, and the synthesiser aggregates stance and credibility into a verdict with confidence and explicit citations. The reason we hit 0.0% hallucination is here: cited passage IDs are restricted to retrieved IDs by construction. Source-relevance defaults to 0.5 when metadata is missing — that prevents the weak-source guard from silently disengaging on non-FEVER data."

**Slide 11 (0:35):** "The trained verifier is DeBERTa-v3-base, fine-tuned on 20 thousand FEVER train triples for two epochs with class weighting. Dev set: 0.819 accuracy, 0.795 macro F1. We fit a temperature scalar on the dev split to calibrate confidence — written out as a sidecar JSON. Model is on Hugging Face."

**Slide 12 (0:45):** "End-to-end on 500 decontaminated FEVER dev claims, with held-out threshold tuning: 0.704 accuracy, 0.701 macro F1. And the headline number — 0.0% hallucinated citations. Target was under 5%, we cleared it by construction. The journey: 0.380 baseline, then title retrieval, reranking, source-aware synthesis, and the trained verifier. Demo next."

### Vasile (demo voice-over, ~2:00)

(Live, over the video.) "Three political claims, three verdicts. First: 'Barack Obama was the 44th President.' Retrieval pulls the Obama Wikipedia page, the verifier classifies SUPPORTED with high confidence, and the citations point to the exact passages. Second: 'The first president of Romania was Lionel Messi.' The system retrieves Romanian presidency evidence — Ion Iliescu was the actual first president — classifies REFUTED, and again cites the source. Third: 'The next Romanian president after 2025 will lead a center-right coalition.' The corpus has no evidence about future elections, the verifier returns NOT_ENOUGH_INFO — this is the abstention behaviour we designed for. Thanks."

## 8. Technical implementation

### Codebase changes (`presentation/`)

- Replace `App.tsx` content while preserving the WebSocket remote control, slide-jump hotkeys, progress bar, and `localStorage` slide persistence.
- Replace `SLIDE_INFO` with the 14-slide list.
- Add Inter + JetBrains Mono via `@fontsource/inter` and `@fontsource/jetbrains-mono`.
- Add a `<GridBackground/>` component (SVG pattern, 24px square, low opacity).
- Add reusable primitives:
  - `<FigureLabel id="05" title="RETRIEVAL RECALL" />` (top-right chrome)
  - `<SectionLabel section="A" title="DATA & INGESTION" />` (top-left chrome, **no speaker name**)
  - `<SchematicBox/>` (1px hairline, optional corner brackets, optional `dashed`)
  - `<MonoNumber value={0.704} size="hero" />`
  - `<JourneyLine points={[0.38, 0.448, 0.51, 0.69, 0.704]} />` (animated SVG path draw)
  - `<GroupedBars/>` for slide 5
  - `<Funnel/>` for slide 4
- The 5-tool architecture diagram (slide 3) gets its own component.
- All animations move to 200ms `ease-out` fades, no springs (Framer Motion is already in the template).

### File layout

```
presentation/src/
├── App.tsx                  # main presentation, 14 slides
├── PresenterRemote.tsx      # unchanged, already in template
├── theme.ts                 # color tokens, type tokens
├── components/
│   ├── GridBackground.tsx
│   ├── FigureLabel.tsx
│   ├── SectionLabel.tsx
│   ├── SchematicBox.tsx
│   ├── MonoNumber.tsx
│   ├── JourneyLine.tsx
│   ├── GroupedBars.tsx
│   ├── Funnel.tsx
│   ├── PipelineDiagram.tsx  # slide 3
│   ├── OrchestratorDiagram.tsx # slide 10
│   ├── NliSchematic.tsx     # slide 7
│   ├── DecomposerTree.tsx   # slide 6
│   └── TrainingPipeline.tsx # slide 8
├── slides/                  # one file per slide for clarity
│   ├── 01-title.tsx
│   ├── 02-problem.tsx
│   ├── ...
│   └── 14-close.tsx
├── App.css                  # global resets
└── index.css                # font imports, grid background CSS
```

### Backup of current content

- Existing `App.tsx` (MaxOnScribe content) is already backed up as `App.tsx.backup`. No further preservation needed.

## 9. Open risks & follow-ups

- **Demo claim #2** must be dry-run on the actual built index before recording. If it returns NEI, switch to the Spain PM fallback (already noted in §6).
- **Index size on Mihai's laptop:** the political-filter index build is the heaviest step (~20–40 min). Schedule the dry-run with enough buffer.
- **Font loading:** `@fontsource/jetbrains-mono` adds ~150KB; load only the subsets we need (`latin`).
- **Projector contrast:** the `#f5f7fb` background plus `#0b2447` ink gives a solid ~14:1 contrast ratio. Cyan-on-light is closer to 4.5:1 — keep cyan reserved for big numbers, never small text.
- **2-minute voice-over fit:** Vasile should rehearse against the cut video to make sure timing lands.

## 10. Out of scope (explicitly)

- Per-slide speaker notes embedded in the React app (script lives in this spec).
- Subtitles or baked-in audio on the demo video.
- Live remote-control UI tweaks (existing `PresenterRemote.tsx` works as-is).
- A printable handout (not requested).
