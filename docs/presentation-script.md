# Presentation Script

**Total runtime:** ~10 minutes
**Format:** 14 slides + ~2-minute pre-recorded demo video
**Speakers:** Mihai (01–06) → David (07) → Andrei (08–09) → Vasile (10–14, incl. demo VO)
**Handoffs:** 06 → 07 (Mihai → David), 07 → 08 (David → Andrei), 09 → 10 (Andrei → Vasile).

The script below is talk-notes, not lines to memorise. Adapt phrasing live; the timing and the *content beats* are what matter.

---

## At-a-glance running order

| # | Slide | Speaker | Time | Cue |
|---|---|---|---|---|
| 01 | Title | Mihai | 0:10 | Open the deck |
| 02 | The Problem | Mihai | 0:30 | Two failure-mode panels |
| 03 | State of the Art | Mihai | 0:40 | Comparison matrix |
| 04 | Architecture | Mihai | 0:35 | Five-tool pipeline |
| 05 | Corpus & filter | Mihai | 0:45 | Funnel + stat cards |
| 06 | Retrieval recall | Mihai | 0:30 | Recall@k bars |
| 07 | Claim processing | David | 1:00 | Decompose → NLI |
| 08 | SFT + DPO | Andrei | 0:50 | Training pipeline |
| 09 | Verdict-head eval | Andrei | 0:55 | Side-by-side card |
| 10 | Orchestrator | Vasile | 0:35 | Tool calls + tracing |
| 11 | Trained verifier | Vasile | 0:25 | Config + dev metrics |
| 12 | Headline result | Vasile | 0:40 | 0.704 / 0.0% hero |
| 13 | Demo video | Vasile (VO) | ~2:00 | Video plays automatically |
| 14 | Close + links | Vasile | 0:15 | Thanks |

**Per-speaker totals:** Mihai ≈ 3:30 · David ≈ 1:00 · Andrei ≈ 1:45 · Vasile ≈ 3:45 (incl. 2:00 demo VO)
**Net runtime:** ~10:00.

**Handoff convention:** don't say each other's names on stage. The last sentence of each section gestures naturally toward the next person — see "→ HANDOFF" notes. The next speaker just steps up and starts.

---

## Mihai — slides 01–06 (≈ 3:30)

### Slide 01 · Title (0:10)
> *Title shows: project name + four team names.*

"We built an automated fact-checking agent for political claims. I'm Mihai, presenting with David, Andrei, and Vasile."

→ Advance.

### Slide 02 · The Problem (0:30)
> *Two failure-mode panels appear, then the abstention principle.*

"Automated fact-checkers fail in two ways. Aggressive classifiers over-flag — false positives drown the real signal. Confident LLMs do the opposite: they generate verdicts that sound right and cite sources that don't exist. We built this agent around one principle — **abstain rather than fabricate**. If the evidence isn't there, the system says so."

→ Advance.

### Slide 03 · State of the Art (0:40)
> *Comparison matrix: FEVER baselines, LLM-only, LLM+retrieval, Ours.*

"Prior work splits three ways. **FEVER baselines** — DrQA, ESIM — retrieve and classify, but they don't calibrate and citation grounding is implicit. **LLM-only systems** produce fluent verdicts with hallucinated citations. **Recent tool-using fact-checkers** like Factool and Loki retrieve evidence and emit free-form citations — better, but the citation guarantee is best-effort, not enforced. **Our contribution** is the bottom row: a tool-augmented orchestrator plus a trained discriminative verifier, with citations restricted to retrieved passage IDs by construction and verdict probabilities temperature-scaled on a held-out dev split."

→ Advance.

### Slide 04 · Architecture (0:35)
> *Five tool boxes appear left-to-right; owner pills colour-code the work.*

"Five tools, one orchestrator. A claim enters, the **decomposer** splits it into atomic sub-claims, the **retriever** pulls passages, the **stance classifier** labels each (claim, passage) pair, the **credibility scorer** weights sources, and the **verifier** emits the final 3-way verdict. The orchestrator decides which tool to call at each step. The colour pills mark the four workstreams — A, B, C, D — that map to the four of us."

→ Advance.

### Slide 05 · Corpus & filter (0:45)
> *Funnel: 5.4M → 263K → 662K. Amber caveat box on the right.*

"I own data and ingestion. We index FEVER and LIAR/PolitiFact — about 158 thousand claims total. The catch: the full FEVER Wikipedia dump is 5.4 million pages, far too much to embed and serve. So we filter on political-relevance keywords — politicians, institutions, elections, policies — and keep about 263 thousand pages, chunked into 662 thousand passages in ChromaDB. The honest trade-off is selection bias: claims outside the political scope retrieve worse. We accepted that because Phase 1 targets political claims."

→ Advance.

### Slide 06 · Retrieval recall (0:30)
> *Bar chart animates in: tiny grey dense-only bars, tall cyan hybrid bars.*

"Dense embeddings alone weren't enough — Recall@10 was 0.023, meaning the right Wikipedia page almost never made the top results. Hybrid retrieval — a FEVER title index combined with dense embeddings, plus a cross-encoder reranker over the top 50 candidates — took Recall@10 to 0.857. That one change unlocked everything downstream."

→ HANDOFF: "Once the right evidence is in front of us, the next question is how we make sense of it." *Step aside; David takes the clicker.*

---

## David — slide 07 (≈ 1:00)

### Slide 07 · Claim processing (1:00)
> *Two-step diagram: top — decomposer tree; bottom — NLI flow with softmax bars.*

"I own claim processing. Two steps. **Decompose:** the compound claim 'Obama, elected in 2008, signed the ACA in 2010' splits into two atomic sub-claims that can be verified independently — primary path is an Ollama-hosted instruction-tuned LLM with a deterministic rule-based fallback so the pipeline doesn't break when Ollama isn't running. **Stance:** for every (atomic claim, retrieved passage) pair, a DeBERTa-v3 NLI model emits SUPPORTS, REFUTES, or NEUTRAL — here, 0.97 on SUPPORTS for the Obama example. The detail that matters: every classification carries the passage ID it was computed for. Citations cannot reference anything outside the retrieved set."

→ HANDOFF: "Per-claim labels and per-passage stance are the training signal — that's where Andrei picks up." *Hand the clicker over.*

---

## Andrei — slides 08–09 (≈ 1:45)

### Slide 08 · SFT + DPO (0:50)
> *Five-stage pipeline: triples → SFT → preference pairs → DPO → LoRA adapter.*

"I own model training. We started with TinyLlama-1.1B, fine-tuned with LoRA on 126 thousand claim–evidence–verdict triples. **SFT** first, to teach the verdict format and the citation grammar. Then **DPO** — preference learning, where the model is rewarded for well-cited conservative verdicts and penalised for confident-but-unsupported ones. The prompt explicitly asks the model to cite passage IDs and to output NOT_ENOUGH_INFO when evidence is thin. The adapter is published on Hugging Face."

→ Advance.

### Slide 09 · Verdict-head eval (0:55)
> *Two cards side-by-side: TinyLlama+DPO (with guardrail note) vs DeBERTa verifier (selected).*

"Then we evaluated. On an abstention smoke test — an Everest claim given Mariana Trench evidence — the raw TinyLlama returned SUPPORTED, so we wrapped inference with a deterministic citation-and-evidence guardrail. In parallel, we explored a DeBERTa verifier as a discriminative second verdict head. On FEVER dev it hit **0.819 accuracy** — more reliable than the generative head alone. The lesson on the slide: generative heads produce fluent verdicts; discriminative heads produce calibrated 3-way labels. For this task, on this dataset, discriminative won — so DeBERTa got promoted to the verdict role. TinyLlama still ships, but for the conversational explanation layer, not for deciding."

→ HANDOFF: "And the verifier is just one piece of how Vasile stitches everything together." *Step aside.*

---

## Vasile — slides 10–14 (≈ 3:45 incl. demo VO)

### Slide 10 · Orchestrator (0:35)
> *Orchestrator state diagram on the left; citation tracing + credibility scorer cards on the right.*

"I own scoring, synthesis, and end-to-end evaluation. The orchestrator runs the tools in sequence, accumulating evidence, and the synthesiser aggregates stance and credibility into a verdict with explicit citations. The reason the hallucination rate is zero is right here: **cited passage IDs ⊆ retrieved passage IDs, enforced in the synthesiser.** Source relevance defaults to 0.5 when metadata is missing — that prevents the weak-source guard from silently disengaging on non-FEVER data."

→ Advance.

### Slide 11 · Trained verifier (0:25)
> *Training-config table on the left; dev metrics + HF handle on the right.*

"The verifier specifics: DeBERTa-v3-base, fine-tuned on 20 thousand FEVER train triples for two epochs with class weighting and fp16. Dev set: 0.819 accuracy, 0.795 macro F1. We then fit a single temperature scalar on dev to calibrate confidence — written out as a sidecar JSON the agent loads automatically."

→ Advance.

### Slide 12 · Headline result (0:40)
> *Hero number 0.704 on the left; 0.0% hallucination badge on the right; journey line animates.*

"End-to-end on **500 decontaminated FEVER dev claims**, with held-out threshold tuning: **0.704 accuracy, 0.701 macro F1**. And the headline — **zero percent hallucinated citations**. Our target was under five percent; we cleared it by construction. The journey at the bottom traces the iteration: 0.380 baseline → title retrieval → cross-encoder reranking → source-aware synthesis → trained verifier."

→ "Demo next."

---

### Slide 13 · Demo video — voice-over (~2:00)

Vasile narrates live over the silent screen recording. The video shows three claims, in order, going through the REPL.

> *Video starts on slide enter (autoplay, muted).*

**Intro (~10 s):**
"This is the chat REPL running against the indexed corpus. Three political claims, three verdicts. Each one shows retrieval, the structured result, and the conversational explanation."

**Claim 1 — SUPPORTED (~35 s):**
"First: *'Barack Obama was the 44th President of the United States.'* Retrieval pulls the Obama Wikipedia page. The verifier classifies SUPPORTED with high confidence. The citations point to the exact passages that say so — nothing fabricated."

**Claim 2 — REFUTED (~35 s):**
"Second: *'The first president of Romania was Lionel Messi.'* The system retrieves the Romanian presidency evidence — Ion Iliescu was the actual first president — and the Messi page, which makes clear he's a footballer. The verifier classifies REFUTED. Citations on both sides."

**Claim 3 — NOT_ENOUGH_INFO (~35 s):**
"Third: *'The next Romanian president after 2025 will lead a center-right coalition.'* The corpus has no evidence about future elections. The verifier returns NOT_ENOUGH_INFO. This is the abstention behaviour we designed for — and the reason the hallucination rate is zero."

**Outro (~5 s):**
"Three claims, three different verdicts, zero hallucinated citations. Thank you."

→ Advance to slide 14.

---

### Slide 14 — Close + links (0:15)

Vasile (or whoever's at the front) lets the slide breathe for a beat.

"Code and both Hugging Face models are linked here. Happy to take questions."

---

## Practical notes for the day

**Clicker rotation.** Three handoffs — 06→07 (Mihai→David), 07→08 (David→Andrei), 09→10 (Andrei→Vasile). David's slot is short (one slide), so plan the choreography: Mihai finishes, David steps in for one slide, Andrei is already in position to take over. If you're using the phone remote at `<IP>:5173/present`, agree in advance who taps "next" on each handoff.

**Slide jumps.** `Alt + 1..9` jumps to slides 1–9 directly. Useful in Q&A if someone asks about a specific component.

**If the demo video fails to load:**
- Check that `presentation/public/demo.mp4` exists.
- Refresh; use `Alt+1` to return to title, then `→` to slide 13.
- Worst case, do the demo live: open a terminal next to the deck and run `python -m src.scripts.chat`.

**If demo claim #2 returns NEI instead of REFUTED:**
- Swap the recording to **"Lionel Messi served as Prime Minister of Spain."** Same comic energy, Spain's PMs are well-covered in FEVER, REFUTED is reliable.
- Update the matching card on slide 13 to match the new text before recording.

**Timing safety net.** If Vasile is running long, slide 11 (verifier config) can be skipped — the headline numbers on slide 12 carry the same point. Use `Alt + 1..9` to jump.

**Audience cues.** You're talking to ~100 people, mixed ML background. The numbers that matter are **0.704** and **0.0%**. The technical details (NLI, DPO, hybrid retrieval, calibration) signal rigor — you don't need to explain them deeply unless asked.

**Q&A pre-empts.**
- *"How did you measure hallucination?"* — A cited passage ID is hallucinated if it isn't in the retrieved set. We enforce that inclusion by construction, so the rate is 0% by design — but we still measure on every eval to catch regressions.
- *"Why DeBERTa instead of TinyLlama for verdicts?"* — Generative heads optimise for fluent text; discriminative heads optimise for calibrated 3-way labels. For this dataset, the discriminative head was more reliable. TinyLlama is still in the codebase for the conversational explanation layer.
- *"Does it work outside political claims?"* — Retrieval degrades outside the political-keyword corpus. Phase 1 scope was political; expanding the index is the next step.
- *"Why temperature scaling?"* — A single scalar fit on dev maps raw softmax probabilities to better-calibrated confidence. Improves ECE without changing the argmax verdict.
- *"How does this compare to Factool / Loki?"* — Same general shape (tool-using LLM + retrieval), but they emit free-form citations; we restrict citations to retrieved IDs and use a trained calibrated verifier head rather than asking the LLM to self-classify.

**One-line rehearsal goal.** Each speaker should be able to do their section *standalone in ~2 minutes without the slides*. The slides reinforce; they don't carry the talk.
