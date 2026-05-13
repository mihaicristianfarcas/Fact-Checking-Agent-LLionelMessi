# Presentation Script

**Total runtime:** ~10 minutes
**Format:** 14 slides + ~2-minute pre-recorded demo video
**Speakers:** Mihai (slides 01–07) → Vasile (slides 08–14 + demo)
**Handoff:** one clean transition at slide 07 → 08.

The script below is talk-notes, not lines to memorise. Adapt phrasing live; the timing and the *content beats* are what matter.

---

## At-a-glance running order

| # | Slide | Speaker | Time | Cue |
|---|---|---|---|---|
| 01 | Title | Mihai | 0:10 | Open the deck |
| 02 | The Problem | Mihai | 0:35 | Two failure-mode panels |
| 03 | State of the Art | Mihai | 0:45 | Comparison matrix |
| 04 | Architecture | Mihai | 0:40 | Five-tool pipeline |
| 05 | Corpus & filter | Mihai | 0:50 | Funnel + stat cards |
| 06 | Retrieval recall | Mihai | 0:40 | Recall@k bars |
| 07 | Claim processing | Mihai | 0:50 | Decompose → NLI |
| 08 | SFT + DPO | Vasile | 0:40 | Training pipeline |
| 09 | Verdict-head eval | Vasile | 0:45 | Side-by-side card |
| 10 | Orchestrator | Vasile | 0:35 | Tool calls + tracing |
| 11 | Trained verifier | Vasile | 0:30 | Config + dev metrics |
| 12 | Headline result | Vasile | 0:45 | 0.704 / 0.0% hero |
| 13 | Demo video | Vasile (VO) | ~2:00 | Video plays automatically |
| 14 | Close + links | Vasile | 0:15 | Thanks |

**Per-speaker totals:** Mihai ≈ 4:30 · Vasile ≈ 5:30 (incl. 2:00 demo VO).
**Net runtime:** ~10:00.

---

## Mihai — slides 01–07 (≈ 4:30)

### Slide 01 · Title (0:10)
> *Title shows: project name + four team names.*

"We built an automated fact-checking agent for political claims. I'm Mihai, presenting with Vasile — David and Andrei built parts of the system with us."

→ Advance.

### Slide 02 · The Problem (0:35)
> *Two failure-mode panels appear, then the abstention principle.*

"Automated fact-checkers fail in two ways. Aggressive classifiers over-flag — false positives drown the real signal. Confident LLMs do the opposite: they generate verdicts that sound right and cite sources that don't exist. We built this agent around one principle — **abstain rather than fabricate**. If the evidence isn't there, the system says so explicitly."

→ Advance.

### Slide 03 · State of the Art (0:45)
> *Comparison matrix: FEVER baselines, LLM-only, LLM+retrieval, Ours.*

"This is the prior-work landscape. **FEVER baselines** — DrQA, ESIM — retrieve and classify, but they're brittle outside Wikipedia and they don't calibrate. **LLM-only systems** like a vanilla GPT prompt produce fluent verdicts and hallucinated citations. **Recent tool-using fact-checkers** like Factool and Loki retrieve evidence and emit free-form citations — better, but the citation guarantee is best-effort, not enforced. **Our contribution** is the bottom row: a tool-augmented orchestrator plus a trained discriminative verifier, with citations restricted to retrieved passage IDs by construction and verdict probabilities temperature-scaled on a held-out dev split."

→ Advance.

### Slide 04 · Architecture (0:40)
> *Five tool boxes appear left-to-right; owner pills colour-code the work.*

"Five tools, one orchestrator. A claim enters the pipeline, the **decomposer** splits it into atomic sub-claims, the **retriever** pulls passages from an indexed corpus, the **stance classifier** labels each (claim, passage) pair, the **credibility scorer** weights sources, and the **verifier** emits the final 3-way verdict. The orchestrator decides which tool to call at each step and aggregates the evidence. The audit trail — claim → passages → stance → verdict — is the point."

→ Advance.

### Slide 05 · Corpus & filter (0:50)
> *Funnel: 5.4M → 263K → 662K. Amber caveat box on the right.*

"Data and ingestion. We work with FEVER plus LIAR/PolitiFact — about 158 thousand claims total. The catch: the full FEVER Wikipedia dump is 5.4 million pages, far too much to embed and serve cheaply. So we filter on political-relevance keywords — politicians, institutions, elections, policies — keep ~263 thousand pages, chunk them at ~500 characters with MiniLM-L6 embeddings, and end up with 662 thousand passages in ChromaDB. The honest trade-off is selection bias: claims outside the political scope retrieve worse. Phase 1 scope is political claims, and that's exactly why."

→ Advance.

### Slide 06 · Retrieval recall (0:40)
> *Bar chart animates: tiny grey dense-only bars, tall cyan hybrid bars.*

"Dense embeddings alone weren't enough. **Recall@10 was 0.023** — the right Wikipedia page almost never made the top results. The fix was hybrid retrieval: combine a FEVER title-lookup index with dense MiniLM embeddings, take 50 candidates, and rerank with a cross-encoder. **Recall@10 jumped to 0.857.** That single change unlocked everything downstream — without it, the verifier has nothing to verify against."

→ Advance.

### Slide 07 · Claim processing (0:50)
> *Two-step diagram: top — decomposer tree; bottom — NLI flow with softmax bars.*

"Once we have the right evidence, we still have to make sense of it. Two steps. **Decompose:** the compound claim 'Obama, elected in 2008, signed the ACA in 2010' splits into two atomic sub-claims that can be verified independently. We use an Ollama-hosted instruction-tuned LLM with a deterministic rule-based fallback. **Stance:** for every (atomic claim, retrieved passage) pair, a DeBERTa-v3 NLI model emits SUPPORTS, REFUTES, or NEUTRAL — here, 0.97 on SUPPORTS for the Obama example. Every classification carries the passage ID it was generated for; citations cannot escape the retrieved set."

→ HANDOFF: "Per-claim labels and per-passage stance are the training signal — and that's where Vasile picks up." *Step aside; Vasile takes the clicker.*

---

## Vasile — slides 08–14 (≈ 5:30 incl. demo VO)

### Slide 08 · SFT + DPO (0:40)
> *Five-stage pipeline: triples → SFT → preference pairs → DPO → LoRA adapter.*

"Model training. We took TinyLlama-1.1B and fine-tuned it with LoRA on 126 thousand claim–evidence–verdict triples. Two stages: **SFT** to teach the verdict format and the citation grammar; then **DPO** — preference learning, where the model is rewarded for well-cited, conservative verdicts and penalised for confident-but-unsupported ones. The system prompt explicitly asks for passage-ID citations and NOT_ENOUGH_INFO when evidence is thin. The adapter is published on Hugging Face."

→ Advance.

### Slide 09 · Verdict-head eval (0:45)
> *Side-by-side comparison card: TinyLlama+DPO vs DeBERTa verifier.*

"Then we evaluated. On an abstention smoke test — an Everest claim given Mariana Trench evidence — the raw TinyLlama returned SUPPORTED, so we wrapped inference with a deterministic citation-and-evidence guardrail. In parallel, we trained a DeBERTa verifier as a discriminative second verdict head. On FEVER dev it hit **0.819 accuracy** — more reliable than the generative head alone. The lesson is on the slide: generative heads produce fluent verdicts; discriminative heads produce calibrated 3-way labels. For this task, on this dataset, the data picked the discriminative one — so we promoted DeBERTa to the verdict role and kept TinyLlama for the conversational explanation layer."

→ Advance.

### Slide 10 · Orchestrator (0:35)
> *Orchestrator state diagram on the left; citation tracing + credibility scorer cards on the right.*

"The orchestrator wires the tools together — runs them in sequence, accumulates evidence, and the synthesiser aggregates stance and credibility into a verdict with explicit citations. The reason the hallucination rate is zero is right here: **cited passage IDs ⊆ retrieved passage IDs, enforced in the synthesiser.** Source relevance defaults to 0.5 when metadata is missing — that prevents the weak-source guard from silently disengaging on non-FEVER data."

→ Advance.

### Slide 11 · Trained verifier (0:30)
> *Training-config table on the left; dev metrics + HF handle on the right.*

"The verifier specifics: DeBERTa-v3-base, fine-tuned on 20 thousand FEVER train triples for two epochs with class weighting and fp16. Dev set: 0.819 accuracy, 0.795 macro F1. We then fit a single temperature scalar on the dev split to calibrate confidence, written out as a sidecar JSON the agent loads automatically. The model is on Hugging Face."

→ Advance.

### Slide 12 · Headline result (0:45)
> *Hero number 0.704 on the left; 0.0% hallucination badge on the right; journey line animates.*

"End-to-end on **500 decontaminated FEVER dev claims** with held-out threshold tuning: **0.704 accuracy, 0.701 macro F1**. And the headline — **zero percent hallucinated citations**. Our target was under five percent; we cleared it by construction. The journey at the bottom traces the iteration: 0.380 baseline → title retrieval → cross-encoder reranking → source-aware synthesis → trained verifier."

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

**Claim 3 — NOT ENOUGH INFO (~35 s):**
"Third: *'The next Romanian president after 2025 will lead a center-right coalition.'* The corpus has no evidence about future elections. The verifier returns NOT_ENOUGH_INFO. This is the abstention behaviour we designed for — and the reason the hallucination rate is zero."

**Outro (~5 s):**
"Three claims, three different verdicts, zero hallucinated citations. Thank you."

→ Advance to slide 14.

---

### Slide 14 — Close + links (0:15)

Vasile lets the slide breathe for a beat.

"Code and both Hugging Face models are linked here. Happy to take questions."

---

## Practical notes for the day

**Clicker rotation.** Only one handoff — at slide 07 → 08. Easiest if Vasile steps to the clicker while Mihai is finishing the claim-processing slide. If you're using the phone remote at `<IP>:5173/present`, agree in advance who taps "next" on each slide.

**Slide jumps.** `Alt + 1..9` jumps to slides 1–9 directly. Useful in Q&A.

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
