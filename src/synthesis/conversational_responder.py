"""Conversational responder.

Wraps a small local instruction-tuned chat model that turns the structured
output of the fact-checking pipeline (verdict + evidence) into a friendly
streamed reply. The pipeline still does all of the verification work — this
module only narrates the result.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from threading import Thread
from typing import Any, Iterator, Optional

from src.agent.orchestrator import PipelineTrace
from src.data_ingestion.retriever.evidence_retriever import RetrievalResult
from src.synthesis.verdict_synthesizer import SynthesisResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

VERDICT_PHRASES = {
    "SUPPORTED": "supported by the evidence",
    "REFUTED": "contradicted by the evidence",
    "NOT_ENOUGH_INFO": "not verifiable from the available evidence",
}

SYSTEM_PROMPT = (
    "You are a fact-checking assistant. A specialised verification pipeline "
    "has already classified the user's claim and retrieved the supporting "
    "evidence passages from a fixed corpus. You do NOT make the verdict — "
    "you deliver it to the user, anchored to the exact evidence the "
    "pipeline used.\n\n"
    "Strict grounding rules:\n"
    "- Use ONLY the evidence passages provided in the user message. Treat "
    "your own pretraining knowledge as off-limits for this task.\n"
    "- Do not invent or paraphrase-into-existence any fact, date, place, "
    "number, name, source, or quotation that is not present in the "
    "provided evidence.\n"
    "- If a detail you would naturally want to mention is not in the "
    "evidence, omit it. Better to be brief than to drift.\n"
    "- If the evidence section is empty, say so plainly and stop.\n"
    "- The pipeline's verdict reflects its own statistical signal, not "
    "necessarily what the quoted snippets say. If the snippets do not "
    "actually justify the verdict, say so plainly rather than inventing "
    "a justification.\n\n"
    "Addressing the claim as written (CRITICAL — read carefully):\n"
    "- Judge the user's claim as it is written, including any negation. "
    "A claim like \"X is not Y\" is the OPPOSITE of \"X is Y\"; do not "
    "conflate them.\n"
    "- If the pipeline verdict is SUPPORTED, the claim itself is true. "
    "Confirm it. Do not start your reply with \"No\".\n"
    "- If the pipeline verdict is REFUTED, the claim itself is false. "
    "Tell the user the claim is incorrect and state what the evidence "
    "actually shows. Do NOT start your reply with \"Yes\" — that reads "
    "as agreement with a false claim. Open with phrasing like \"No,\", "
    "\"Actually,\", or \"That's not correct —\".\n"
    "- Never write a sentence of the form \"Yes, <restatement that "
    "contradicts the user's claim>\". The opening word must match the "
    "truth value of the user's claim as written.\n\n"
    "Required output format (always, in this exact order):\n"
    "1. A 1-2 sentence conversational verdict in plain language. State "
    "whether the claim is supported or refuted, and name the source(s) "
    "you are relying on (e.g. the Wikipedia article title shown in "
    "parentheses before each evidence line).\n"
    "2. A blank line, then the line `Evidence:`.\n"
    "3. Between 1 and 3 bullet lines, each in the form:\n"
    "       - (Source name) \"verbatim quote from the provided evidence\"\n"
    "   Copy the quoted text exactly as it appears in the evidence — do "
    "not paraphrase, summarise, translate, or fix typos inside the "
    "quotes. Pick the snippets that most directly address the claim.\n\n"
    "If you cannot find any provided snippet that supports the verdict, "
    "say so explicitly instead of fabricating one.\n\n"
    "Tone: friendly and conversational, never clinical. Match the user's "
    "language. Do not output JSON, XML, headings, or labels other than "
    "the literal `Evidence:` line described above.\n\n"
    "Worked example A — SUPPORTED claim (follow this format exactly):\n"
    "----\n"
    "User's claim: \"Lionel Messi plays for FC Barcelona.\"\n"
    "Pipeline verdict: SUPPORTED (high-confidence; supported by the "
    "evidence).\n"
    "Evidence retrieved from the corpus:\n"
    "- (Lionel_Messi) \"Lionel Messi is an Argentine professional "
    "footballer who plays as a forward for the Spanish club FC "
    "Barcelona.\"\n\n"
    "Your reply:\n"
    "Yes — according to the Wikipedia article on Lionel Messi, that's "
    "right.\n\n"
    "Evidence:\n"
    "- (Lionel_Messi) \"Lionel Messi is an Argentine professional "
    "footballer who plays as a forward for the Spanish club FC "
    "Barcelona.\"\n"
    "----\n\n"
    "Worked example B — REFUTED claim with a negation (note the opener):\n"
    "----\n"
    "User's claim: \"Lionel Messi is not from Argentina.\"\n"
    "Pipeline verdict: REFUTED (high-confidence; contradicted by the "
    "evidence).\n"
    "Evidence retrieved from the corpus:\n"
    "- (Lionel_Messi) \"Lionel Messi is an Argentine professional "
    "footballer.\"\n\n"
    "Your reply:\n"
    "No, that claim is incorrect — according to the Wikipedia article "
    "on Lionel Messi, he is Argentine.\n\n"
    "Evidence:\n"
    "- (Lionel_Messi) \"Lionel Messi is an Argentine professional "
    "footballer.\"\n"
    "----"
)


def _stream_insufficient_evidence_template(
    claim: str, pipeline_verdict: str
) -> Iterator[str]:
    """Deterministic reply when no usable evidence survives the quality filter.

    Bypasses the chat model entirely so it cannot fabricate a justification
    around a verdict that nothing in the corpus actually supports — the
    Macron->Iran failure mode where retrieval returned topically-adjacent
    but not claim-addressing pages.
    """
    body = (
        "I couldn't find any usable evidence in the corpus that directly "
        f"addresses this claim, so I can't verify it. The pipeline reported "
        f"`{pipeline_verdict}`, but with no quotable supporting passages I "
        "would treat that label as unreliable for this question."
    )
    for line in body.splitlines(keepends=True):
        yield line


def _stream_not_enough_info_template(claim: str) -> Iterator[str]:
    """Deterministic reply for a NOT_ENOUGH_INFO verdict.

    By definition the corpus does not contain enough evidence to decide the
    claim, so we do not surface any retrieved snippets — they would either
    mislead the user or invite the chat model to fabricate justification
    around them.
    """
    body = (
        "I don't have enough evidence in the corpus to verify or refute "
        "this claim, so I'll have to leave it as unverified."
    )
    for line in body.splitlines(keepends=True):
        yield line


def confidence_band(confidence: float) -> str:
    """Bucket the pipeline confidence into qualitative bands.

    Exposing a raw 100% to a small chat model encourages it to fabricate a
    confident justification even when the quoted evidence is weak (the
    Macron->Iran failure mode). A coarse band keeps the signal without the
    overconfidence cue.
    """
    if confidence >= 0.75:
        return "high-confidence"
    if confidence >= 0.55:
        return "moderate-confidence"
    return "low-confidence"


_QUALITY_MIN_ALPHA_CHARS = 30
_QUALITY_MIN_UNIQUE_TOKEN_RATIO = 0.45
_QUALITY_MAX_REPETITION_RATIO = 0.35


def is_low_quality_passage(text: str) -> bool:
    """Detect FEVER infobox-derived garbage like ``NATO NATO Russia Russia.``.

    These passages survive retrieval because they keyword-match the claim,
    but they contain no real information and confuse the chat model into
    inventing narratives around meaningless repetition. Drop them before
    they reach the prompt.

    Heuristic gates (any one triggers rejection):
    - too few alphabetic characters,
    - low unique-token ratio (the same token repeated),
    - one token dominates (high single-token frequency),
    - leading token is a 1-2 character orphan suggesting mid-word truncation.
    """
    if not text or not text.strip():
        return True

    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars < _QUALITY_MIN_ALPHA_CHARS:
        return True

    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z'-]*", text)]
    if len(tokens) < 5:
        return True

    first = tokens[0]
    if len(first) <= 2 and first.lower() not in {
        "a", "an", "i", "in", "on", "of", "to", "is", "it", "as", "at",
        "by", "or", "no", "we", "he", "us",
    }:
        return True

    lowered = [t.lower() for t in tokens]
    unique_ratio = len(set(lowered)) / len(lowered)
    if unique_ratio < _QUALITY_MIN_UNIQUE_TOKEN_RATIO:
        return True

    counts: dict[str, int] = {}
    for tok in lowered:
        counts[tok] = counts.get(tok, 0) + 1
    top_freq = max(counts.values())
    if top_freq / len(lowered) > _QUALITY_MAX_REPETITION_RATIO:
        return True

    return False


@dataclass(frozen=True)
class EvidenceSnippet:
    """A trimmed evidence passage used in the responder prompt."""

    source: str
    text: str
    passage_id: str


def select_evidence_snippets(
    trace: PipelineTrace,
    *,
    max_snippets: int = 4,
    max_chars_per_snippet: int = 350,
    drop_low_quality: bool = True,
) -> list[EvidenceSnippet]:
    """Pick the top retrieved passages across atomic claims, deduped by id.

    Preference order: (1) passages cited by the synthesis result, (2) the
    highest-scoring retrieved passages. Truncates each snippet to keep the
    prompt small enough for a 1-2B model context window.

    ``drop_low_quality`` filters FEVER-infobox-style garbage by default.
    Callers can disable it as a fallback when the strict pass leaves the
    responder with nothing to ground on for a verdict the pipeline is
    otherwise confident about.
    """
    cited_ids = list(trace.synthesis.cited_passage_ids) if trace.synthesis else []
    cited_set = set(cited_ids)

    all_retrievals: list[RetrievalResult] = []
    for results in trace.retrievals.values():
        all_retrievals.extend(results)

    by_id: dict[str, RetrievalResult] = {}
    for r in all_retrievals:
        existing = by_id.get(r.passage.id)
        if existing is None or r.score > existing.score:
            by_id[r.passage.id] = r

    cited_first = [by_id[pid] for pid in cited_ids if pid in by_id]
    remaining = sorted(
        (r for pid, r in by_id.items() if pid not in cited_set),
        key=lambda r: r.score,
        reverse=True,
    )
    ordered = cited_first + remaining

    snippets: list[EvidenceSnippet] = []
    for r in ordered:
        if len(snippets) >= max_snippets:
            break
        text = r.passage.text.strip()
        if drop_low_quality and is_low_quality_passage(text):
            logger.debug(
                "Dropping low-quality passage %s from responder prompt.",
                r.passage.id,
            )
            continue
        if not text:
            continue
        if len(text) > max_chars_per_snippet:
            text = text[: max_chars_per_snippet - 3].rstrip() + "..."
        snippets.append(
            EvidenceSnippet(
                source=r.passage.source or r.passage.id,
                text=text,
                passage_id=r.passage.id,
            )
        )
    return snippets


def build_responder_messages(
    claim: str,
    synthesis: SynthesisResult,
    snippets: list[EvidenceSnippet],
) -> list[dict[str, str]]:
    """Build the chat-template messages for the responder model."""
    verdict_label = synthesis.verdict
    verdict_phrase = VERDICT_PHRASES.get(verdict_label, verdict_label.lower())
    band = confidence_band(synthesis.confidence)

    if snippets:
        evidence_block = "\n".join(
            f"- ({s.source}) \"{s.text}\"" for s in snippets
        )
    else:
        evidence_block = "(no evidence retrieved)"

    user_content = (
        f"User's claim: \"{claim}\"\n\n"
        f"Pipeline verdict: {verdict_label} "
        f"({band}; {verdict_phrase}).\n\n"
        f"Evidence retrieved from the corpus:\n{evidence_block}\n\n"
        "Reply to the user using ONLY the evidence above. Follow the "
        "required format: a 1-2 sentence conversational verdict, then a "
        "blank line, then `Evidence:`, then 1-3 verbatim quotes from the "
        "evidence each on its own bullet line in the form "
        "`- (Source) \"quote\"`. Do not invent quotes; copy them as written."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class ConversationalResponder:
    """Streams a friendly reply from a local HF chat model.

    Heavy ML imports are deferred to first use so that prompt-building tests
    can run without ``transformers`` installed for inference.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_id: str = DEFAULT_MODEL_ID,
        max_new_tokens: int = 280,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _lazy_load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading conversational responder model %s", self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            model_kwargs.update(device_map="auto", torch_dtype=torch.float16)
        else:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                model_kwargs.update(
                    device_map={"": "mps"}, torch_dtype=torch.float16
                )
            else:
                model_kwargs.update(torch_dtype=torch.float32)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, **model_kwargs
        )
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

    def stream_response(self, trace: PipelineTrace) -> Iterator[str]:
        """Yield response chunks as the model produces them."""
        if trace.synthesis is None:
            raise ValueError("trace.synthesis must be populated before streaming.")

        if trace.synthesis.verdict == "NOT_ENOUGH_INFO":
            yield from _stream_not_enough_info_template(trace.original_claim)
            return

        snippets = select_evidence_snippets(trace)
        if not snippets:
            # Strict quality filter rejected everything. Rather than surface
            # an "I can't verify it" template that contradicts a confident
            # SUPPORTED/REFUTED verdict, retry without the filter so the
            # model still has something concrete to ground on.
            snippets = select_evidence_snippets(trace, drop_low_quality=False)

        if not snippets:
            yield from _stream_insufficient_evidence_template(
                trace.original_claim, trace.synthesis.verdict
            )
            return

        self._lazy_load()

        import torch
        from transformers import TextIteratorStreamer

        messages = build_responder_messages(
            trace.original_claim, trace.synthesis, snippets
        )

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        do_sample = self.temperature > 0.0
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else 1.0,
            top_p=self.top_p if do_sample else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            thread.join()
