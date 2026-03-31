"""
Responsibility:
    Break a complex, compound user claim into a list of atomic, independently
    verifiable sub-claims.  Each sub-claim is self-contained (no dangling
    pronouns, no implicit subjects) and corresponds to a single verifiable
    proposition.

Model:
    Uses Ollama (gemma3:4b) running locally — no API key, no cost, fits in
    4 GB VRAM.  Install and pull once:
        brew install ollama   (or https://ollama.com)
        ollama pull gemma3:4b

Design Principles:
    • EXTRACTIVE, not generative — the LLM is instructed to only rephrase
      content that is *already present* in the original claim text. This is
      the primary guard against hallucination at the decomposition stage.
    • Structured JSON output — Ollama's format="json" mode constrains the
      model to emit valid JSON, and Pydantic validates the schema before
      the result ever leaves this module.
    • Atomic validation pass — a lightweight heuristic checks that each
      returned sub-claim satisfies a minimum quality bar (non-empty, no
      first-person pronouns, etc.).
    • Graceful fallback — if the Ollama call fails or the output cannot be
      parsed, the module returns the original claim wrapped in a single-item
      list so the pipeline never breaks.

Downstream contract:
    The list[AtomicClaim] returned here is passed directly to the
    EvidenceRetriever (Task A) — each AtomicClaim.text becomes one
    retriever query.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import ollama
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AtomicClaim:
    """
    A single, independently verifiable proposition extracted from a compound
    claim.  Passed downstream to the EvidenceRetriever and later to the
    StanceClassifier.

    Attributes:
        text              : The atomic claim as a clean, standalone sentence.
        source_claim      : The original compound claim this was derived from.
        claim_index       : 0-based position within the decomposition result list.
        claim_id          : Unique ID for tracing through the pipeline.
        extraction_method : How this claim was produced (llm | passthrough).
    """

    text: str
    source_claim: str
    claim_index: int = 0
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    extraction_method: str = "llm"

    def __repr__(self) -> str:
        return (
            f"AtomicClaim(index={self.claim_index}, "
            f"id={self.claim_id!r}, text={self.text!r})"
        )


@dataclass
class DecompositionResult:
    """
    Full output of a single decomposition call.

    Attributes:
        original_claim : The raw input text.
        atomic_claims  : Ordered list of extracted atomic claims.
        was_compound   : True if more than one sub-claim was found.
        model_used     : The Ollama model that produced this result.
        latency_ms     : Wall-clock time for the Ollama call.
        error          : Non-None if a fallback was used.
    """

    original_claim: str
    atomic_claims: list[AtomicClaim]
    was_compound: bool
    model_used: str
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def texts(self) -> list[str]:
        """Convenience accessor — the plain text of each atomic claim."""
        return [ac.text for ac in self.atomic_claims]


# ─────────────────────────────────────────────────────────────────────────────
# Internal Pydantic schema for LLM JSON output validation
# ─────────────────────────────────────────────────────────────────────────────


class _LLMDecompositionOutput(BaseModel):
    """Strict schema the LLM must conform to."""

    atomic_claims: list[str] = Field(
        ...,
        min_length=1,
        description="List of atomic, standalone claim strings.",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of how the claim was split.",
    )

    @field_validator("atomic_claims")
    @classmethod
    def claims_must_be_non_empty(cls, v: list[str]) -> list[str]:
        cleaned = [c.strip() for c in v if c.strip()]
        if not cleaned:
            raise ValueError("atomic_claims must contain at least one non-empty string.")
        return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precision claim-decomposition engine for an automated fact-checking \
system.  Your sole task is to break a compound claim into the smallest possible \
set of atomic, independently verifiable propositions.

STRICT RULES — violating any of these makes your output unusable:
1. EXTRACTIVE ONLY: Every word in each atomic claim must come from the original \
   claim text.  Do NOT add external knowledge, synonyms, or inferences that are \
   not explicitly stated.
2. SELF-CONTAINED: Replace all pronouns and shorthand with the explicit subject \
   from the original claim so each sub-claim can be understood and verified \
   without reading the other sub-claims.
3. ONE FACT PER CLAIM: Each output claim must express exactly one verifiable \
   proposition (e.g., one attribute, one relationship, one event).
4. PRESERVE MEANING: Do not negate, qualify, or rephrase claims in a way that \
   changes their truth conditions.
5. NO META-COMMENTARY: Output only the JSON object — no preamble, no apologies, \
   no markdown fences.
6. ALWAYS SPLIT ON "and": If the claim contains "and" joining two distinct \
   facts, you MUST produce at least two atomic claims. Never merge them into one.

Output format — respond with ONLY a valid JSON object that matches this schema:
{
  "atomic_claims": ["<claim 1>", "<claim 2>", ...],
  "reasoning": "<one sentence explaining the split>"
}

EXAMPLE:
  INPUT:  "Marie Curie was a physicist and won the Nobel Prize."
  OUTPUT: {"atomic_claims": ["Marie Curie was a physicist.", "Marie Curie won the Nobel Prize."], "reasoning": "Two distinct verifiable facts joined by and."}

If the input is already a single atomic claim with no compound structure, \
return it as-is in a one-element list with reasoning "Already atomic."
"""

_USER_PROMPT_TEMPLATE = """\
Decompose the following claim into atomic sub-claims.

CLAIM: {claim}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main Class
# ─────────────────────────────────────────────────────────────────────────────


class ClaimDecomposer:
    """
    Decomposes a compound claim into atomic, independently verifiable
    sub-claims using Ollama (gemma3:4b) running locally.

    Prerequisites:
        ollama pull gemma3:4b   # ~3.3 GB, run once

    Usage::

        decomposer = ClaimDecomposer()
        result = decomposer.decompose(
            "Actor Nikolaj Coster-Waldau worked with Fox and is 50 years old."
        )
        # result.atomic_claims → [
        #   AtomicClaim(text="Nikolaj Coster-Waldau worked with Fox."),
        #   AtomicClaim(text="Nikolaj Coster-Waldau is 50 years old."),
        # ]

        # ── Task A handoff ────────────────────────────────────────────────
        for ac in result.atomic_claims:
            passages = retriever.retrieve(ac.text, top_k=5)
            # → list[RetrievalResult]  (consumed by StanceClassifier next)

    Args:
        model       : Ollama model tag. Default is gemma3:4b (~3.3 GB VRAM).
                      Alternatives that fit in 4 GB: llama3.2:3b, qwen2.5:3b
        max_retries : How many times to retry on transient Ollama errors.
        temperature : Sampling temperature (0 = deterministic; recommended).
        host        : Ollama server URL. Defaults to http://localhost:11434.
    """

    DEFAULT_MODEL = "gemma3:4b"
    _BARE_CLAIM_WORD_THRESHOLD = 15  # words; below this, skip the model call

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_retries: int = 2,
        temperature: float = 0.0,
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self._client = ollama.Client(host=host)

    # ── Public API ────────────────────────────────────────────────────────────

    def decompose(self, claim: str) -> DecompositionResult:
        """
        Decompose *claim* into atomic sub-claims.

        Returns a :class:`DecompositionResult` — never raises.  On any
        unrecoverable error the original claim is wrapped in a single
        :class:`AtomicClaim` (passthrough) so the pipeline can continue.

        Args:
            claim: The raw compound claim text.

        Returns:
            DecompositionResult with one or more AtomicClaim objects.
        """
        claim = claim.strip()
        if not claim:
            logger.warning("ClaimDecomposer received an empty claim.")
            return self._passthrough(claim, error="Empty claim text.")

        # Fast path: already short/atomic — skip the model entirely.
        if self._is_trivially_atomic(claim):
            logger.debug("Claim is trivially atomic — skipping model call.")
            return self._build_result(
                original=claim,
                texts=[claim],
                model="passthrough",
                method="passthrough",
            )

        return self._call_ollama_with_retry(claim)

    def decompose_batch(self, claims: list[str]) -> list[DecompositionResult]:
        """
        Decompose a list of claims sequentially.

        Args:
            claims: List of raw claim strings.

        Returns:
            One DecompositionResult per input claim, in the same order.
        """
        return [self.decompose(c) for c in claims]

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _is_trivially_atomic(self, claim: str) -> bool:
        """
        Heuristic: skip the model if the claim is short AND contains no
        coordinating/adversative signals that suggest compounding.
        """
        word_count = len(claim.split())
        compound_signals = re.search(
            r"\b(and|but|also|additionally|furthermore|moreover|while|whereas)\b|;",
            claim,
            flags=re.IGNORECASE,
        )
        return word_count <= self._BARE_CLAIM_WORD_THRESHOLD and not compound_signals

    def _call_ollama_with_retry(self, claim: str) -> DecompositionResult:
        """
        Call the local Ollama server with exponential back-off on errors.

        Key Ollama parameters used:
            format="json"    — constrains the model to emit valid JSON,
                               which is the main reliability advantage over
                               raw text generation at this model size.
            temperature=0    — deterministic output; important for a
                               fact-checking pipeline.
        """
        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            try:
                start = time.perf_counter()

                response = self._client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _USER_PROMPT_TEMPLATE.format(claim=claim),
                        },
                    ],
                    format="json",               # forces valid JSON output
                    options={"temperature": self.temperature},
                )

                latency_ms = (time.perf_counter() - start) * 1000
                raw_text = response["message"]["content"]

                parsed = self._parse_and_validate(raw_text)
                texts = self._post_process(parsed.atomic_claims, claim)

                result = self._build_result(
                    original=claim,
                    texts=texts,
                    model=self.model,
                    method="llm",
                    latency_ms=latency_ms,
                )
                logger.info(
                    "Decomposed into %d atomic claim(s) in %.0f ms.",
                    len(result.atomic_claims),
                    latency_ms,
                )
                return result

            except (json.JSONDecodeError, ValueError) as exc:
                # Parse/validation errors are deterministic — no point retrying.
                last_error = f"Parse/validation error: {exc}"
                logger.warning(
                    "Attempt %d/%d — %s",
                    attempt + 1, self.max_retries + 1, last_error,
                )
                break

            except ollama.ResponseError as exc:
                # Ollama-specific API error (e.g. model not found, server error).
                last_error = f"Ollama ResponseError: {exc}"
                logger.warning(
                    "Attempt %d/%d — %s",
                    attempt + 1, self.max_retries + 1, last_error,
                )
                # Model not found is not recoverable — don't retry.
                if "not found" in str(exc).lower():
                    logger.error(
                        "Model %r not found. Run: ollama pull %s",
                        self.model, self.model,
                    )
                    break
                time.sleep(2 ** attempt)

            except Exception as exc:
                # Covers ConnectionError when the Ollama daemon is not running.
                last_error = f"Unexpected error: {exc}"
                logger.warning(
                    "Attempt %d/%d — %s",
                    attempt + 1, self.max_retries + 1, last_error,
                )
                time.sleep(2 ** attempt)

        logger.error(
            "ClaimDecomposer falling back to passthrough. Last error: %s", last_error
        )
        return self._passthrough(claim, error=last_error)

    def _parse_and_validate(self, raw_text: str) -> _LLMDecompositionOutput:
        """
        Parse the model's raw string response into the validated Pydantic model.

        Ollama's format="json" ensures clean JSON output, but we strip
        markdown fences defensively just in case.
        """
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.DOTALL
        )
        data = json.loads(cleaned)
        return _LLMDecompositionOutput(**data)

    def _post_process(self, texts: list[str], original: str) -> list[str]:
        """
        Lightweight quality checks on the model-returned claim texts.

        Drops:
        • empty or trivially short strings
        • claims containing first-person pronouns (hallucination signal)

        Applies:
        • terminal punctuation normalisation
        • order-preserving deduplication
        """
        results: list[str] = []
        first_person_re = re.compile(
            r"\b(I|me|my|mine|we|our|ours|us)\b", re.IGNORECASE
        )

        for text in texts:
            text = text.strip().rstrip(".")
            if text:
                text = text + "."  # normalise terminal punctuation

            if not text or len(text) < 5:
                logger.debug("Dropping empty/trivial sub-claim: %r", text)
                continue

            if first_person_re.search(text):
                logger.warning(
                    "Dropping sub-claim with first-person pronoun: %r", text
                )
                continue

            results.append(text)

        # If every sub-claim was filtered, fall back to the original.
        if not results:
            logger.warning("All sub-claims were filtered — using original claim.")
            results = [original]

        # Deduplicate while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for t in results:
            key = t.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(t)

        return deduped

    def _build_result(
        self,
        original: str,
        texts: list[str],
        model: str,
        method: str,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> DecompositionResult:
        atomic_claims = [
            AtomicClaim(
                text=t,
                source_claim=original,
                claim_index=i,
                extraction_method=method,
            )
            for i, t in enumerate(texts)
        ]
        return DecompositionResult(
            original_claim=original,
            atomic_claims=atomic_claims,
            was_compound=len(atomic_claims) > 1,
            model_used=model,
            latency_ms=latency_ms,
            error=error,
        )

    def _passthrough(
        self, claim: str, error: Optional[str] = None
    ) -> DecompositionResult:
        """Return the original claim as a single atomic claim (last-resort fallback)."""
        return self._build_result(
            original=claim,
            texts=[claim] if claim else ["Unknown claim."],
            model="passthrough",
            method="passthrough",
            error=error,
        )