"""Prompt and output helpers for the trained fact-checking model.

This module intentionally has no ML dependencies.  It is used by both the
training data preparation code and lightweight tests for inference formatting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SYSTEM_PROMPT = (
    "You are an expert fact-checking agent. Your task is to review a given "
    "Claim along with a list of Evidence passages. You must output a Verdict "
    "of either SUPPORTED, REFUTED, or NOT_ENOUGH_INFO. You must also provide "
    "a short explanation, citing the specific evidence IDs used (e.g., "
    "[source_id]). If the evidence does not clearly support or refute the "
    "claim, you must choose NOT_ENOUGH_INFO and explain why. Use only the "
    "provided evidence, never world knowledge. If evidence is about a different "
    "topic than the claim, output NOT_ENOUGH_INFO."
)

SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
VERDICT_LABELS = (SUPPORTED, REFUTED, NOT_ENOUGH_INFO)

_LABEL_ALIASES = {
    "SUPPORTS": SUPPORTED,
    "SUPPORTED": SUPPORTED,
    "SUPPORTING": SUPPORTED,
    "REFUTES": REFUTED,
    "REFUTED": REFUTED,
    "REFUTING": REFUTED,
    "NOT ENOUGH INFO": NOT_ENOUGH_INFO,
    "NOT_ENOUGH_INFO": NOT_ENOUGH_INFO,
    "NOT-ENOUGH-INFO": NOT_ENOUGH_INFO,
    "NEI": NOT_ENOUGH_INFO,
}


@dataclass(frozen=True)
class ParsedModelOutput:
    """Structured view of the trained model's generated text."""

    raw_text: str
    verdict: str | None
    explanation: str
    citations: list[str]


def _field(item: Mapping[str, Any] | Any, name: str, default: Any = "") -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def build_user_prompt(
    claim: str,
    evidence_passages: Sequence[Mapping[str, Any] | Any],
) -> str:
    """Build the user prompt containing the claim and retrieved evidence."""
    prompt = f"Claim: {claim}\n\nEvidence:\n"
    if not evidence_passages:
        return prompt + "None.\n"

    for evidence in evidence_passages:
        evidence_id = _field(evidence, "id", "unknown")
        text = _field(evidence, "text", "")
        prompt += f"[{evidence_id}]: {text}\n"
    return prompt


def normalize_verdict(label: str | None) -> str | None:
    """Normalize FEVER/model labels into the project verdict vocabulary."""
    if label is None:
        return None
    key = re.sub(r"\s+", " ", str(label).strip().upper().replace("_", " "))
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    hyphen_key = key.replace(" ", "-")
    return _LABEL_ALIASES.get(hyphen_key)


def extract_citations(text: str) -> list[str]:
    """Return citation IDs used in bracket form, preserving first-seen order."""
    seen: set[str] = set()
    citations: list[str] = []
    for match in re.finditer(r"\[([^\[\]\n]+)\]", text):
        citation = match.group(1).strip()
        if citation and citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations


def parse_model_output(text: str) -> ParsedModelOutput:
    """Parse verdict, explanation, and citations from generated model text."""
    verdict: str | None = None
    verdict_match = re.search(
        r"verdict\s*:\s*(SUPPORTED|REFUTED|NOT[_\s-]+ENOUGH[_\s-]+INFO|NEI)",
        text,
        flags=re.IGNORECASE,
    )
    if verdict_match:
        verdict = normalize_verdict(verdict_match.group(1))
    else:
        fallback_patterns = [
            (SUPPORTED, r"\b(SUPPORTED|SUPPORTS|SUPPORTING)\b"),
            (REFUTED, r"\b(REFUTED|REFUTES|REFUTING)\b"),
            (
                NOT_ENOUGH_INFO,
                r"\b(NOT[_\s-]+ENOUGH[_\s-]+INFO|NEI)\b",
            ),
        ]
        for label, pattern in fallback_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                verdict = label
                break

    explanation = ""
    explanation_match = re.search(
        r"explanation\s*:\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if explanation_match:
        explanation = explanation_match.group(1).strip()

    return ParsedModelOutput(
        raw_text=text,
        verdict=verdict,
        explanation=explanation,
        citations=extract_citations(text),
    )
