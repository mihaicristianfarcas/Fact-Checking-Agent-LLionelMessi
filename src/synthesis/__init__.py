"""Verdict synthesis for the fact-checking pipeline."""

from .verdict_synthesizer import VerdictSynthesizer, SynthesisResult, AtomicVerdict

__all__ = [
    "VerdictSynthesizer",
    "SynthesisResult",
    "AtomicVerdict",
]
