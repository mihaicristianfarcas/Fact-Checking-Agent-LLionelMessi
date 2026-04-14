"""
Fact-Checking Agent Orchestrator.

Runs the full pipeline:

    claim → decompose → retrieve → stance classify → synthesize (with credibility scoring)

The orchestrator owns the "agent loop" described in the RFC.  It coordinates
decomposition, retrieval, stance classification, and synthesis (which
internally applies credibility scoring) and produces a fully-traced result.

Two execution modes:
    1. **Deterministic** (default) — runs the fixed pipeline in order.
       Safest, fastest, and always reproducible.
    2. **Adaptive** — uses lightweight heuristics to retrieve extra
       evidence when initial retrieval scores are low.  No LLM router;
       just rule-based branching that still produces identical traces
       for identical inputs.

The PipelineTrace captures every intermediate result so downstream
evaluation and debugging can inspect exactly what happened.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.claim_processing.decomposer import ClaimDecomposer, DecompositionResult
from src.claim_processing.stance_classifier import StanceClassifier, StanceResult
from src.data_ingestion.retriever.evidence_retriever import (
    EvidenceRetriever,
    RetrievalResult,
)
from src.scoring.credibility_scorer import CredibilityScorer
from src.synthesis.verdict_synthesizer import SynthesisResult, VerdictSynthesizer

logger = logging.getLogger(__name__)


@dataclass
class PipelineTrace:
    """Full execution trace of a single fact-check run.

    Captures every intermediate for evaluation, debugging, and presentation.
    """

    original_claim: str
    decomposition: Optional[DecompositionResult] = None
    retrievals: dict[str, list[RetrievalResult]] = field(default_factory=dict)
    stance_results: dict[str, StanceResult] = field(default_factory=dict)
    synthesis: Optional[SynthesisResult] = None
    total_latency_ms: float = 0.0
    steps_executed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        retrieval_summary = {}
        for claim_text, results in self.retrievals.items():
            retrieval_summary[claim_text] = [
                {
                    "passage_id": r.passage.id,
                    "source": r.passage.source,
                    "score": round(r.score, 4),
                    "rank": r.rank,
                    "text_preview": r.passage.text[:120],
                }
                for r in results
            ]

        stance_summary = {
            ct: sr.to_dict() for ct, sr in self.stance_results.items()
        }

        return {
            "original_claim": self.original_claim,
            "decomposition": {
                "was_compound": self.decomposition.was_compound,
                "atomic_claims": self.decomposition.texts,
                "model_used": self.decomposition.model_used,
                "latency_ms": round(self.decomposition.latency_ms, 1),
            }
            if self.decomposition
            else None,
            "retrievals": retrieval_summary,
            "stance_results": stance_summary,
            "synthesis": self.synthesis.to_dict() if self.synthesis else None,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "steps_executed": self.steps_executed,
        }


class FactCheckAgent:
    """End-to-end fact-checking agent.

    Orchestrates decomposition, retrieval, stance classification, credibility
    scoring, and verdict synthesis into a single call.

    Args:
        decomposer          : ClaimDecomposer instance (or None to create one).
        retriever           : EvidenceRetriever instance (or None to create one).
        stance_classifier   : StanceClassifier instance (or None to create one).
        credibility_scorer  : CredibilityScorer instance (or None to create one).
        synthesizer         : VerdictSynthesizer instance (or None to create one).
        top_k               : Number of passages to retrieve per atomic claim.
        adaptive            : Enable adaptive routing heuristics.
        low_score_threshold : In adaptive mode, retrieve extra if top score is
                              below this threshold.
    """

    def __init__(
        self,
        decomposer: Optional[ClaimDecomposer] = None,
        retriever: Optional[EvidenceRetriever] = None,
        stance_classifier: Optional[StanceClassifier] = None,
        credibility_scorer: Optional[CredibilityScorer] = None,
        synthesizer: Optional[VerdictSynthesizer] = None,
        top_k: int = 5,
        adaptive: bool = False,
        low_score_threshold: float = 0.30,
    ) -> None:
        self.decomposer = decomposer
        self.retriever = retriever
        self.stance_classifier = stance_classifier
        self.credibility_scorer = credibility_scorer or CredibilityScorer()
        self.synthesizer = synthesizer or VerdictSynthesizer(
            credibility_scorer=self.credibility_scorer
        )
        self.top_k = top_k
        self.adaptive = adaptive
        self.low_score_threshold = low_score_threshold

        self._components_initialized = False

    def _lazy_init(self) -> None:
        """Lazily create heavy components so import is fast."""
        if self._components_initialized:
            return

        if self.decomposer is None:
            logger.info("Initializing ClaimDecomposer...")
            self.decomposer = ClaimDecomposer()

        if self.retriever is None:
            logger.info("Initializing EvidenceRetriever...")
            self.retriever = EvidenceRetriever()

        if self.stance_classifier is None:
            logger.info("Initializing StanceClassifier...")
            self.stance_classifier = StanceClassifier()

        self._components_initialized = True

    def check(self, claim: str) -> SynthesisResult:
        """Fact-check a claim and return the final verdict.

        This is the main public entry point — one call, full pipeline.
        """
        trace = self.check_with_trace(claim)
        return trace.synthesis

    def check_with_trace(self, claim: str) -> PipelineTrace:
        """Fact-check a claim and return the full execution trace."""
        self._lazy_init()
        start = time.perf_counter()

        trace = PipelineTrace(original_claim=claim)

        # ── Step 1: Decompose ─────────────────────────────────────────────
        decomposition = self.decomposer.decompose(claim)
        trace.decomposition = decomposition
        trace.steps_executed.append("decompose")
        atomic_texts = decomposition.texts
        logger.info(
            "Decomposed into %d atomic claim(s).", len(atomic_texts)
        )

        # ── Step 2: Retrieve ──────────────────────────────────────────────
        for ac_text in atomic_texts:
            results = self.retriever.retrieve(ac_text, top_k=self.top_k)

            if (
                self.adaptive
                and results
                and results[0].score < self.low_score_threshold
            ):
                logger.info(
                    "Adaptive: low top score (%.3f) for %r, doubling top_k.",
                    results[0].score,
                    ac_text[:60],
                )
                results = self.retriever.retrieve(
                    ac_text, top_k=self.top_k * 2
                )

            trace.retrievals[ac_text] = results
        trace.steps_executed.append("retrieve")

        # ── Step 3: Stance classify ───────────────────────────────────────
        stance_results: list[StanceResult] = []
        for ac_text in atomic_texts:
            retrievals = trace.retrievals[ac_text]
            sr = self.stance_classifier.classify(ac_text, retrievals)
            trace.stance_results[ac_text] = sr
            stance_results.append(sr)
        trace.steps_executed.append("stance_classify")

        # ── Step 4: Synthesize (credibility scoring happens inside) ───────
        synthesis = self.synthesizer.synthesize(
            original_claim=claim,
            atomic_texts=atomic_texts,
            stance_results=stance_results,
        )
        trace.synthesis = synthesis
        trace.steps_executed.append("synthesize")

        trace.total_latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Pipeline complete: %s (conf=%.3f) in %.0f ms",
            synthesis.verdict,
            synthesis.confidence,
            trace.total_latency_ms,
        )
        return trace

    def check_batch(self, claims: list[str]) -> list[SynthesisResult]:
        """Fact-check multiple claims sequentially."""
        return [self.check(c) for c in claims]

    def check_batch_with_traces(
        self, claims: list[str]
    ) -> list[PipelineTrace]:
        """Fact-check multiple claims and return all traces."""
        return [self.check_with_trace(c) for c in claims]
