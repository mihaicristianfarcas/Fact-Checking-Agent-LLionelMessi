"""
Verdict Synthesizer.

Combines per-atomic-claim stance results (optionally weighted by source
credibility) into a single final verdict for the original compound claim.

Responsibilities:
    1. Decide a verdict per atomic claim from its StanceResult + credibility.
    2. Aggregate atomic verdicts into one claim-level verdict.
    3. Produce a short cited explanation where every citation is a passage ID
       that appeared in the retrieved evidence — no free-form citations.
    4. Compute a final confidence score calibrated to the strength of evidence.

Design constraints:
    • ONLY cite passage IDs that exist in the retrieval results.
    • Prefer NOT_ENOUGH_INFO over a weak SUPPORTED/REFUTED.
    • A single strong REFUTED atomic claim overrides otherwise-supporting
      evidence (conservative bias matching the RFC's abstention principle).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.claim_processing.stance_classifier import StanceLabel, StanceResult
from src.scoring.credibility_scorer import CredibilityScorer, ScoredPassage

logger = logging.getLogger(__name__)

VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_REFUTED = "REFUTED"
VERDICT_NEI = "NOT_ENOUGH_INFO"


@dataclass
class AtomicVerdict:
    """Verdict for a single atomic claim."""

    claim_text: str
    verdict: str
    confidence: float
    cited_passages: list[str]
    supporting_ids: list[str] = field(default_factory=list)
    refuting_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim_text": self.claim_text,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "cited_passages": self.cited_passages,
            "supporting_ids": self.supporting_ids,
            "refuting_ids": self.refuting_ids,
        }


@dataclass
class SynthesisResult:
    """Final output of the full fact-checking pipeline for one user claim."""

    original_claim: str
    verdict: str
    confidence: float
    explanation: str
    cited_passage_ids: list[str]
    atomic_verdicts: list[AtomicVerdict]
    all_retrieved_ids: list[str]

    @property
    def hallucinated_citations(self) -> list[str]:
        """Citations that do not appear in the retrieved passage set."""
        retrieved_set = set(self.all_retrieved_ids)
        return [cid for cid in self.cited_passage_ids if cid not in retrieved_set]

    @property
    def hallucination_rate(self) -> float:
        if not self.cited_passage_ids:
            return 0.0
        return len(self.hallucinated_citations) / len(self.cited_passage_ids)

    @property
    def citation_present(self) -> bool:
        """NEI verdicts are exempt; otherwise True when at least one citation exists."""
        if self.verdict == VERDICT_NEI:
            return True
        return len(self.cited_passage_ids) > 0

    def to_dict(self) -> dict:
        return {
            "original_claim": self.original_claim,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "cited_passage_ids": self.cited_passage_ids,
            "hallucinated_citations": self.hallucinated_citations,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "atomic_verdicts": [av.to_dict() for av in self.atomic_verdicts],
        }


class VerdictSynthesizer:
    """Aggregates atomic-claim evidence into a single claim-level verdict.

    Args:
        credibility_scorer : Optional pre-built scorer. Created with defaults
                             if None.
        nei_confidence_floor : Minimum aggregated confidence required to commit
                               to SUPPORTED or REFUTED; below this, return NEI.
        refute_overrides    : If True, a single confident REFUTED atomic claim
                              makes the overall verdict REFUTED (conservative).
    """

    def __init__(
        self,
        credibility_scorer: CredibilityScorer | None = None,
        nei_confidence_floor: float = 0.45,
        refute_overrides: bool = True,
        decisive_dominance_floor: float = 0.60,
        max_citations_per_claim: int = 2,
        min_refuting_passages: int = 1,
        single_refute_confidence_floor: float = 0.0,
        refute_source_relevance_floor: float = 0.55,
        refute_conflict_margin: float = 0.15,
        use_decisive_confidence: bool = True,
        min_decisive_weight: float = 0.45,
    ) -> None:
        self.scorer = credibility_scorer or CredibilityScorer()
        self.nei_floor = nei_confidence_floor
        self.refute_overrides = refute_overrides
        self.dominance_floor = decisive_dominance_floor
        self.max_citations_per_claim = max_citations_per_claim
        self.min_refuting_passages = min_refuting_passages
        self.single_refute_confidence_floor = single_refute_confidence_floor
        self.refute_source_relevance_floor = refute_source_relevance_floor
        self.refute_conflict_margin = refute_conflict_margin
        self.use_decisive_confidence = use_decisive_confidence
        self.min_decisive_weight = min_decisive_weight

    def synthesize(
        self,
        original_claim: str,
        atomic_texts: list[str],
        stance_results: list[StanceResult],
    ) -> SynthesisResult:
        """Produce a final verdict from per-atomic-claim stance results.

        Args:
            original_claim  : The raw user claim.
            atomic_texts    : One text per atomic claim.
            stance_results  : One StanceResult per atomic claim (same order).

        Returns:
            SynthesisResult with verdict, confidence, explanation, and citations.
        """
        if len(atomic_texts) != len(stance_results):
            raise ValueError(
                f"atomic_texts ({len(atomic_texts)}) and stance_results "
                f"({len(stance_results)}) must have the same length."
            )

        all_retrieved_ids: list[str] = []
        _retrieved_set: set[str] = set()
        atomic_verdicts: list[AtomicVerdict] = []

        for text, sr in zip(atomic_texts, stance_results):
            scored = self.scorer.score(sr)

            for sp in scored:
                pid = sp.stance.passage_id
                if pid not in _retrieved_set:
                    _retrieved_set.add(pid)
                    all_retrieved_ids.append(pid)

            av = self._atomic_verdict(text, sr, scored)
            atomic_verdicts.append(av)

        verdict, confidence = self._aggregate(atomic_verdicts)
        cited_ids = self._collect_citations(atomic_verdicts)
        explanation = self._build_explanation(
            original_claim, verdict, confidence, atomic_verdicts
        )

        result = SynthesisResult(
            original_claim=original_claim,
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            cited_passage_ids=cited_ids,
            atomic_verdicts=atomic_verdicts,
            all_retrieved_ids=all_retrieved_ids,
        )

        logger.info(
            "Synthesis: %r -> %s (conf=%.3f, %d citations, halluc=%.2f%%)",
            original_claim[:60],
            verdict,
            confidence,
            len(cited_ids),
            result.hallucination_rate * 100,
        )
        return result

    def _atomic_verdict(
        self,
        claim_text: str,
        stance_result: StanceResult,
        scored_passages: list[ScoredPassage],
    ) -> AtomicVerdict:
        """Decide verdict for one atomic claim."""
        if not scored_passages:
            return AtomicVerdict(
                claim_text=claim_text,
                verdict=VERDICT_NEI,
                confidence=0.0,
                cited_passages=[],
            )

        supporting_weight = 0.0
        refuting_weight = 0.0
        total_weight = 0.0
        supporting_ids: list[str] = []
        refuting_ids: list[str] = []
        supporting_passages: list[ScoredPassage] = []
        refuting_passages: list[ScoredPassage] = []

        for sp in scored_passages:
            w = sp.weighted_confidence
            total_weight += w
            if sp.stance.stance == StanceLabel.SUPPORTING:
                supporting_weight += w
                supporting_ids.append(sp.stance.passage_id)
                supporting_passages.append(sp)
            elif sp.stance.stance == StanceLabel.REFUTING:
                refuting_weight += w
                refuting_ids.append(sp.stance.passage_id)
                refuting_passages.append(sp)

        if supporting_weight == 0 and refuting_weight == 0:
            return AtomicVerdict(
                claim_text=claim_text,
                verdict=VERDICT_NEI,
                confidence=0.0,
                cited_passages=[],
            )

        if refuting_weight > supporting_weight:
            verdict = VERDICT_REFUTED
            winner_weight = refuting_weight
            winner_passages = refuting_passages
        else:
            verdict = VERDICT_SUPPORTED
            winner_weight = supporting_weight
            winner_passages = supporting_passages

        decisive_total = supporting_weight + refuting_weight
        dominance = winner_weight / decisive_total if decisive_total else 0.0
        coverage_conf = winner_weight / total_weight if total_weight else 0.0
        if self.use_decisive_confidence:
            strength = min(1.0, winner_weight / max(self.min_decisive_weight, 1e-9))
            conf = min(1.0, 0.70 * dominance + 0.30 * strength)
        else:
            conf = coverage_conf
        cited = self._select_citations(winner_passages)

        # Commit only when the winning evidence is both strong overall and
        # clearly dominates the opposing decisive evidence.
        if (
            conf < self.nei_floor
            or dominance < self.dominance_floor
            or winner_weight < self.min_decisive_weight
        ):
            verdict = VERDICT_NEI
            cited = []

        if verdict == VERDICT_REFUTED:
            verdict, conf, cited = self._guard_weak_source_refute(
                verdict=verdict,
                confidence=conf,
                total_weight=total_weight,
                supporting_weight=supporting_weight,
                refuting_weight=refuting_weight,
                supporting_passages=supporting_passages,
                refuting_passages=refuting_passages,
            )

        if verdict == VERDICT_REFUTED and self.min_refuting_passages > 1:
            strongest_refute = max(
                (sp.stance.confidence for sp in refuting_passages),
                default=0.0,
            )
            if (
                len(refuting_passages) < self.min_refuting_passages
                and strongest_refute < self.single_refute_confidence_floor
            ):
                verdict = VERDICT_NEI
                cited = []

        return AtomicVerdict(
            claim_text=claim_text,
            verdict=verdict,
            confidence=conf,
            cited_passages=cited,
            supporting_ids=supporting_ids,
            refuting_ids=refuting_ids,
        )

    def _aggregate(
        self, atomic_verdicts: list[AtomicVerdict]
    ) -> tuple[str, float]:
        """Combine atomic verdicts into one claim-level verdict."""
        if not atomic_verdicts:
            return VERDICT_NEI, 0.0

        verdicts = [av.verdict for av in atomic_verdicts]

        if self.refute_overrides and VERDICT_REFUTED in verdicts:
            refuted = [
                av for av in atomic_verdicts if av.verdict == VERDICT_REFUTED
            ]
            avg_conf = sum(r.confidence for r in refuted) / len(refuted)
            return VERDICT_REFUTED, avg_conf

        if all(av.verdict == VERDICT_SUPPORTED for av in atomic_verdicts):
            avg_conf = sum(av.confidence for av in atomic_verdicts) / len(atomic_verdicts)
            return VERDICT_SUPPORTED, avg_conf

        if VERDICT_NEI in verdicts:
            return VERDICT_NEI, 0.0

        return VERDICT_NEI, 0.0

    def _select_citations(self, winning_passages: list[ScoredPassage]) -> list[str]:
        """Pick a small set of cleaner citations for explanations."""
        ranked = sorted(
            winning_passages,
            key=lambda sp: (
                self._source_penalty(sp.stance.passage_source),
                -self._source_relevance(sp),
                -sp.weighted_confidence,
                sp.stance.retrieval_rank,
            ),
        )

        selected: list[str] = []
        seen_sources: set[str] = set()
        for sp in ranked:
            source_key = sp.stance.passage_source.lower()
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            selected.append(sp.stance.passage_id)
            if len(selected) >= self.max_citations_per_claim:
                break

        if not selected and ranked:
            selected.append(ranked[0].stance.passage_id)

        return selected

    def _source_penalty(self, source: str) -> int:
        """Deprioritize low-signal sources like disambiguation pages."""
        source_lower = source.lower()
        return int(
            "disambiguation" in source_lower
            or "-lrb-disambiguation-rrb-" in source_lower
        )

    def _guard_weak_source_refute(
        self,
        *,
        verdict: str,
        confidence: float,
        total_weight: float,
        supporting_weight: float,
        refuting_weight: float,
        supporting_passages: list[ScoredPassage],
        refuting_passages: list[ScoredPassage],
    ) -> tuple[str, float, list[str]]:
        """Prevent fuzzy-title contradictions from overriding better sources."""
        if verdict != VERDICT_REFUTED or not refuting_passages:
            return verdict, confidence, self._select_citations(refuting_passages)

        strong_refutes = [
            sp for sp in refuting_passages if self._is_high_relevance_refute(sp)
        ]
        best_refute_rel = max(
            (self._source_relevance(sp) for sp in refuting_passages),
            default=1.0,
        )
        best_support_rel = max(
            (self._source_relevance(sp) for sp in supporting_passages),
            default=0.0,
        )

        # If a stronger source supports the claim, weak fuzzy-title refutations
        # should not flip the verdict unless they have source/rerank quality too.
        if supporting_passages and not strong_refutes:
            support_is_stronger_source = (
                best_support_rel >= best_refute_rel + self.refute_conflict_margin
            )
            support_is_material = supporting_weight >= 0.35 * max(refuting_weight, 1e-9)
            if support_is_stronger_source and support_is_material:
                support_conf = supporting_weight / total_weight if total_weight else 0.0
                decisive_total = supporting_weight + refuting_weight
                support_dominance = (
                    supporting_weight / decisive_total if decisive_total else 0.0
                )
                if support_conf >= self.nei_floor and support_dominance >= self.dominance_floor:
                    return (
                        VERDICT_SUPPORTED,
                        support_conf,
                        self._select_citations(supporting_passages),
                    )
                return VERDICT_NEI, confidence, []

            if best_refute_rel < self.refute_source_relevance_floor:
                return VERDICT_NEI, confidence, []

        # With no counter-support, only the weakest source matches are downgraded.
        if not strong_refutes and best_refute_rel < 0.35:
            return VERDICT_NEI, confidence, []

        return VERDICT_REFUTED, confidence, self._select_citations(refuting_passages)

    def _is_high_relevance_refute(self, sp: ScoredPassage) -> bool:
        relevance = self._source_relevance(sp)
        if relevance < self.refute_source_relevance_floor:
            return False

        metadata = sp.stance.passage_metadata
        rerank_score = _metadata_float(metadata, "rerank_score", sp.stance.retrieval_score)
        title_score = _metadata_float(metadata, "title_score", 0.0)
        return (
            sp.stance.confidence >= 0.80
            or rerank_score >= 0.75
            or title_score >= 0.75
        )

    def _source_relevance(self, sp: ScoredPassage) -> float:
        return _metadata_float(sp.stance.passage_metadata, "source_relevance", 1.0)

    def _collect_citations(self, atomic_verdicts: list[AtomicVerdict]) -> list[str]:
        """Deduplicated list of all cited passage IDs."""
        seen: set[str] = set()
        cited: list[str] = []
        for av in atomic_verdicts:
            for pid in av.cited_passages:
                if pid not in seen:
                    seen.add(pid)
                    cited.append(pid)
        return cited

    def _build_explanation(
        self,
        original_claim: str,
        verdict: str,
        confidence: float,
        atomic_verdicts: list[AtomicVerdict],
    ) -> str:
        """Build a short templated explanation that only cites passage IDs."""
        if verdict == VERDICT_NEI:
            return (
                "The retrieved evidence does not contain sufficient information "
                "to verify or refute this claim."
            )

        parts: list[str] = []
        for av in atomic_verdicts:
            if not av.cited_passages:
                continue
            citations = ", ".join(f"[{pid}]" for pid in av.cited_passages)
            status = "supported" if av.verdict == VERDICT_SUPPORTED else "refuted"
            parts.append(f'"{av.claim_text}" is {status} by {citations}.')

        if not parts:
            return (
                "The retrieved evidence does not contain sufficient information "
                "to verify or refute this claim."
            )

        action = "supported" if verdict == VERDICT_SUPPORTED else "refuted"
        header = f"The claim is {action} (confidence: {confidence:.0%}). "
        return header + " ".join(parts)


def _metadata_float(metadata: dict, key: str, default: float) -> float:
    try:
        return float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default
