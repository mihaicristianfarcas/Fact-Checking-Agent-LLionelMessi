"""
Tests for the Verdict Synthesizer.

Run with:
    pytest tests/test_synthesis.py -v
"""

from src.synthesis.verdict_synthesizer import (
    VerdictSynthesizer,
    SynthesisResult,
    VERDICT_SUPPORTED,
    VERDICT_REFUTED,
    VERDICT_NEI,
)
from src.claim_processing.stance_classifier import (
    StanceLabel,
    StanceResult,
    PassageStance,
)


def _ps(
    pid="p1", stance=StanceLabel.SUPPORTING, confidence=0.90,
    retrieval_score=0.85, rank=1, dataset="fever",
) -> PassageStance:
    return PassageStance(
        passage_id=pid,
        passage_text="Evidence text.",
        passage_source="TestPage",
        passage_dataset=dataset,
        retrieval_score=retrieval_score,
        retrieval_rank=rank,
        stance=stance,
        confidence=confidence,
        raw_scores={
            "SUPPORTING": confidence if stance == StanceLabel.SUPPORTING else 0.05,
            "REFUTING": confidence if stance == StanceLabel.REFUTING else 0.05,
            "NEUTRAL": confidence if stance == StanceLabel.NEUTRAL else 0.05,
        },
    )


def _sr(claim: str, passages: list[PassageStance]) -> StanceResult:
    sup = sum(1 for p in passages if p.stance == StanceLabel.SUPPORTING)
    ref = sum(1 for p in passages if p.stance == StanceLabel.REFUTING)
    neu = sum(1 for p in passages if p.stance == StanceLabel.NEUTRAL)
    agg = StanceLabel.SUPPORTING if sup >= ref else StanceLabel.REFUTING if ref > 0 else StanceLabel.NEUTRAL
    return StanceResult(
        claim_text=claim,
        passage_stances=passages,
        aggregate_label=agg,
        aggregate_score=0.8,
        supporting_count=sup,
        refuting_count=ref,
        neutral_count=neu,
        latency_ms=5.0,
        model_name="test",
    )


class TestVerdictSynthesizer:

    def test_single_supported_claim(self):
        synth = VerdictSynthesizer()
        sr = _sr("Claim A.", [
            _ps("p1", StanceLabel.SUPPORTING, 0.90),
            _ps("p2", StanceLabel.SUPPORTING, 0.85),
        ])
        result = synth.synthesize("Claim A.", ["Claim A."], [sr])

        assert isinstance(result, SynthesisResult)
        assert result.verdict == VERDICT_SUPPORTED
        assert result.confidence > 0
        assert "p1" in result.cited_passage_ids or "p2" in result.cited_passage_ids

    def test_refuted_overrides_supported(self):
        synth = VerdictSynthesizer(refute_overrides=True)
        sr_sup = _sr("Sub A.", [_ps("p1", StanceLabel.SUPPORTING, 0.90)])
        sr_ref = _sr("Sub B.", [_ps("p2", StanceLabel.REFUTING, 0.92)])

        result = synth.synthesize(
            "Compound claim.",
            ["Sub A.", "Sub B."],
            [sr_sup, sr_ref],
        )
        assert result.verdict == VERDICT_REFUTED

    def test_thin_refuting_signal_returns_nei(self):
        synth = VerdictSynthesizer()
        sr = _sr("Claim.", [
            _ps("n1", StanceLabel.NEUTRAL, 0.95, retrieval_score=0.70, rank=1),
            _ps("n2", StanceLabel.NEUTRAL, 0.96, retrieval_score=0.68, rank=2),
            _ps("n3", StanceLabel.NEUTRAL, 0.94, retrieval_score=0.66, rank=3),
            _ps("n4", StanceLabel.NEUTRAL, 0.93, retrieval_score=0.64, rank=4),
            _ps("r1", StanceLabel.REFUTING, 0.55, retrieval_score=0.58, rank=5),
        ])

        result = synth.synthesize("Claim.", ["Claim."], [sr])
        assert result.verdict == VERDICT_NEI

    def test_no_evidence_returns_nei(self):
        synth = VerdictSynthesizer()
        sr = StanceResult(
            claim_text="No evidence claim.",
            passage_stances=[],
            aggregate_label=StanceLabel.NEUTRAL,
            aggregate_score=0.0,
            supporting_count=0,
            refuting_count=0,
            neutral_count=0,
            latency_ms=0.0,
            model_name="test",
        )
        result = synth.synthesize("No evidence.", ["No evidence."], [sr])
        assert result.verdict == VERDICT_NEI
        assert result.confidence == 0.0

    def test_all_neutral_returns_nei(self):
        synth = VerdictSynthesizer()
        sr = _sr("Neutral claim.", [
            _ps("p1", StanceLabel.NEUTRAL, 0.60),
            _ps("p2", StanceLabel.NEUTRAL, 0.55),
        ])
        result = synth.synthesize("Neutral.", ["Neutral."], [sr])
        assert result.verdict == VERDICT_NEI

    def test_no_hallucinated_citations(self):
        synth = VerdictSynthesizer()
        sr = _sr("Claim.", [
            _ps("p1", StanceLabel.SUPPORTING, 0.90),
        ])
        result = synth.synthesize("Claim.", ["Claim."], [sr])
        assert result.hallucinated_citations == []
        assert result.hallucination_rate == 0.0

    def test_citation_present_for_non_nei(self):
        synth = VerdictSynthesizer()
        sr = _sr("Claim.", [
            _ps("p1", StanceLabel.SUPPORTING, 0.90),
        ])
        result = synth.synthesize("Claim.", ["Claim."], [sr])
        if result.verdict != VERDICT_NEI:
            assert result.citation_present is True

    def test_explanation_contains_citations(self):
        synth = VerdictSynthesizer()
        sr = _sr("Claim.", [_ps("p1", StanceLabel.SUPPORTING, 0.90)])
        result = synth.synthesize("Claim.", ["Claim."], [sr])
        if result.verdict == VERDICT_SUPPORTED:
            assert "[p1]" in result.explanation

    def test_nei_explanation_is_informative(self):
        synth = VerdictSynthesizer()
        sr = StanceResult(
            claim_text="NEI claim.",
            passage_stances=[],
            aggregate_label=StanceLabel.NEUTRAL,
            aggregate_score=0.0,
            supporting_count=0,
            refuting_count=0,
            neutral_count=0,
            latency_ms=0.0,
            model_name="test",
        )
        result = synth.synthesize("NEI.", ["NEI."], [sr])
        assert "not contain sufficient" in result.explanation.lower() or "insufficient" in result.explanation.lower()

    def test_to_dict_serializable(self):
        import json
        synth = VerdictSynthesizer()
        sr = _sr("Claim.", [_ps("p1", StanceLabel.SUPPORTING, 0.90)])
        result = synth.synthesize("Claim.", ["Claim."], [sr])
        d = result.to_dict()
        serialized = json.dumps(d)
        assert '"verdict"' in serialized

    def test_multiple_atomic_claims(self):
        synth = VerdictSynthesizer()
        sr1 = _sr("Sub A.", [_ps("p1", StanceLabel.SUPPORTING, 0.90)])
        sr2 = _sr("Sub B.", [_ps("p2", StanceLabel.SUPPORTING, 0.85)])
        result = synth.synthesize("A and B.", ["Sub A.", "Sub B."], [sr1, sr2])
        assert result.verdict == VERDICT_SUPPORTED
        assert len(result.atomic_verdicts) == 2

    def test_any_nei_in_compound_returns_nei(self):
        synth = VerdictSynthesizer()
        sr_supported = _sr("Sub A.", [_ps("p1", StanceLabel.SUPPORTING, 0.90)])
        sr_nei = _sr("Sub B.", [
            _ps("n1", StanceLabel.NEUTRAL, 0.92),
            _ps("n2", StanceLabel.NEUTRAL, 0.91, rank=2),
        ])

        result = synth.synthesize(
            "Sub A and Sub B.",
            ["Sub A.", "Sub B."],
            [sr_supported, sr_nei],
        )
        assert result.verdict == VERDICT_NEI

    def test_limits_citations_per_atomic_claim(self):
        synth = VerdictSynthesizer()
        sr = _sr("Claim.", [
            _ps("p1", StanceLabel.SUPPORTING, 0.95, retrieval_score=0.90, rank=1),
            _ps("p2", StanceLabel.SUPPORTING, 0.94, retrieval_score=0.89, rank=2),
            _ps("p3", StanceLabel.SUPPORTING, 0.93, retrieval_score=0.88, rank=3),
        ])

        # Give each passage a distinct source so source-dedup doesn't collapse them.
        sr.passage_stances[0].passage_source = "SourceA"
        sr.passage_stances[1].passage_source = "SourceB"
        sr.passage_stances[2].passage_source = "SourceC"

        result = synth.synthesize("Claim.", ["Claim."], [sr])
        assert len(result.atomic_verdicts[0].cited_passages) <= 2
