"""
Tests for the Source Credibility Scorer.

Run with:
    pytest tests/test_scoring.py -v
"""

from src.scoring.credibility_scorer import CredibilityScorer, ScoredPassage
from src.claim_processing.stance_classifier import StanceLabel, StanceResult, PassageStance


def _make_passage_stance(
    passage_id="p1",
    text="Evidence text.",
    source="TestPage",
    dataset="fever",
    retrieval_score=0.85,
    rank=1,
    stance=StanceLabel.SUPPORTING,
    confidence=0.90,
) -> PassageStance:
    return PassageStance(
        passage_id=passage_id,
        passage_text=text,
        passage_source=source,
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


def _make_stance_result(passage_stances: list[PassageStance]) -> StanceResult:
    sup = sum(1 for p in passage_stances if p.stance == StanceLabel.SUPPORTING)
    ref = sum(1 for p in passage_stances if p.stance == StanceLabel.REFUTING)
    neu = sum(1 for p in passage_stances if p.stance == StanceLabel.NEUTRAL)
    return StanceResult(
        claim_text="Test claim.",
        passage_stances=passage_stances,
        aggregate_label=StanceLabel.SUPPORTING,
        aggregate_score=0.8,
        supporting_count=sup,
        refuting_count=ref,
        neutral_count=neu,
        latency_ms=10.0,
        model_name="test-model",
    )


class TestCredibilityScorer:

    def test_returns_one_scored_per_passage(self):
        ps = [
            _make_passage_stance(passage_id="p1", rank=1),
            _make_passage_stance(passage_id="p2", rank=2),
        ]
        sr = _make_stance_result(ps)
        scorer = CredibilityScorer()
        scored = scorer.score(sr)

        assert len(scored) == 2
        assert all(isinstance(s, ScoredPassage) for s in scored)

    def test_credibility_in_range(self):
        ps = [_make_passage_stance(rank=1, retrieval_score=0.99, confidence=0.99)]
        sr = _make_stance_result(ps)
        scored = CredibilityScorer().score(sr)

        assert 0.0 <= scored[0].credibility <= 1.0

    def test_higher_rank_higher_credibility(self):
        ps1 = _make_passage_stance(passage_id="p1", rank=1, retrieval_score=0.9)
        ps2 = _make_passage_stance(passage_id="p2", rank=5, retrieval_score=0.9)
        sr = _make_stance_result([ps1, ps2])
        scored = CredibilityScorer().score(sr)

        assert scored[0].credibility > scored[1].credibility

    def test_fever_higher_than_politifact(self):
        ps_fever = _make_passage_stance(passage_id="p1", dataset="fever", rank=1)
        ps_poli = _make_passage_stance(passage_id="p2", dataset="politifact", rank=1)
        sr = _make_stance_result([ps_fever, ps_poli])
        scored = CredibilityScorer().score(sr)

        assert scored[0].credibility > scored[1].credibility

    def test_weighted_confidence(self):
        ps = _make_passage_stance(confidence=0.80, rank=1)
        sr = _make_stance_result([ps])
        scored = CredibilityScorer().score(sr)

        assert scored[0].weighted_confidence == ps.confidence * scored[0].credibility

    def test_empty_stance_result(self):
        sr = StanceResult(
            claim_text="Empty claim.",
            passage_stances=[],
            aggregate_label=StanceLabel.NEUTRAL,
            aggregate_score=0.0,
            supporting_count=0,
            refuting_count=0,
            neutral_count=0,
            latency_ms=0.0,
            model_name="test",
        )
        scored = CredibilityScorer().score(sr)
        assert scored == []

    def test_score_batch(self):
        ps = [_make_passage_stance()]
        sr = _make_stance_result(ps)
        batch = CredibilityScorer().score_batch([sr, sr, sr])
        assert len(batch) == 3
        assert all(len(b) == 1 for b in batch)
