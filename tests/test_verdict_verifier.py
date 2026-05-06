from src.claim_processing.verdict_verifier import (
    VerifierPrediction,
    apply_baseline_refute_fallback,
    build_verifier_input,
    calibrate_verifier_prediction,
    override_synthesis_with_prediction,
)
from src.data_ingestion.datasets.base import EvidencePassage
from src.data_ingestion.retriever.evidence_retriever import RetrievalResult
from src.synthesis.verdict_synthesizer import AtomicVerdict, SynthesisResult


def _retrieval(pid: str, rank: int, source: str = "Lionel_Messi") -> RetrievalResult:
    return RetrievalResult(
        passage=EvidencePassage(
            id=pid,
            text=f"Evidence text {rank}.",
            source=source,
            dataset="fever",
        ),
        score=0.9,
        rank=rank,
    )


def test_build_verifier_input_uses_shared_claim_evidence_format():
    text = build_verifier_input(
        "Lionel Messi plays football.",
        [_retrieval("p1", 1)],
    )

    assert text.startswith("Claim: Lionel Messi plays football.")
    assert "Evidence:" in text
    assert "[1] Source: Lionel Messi" in text
    assert "Evidence text 1." in text


def test_override_synthesis_keeps_citations_retrieved_only():
    retrievals = [_retrieval("p1", 1), _retrieval("p2", 2)]
    original = SynthesisResult(
        original_claim="Claim.",
        verdict="NOT_ENOUGH_INFO",
        confidence=0.0,
        explanation="Old result.",
        cited_passage_ids=["not_retrieved"],
        atomic_verdicts=[
            AtomicVerdict(
                claim_text="Claim.",
                verdict="NOT_ENOUGH_INFO",
                confidence=0.0,
                cited_passages=[],
            )
        ],
        all_retrieved_ids=["p1", "p2"],
    )
    prediction = VerifierPrediction(
        label="SUPPORTED",
        confidence=0.8,
        probabilities={
            "SUPPORTED": 0.8,
            "REFUTED": 0.1,
            "NOT_ENOUGH_INFO": 0.1,
        },
    )

    result = override_synthesis_with_prediction(original, prediction, retrievals)

    assert result.verdict == "SUPPORTED"
    assert result.cited_passage_ids == ["p1", "p2"]
    assert result.hallucinated_citations == []


def test_calibrate_verifier_prediction_downgrades_weak_decisive_label():
    prediction = VerifierPrediction(
        label="SUPPORTED",
        confidence=0.55,
        probabilities={
            "SUPPORTED": 0.55,
            "REFUTED": 0.15,
            "NOT_ENOUGH_INFO": 0.30,
        },
    )

    calibrated = calibrate_verifier_prediction(
        prediction,
        supported_threshold=0.60,
    )

    assert calibrated.label == "NOT_ENOUGH_INFO"
    assert calibrated.confidence == 0.30


def test_baseline_refute_fallback_recovers_strong_refute_after_abstention():
    prediction = VerifierPrediction(
        label="NOT_ENOUGH_INFO",
        confidence=0.45,
        probabilities={
            "SUPPORTED": 0.20,
            "REFUTED": 0.35,
            "NOT_ENOUGH_INFO": 0.45,
        },
    )
    baseline = SynthesisResult(
        original_claim="Claim.",
        verdict="REFUTED",
        confidence=0.82,
        explanation="Baseline refutes.",
        cited_passage_ids=["p1"],
        atomic_verdicts=[],
        all_retrieved_ids=["p1"],
    )

    recovered = apply_baseline_refute_fallback(
        prediction,
        baseline,
        min_baseline_confidence=0.75,
        min_verifier_refute_probability=0.25,
    )

    assert recovered.label == "REFUTED"
    assert recovered.confidence == 0.82


def test_baseline_refute_fallback_keeps_probabilities_normalized():
    """Probabilities dict must remain a valid distribution (sum ~= 1.0)."""
    prediction = VerifierPrediction(
        label="NOT_ENOUGH_INFO",
        confidence=0.45,
        probabilities={
            "SUPPORTED": 0.20,
            "REFUTED": 0.35,
            "NOT_ENOUGH_INFO": 0.45,
        },
    )
    baseline = SynthesisResult(
        original_claim="Claim.",
        verdict="REFUTED",
        confidence=0.82,
        explanation="Baseline refutes.",
        cited_passage_ids=["p1"],
        atomic_verdicts=[],
        all_retrieved_ids=["p1"],
    )

    recovered = apply_baseline_refute_fallback(
        prediction,
        baseline,
        min_baseline_confidence=0.75,
        min_verifier_refute_probability=0.25,
    )

    total = sum(recovered.probabilities.values())
    assert abs(total - 1.0) < 1e-6, f"probabilities sum to {total}, not 1.0"
    assert recovered.probabilities["REFUTED"] == 0.82
    # Relative ordering of other classes should be preserved.
    assert (
        recovered.probabilities["NOT_ENOUGH_INFO"]
        > recovered.probabilities["SUPPORTED"]
    )
    assert all(0.0 <= p <= 1.0 for p in recovered.probabilities.values())


def test_baseline_refute_fallback_handles_degenerate_remainder():
    """When other classes sum to 0, fallback should put all mass on REFUTED."""
    prediction = VerifierPrediction(
        label="NOT_ENOUGH_INFO",
        confidence=1.0,
        probabilities={
            "SUPPORTED": 0.0,
            "REFUTED": 0.30,
            "NOT_ENOUGH_INFO": 0.0,
        },
    )
    baseline = SynthesisResult(
        original_claim="Claim.",
        verdict="REFUTED",
        confidence=0.90,
        explanation="Baseline refutes.",
        cited_passage_ids=["p1"],
        atomic_verdicts=[],
        all_retrieved_ids=["p1"],
    )

    recovered = apply_baseline_refute_fallback(
        prediction,
        baseline,
        min_baseline_confidence=0.75,
        min_verifier_refute_probability=0.25,
    )

    total = sum(recovered.probabilities.values())
    assert abs(total - 1.0) < 1e-6
    assert recovered.probabilities["REFUTED"] == 1.0
