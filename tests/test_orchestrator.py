"""
Tests for the Fact-Check Agent Orchestrator.

All external dependencies (Ollama, ChromaDB, NLI model) are mocked.

Run with:
    pytest tests/test_orchestrator.py -v
"""

from unittest.mock import MagicMock

from src.agent.orchestrator import FactCheckAgent, PipelineTrace
from src.claim_processing.decomposer import (
    AtomicClaim,
    ClaimDecomposer,
    DecompositionResult,
)
from src.claim_processing.stance_classifier import (
    PassageStance,
    StanceClassifier,
    StanceLabel,
    StanceResult,
)
from src.data_ingestion.datasets.base import EvidencePassage
from src.data_ingestion.retriever.evidence_retriever import (
    EvidenceRetriever,
    RetrievalResult,
)
from src.synthesis.verdict_synthesizer import SynthesisResult


def _mock_decomposer(texts: list[str]) -> ClaimDecomposer:
    decomposer = MagicMock(spec=ClaimDecomposer)
    decomposer.decompose.return_value = DecompositionResult(
        original_claim="Compound claim.",
        atomic_claims=[
            AtomicClaim(text=t, source_claim="Compound claim.", claim_index=i)
            for i, t in enumerate(texts)
        ],
        was_compound=len(texts) > 1,
        model_used="mock",
        latency_ms=5.0,
    )
    return decomposer


def _mock_retriever(passages_per_claim: int = 2) -> EvidenceRetriever:
    retriever = MagicMock(spec=EvidenceRetriever)

    def _retrieve(query, top_k=5, **kwargs):
        return [
            RetrievalResult(
                passage=EvidencePassage(
                    id=f"passage_{query[:10]}_{i}",
                    text=f"Evidence for {query[:30]}.",
                    source="MockPage",
                    dataset="fever",
                ),
                score=0.90 - i * 0.05,
                rank=i + 1,
            )
            for i in range(min(passages_per_claim, top_k))
        ]

    retriever.retrieve.side_effect = _retrieve
    return retriever


def _mock_stance_classifier() -> StanceClassifier:
    classifier = MagicMock(spec=StanceClassifier)

    def _classify(claim, retrievals):
        stances = [
            PassageStance(
                passage_id=r.passage.id,
                passage_text=r.passage.text,
                passage_source=r.passage.source,
                passage_dataset=r.passage.dataset,
                retrieval_score=r.score,
                retrieval_rank=r.rank,
                stance=StanceLabel.SUPPORTING,
                confidence=0.88,
                raw_scores={"SUPPORTING": 0.88, "REFUTING": 0.06, "NEUTRAL": 0.06},
            )
            for r in retrievals
        ]
        return StanceResult(
            claim_text=claim,
            passage_stances=stances,
            aggregate_label=StanceLabel.SUPPORTING,
            aggregate_score=0.88,
            supporting_count=len(stances),
            refuting_count=0,
            neutral_count=0,
            latency_ms=10.0,
            model_name="mock-nli",
        )

    classifier.classify.side_effect = _classify
    return classifier


class TestFactCheckAgent:

    def _agent(self, atomic_texts=None):
        if atomic_texts is None:
            atomic_texts = ["Atomic claim A."]
        return FactCheckAgent(
            decomposer=_mock_decomposer(atomic_texts),
            retriever=_mock_retriever(),
            stance_classifier=_mock_stance_classifier(),
        )

    def test_check_returns_synthesis_result(self):
        agent = self._agent()
        result = agent.check("Some claim.")
        assert isinstance(result, SynthesisResult)
        assert result.verdict in ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")

    def test_check_with_trace_returns_trace(self):
        agent = self._agent()
        trace = agent.check_with_trace("Some claim.")
        assert isinstance(trace, PipelineTrace)
        assert trace.synthesis is not None
        assert trace.decomposition is not None
        assert len(trace.steps_executed) == 4

    def test_steps_are_in_order(self):
        agent = self._agent()
        trace = agent.check_with_trace("Claim.")
        assert trace.steps_executed == [
            "decompose", "retrieve", "stance_classify", "synthesize"
        ]

    def test_compound_claim_decomposes(self):
        agent = self._agent(["Sub A.", "Sub B."])
        trace = agent.check_with_trace("Sub A and Sub B.")
        assert len(trace.retrievals) == 2
        assert len(trace.stance_results) == 2

    def test_all_citations_from_retrieval(self):
        agent = self._agent()
        trace = agent.check_with_trace("Claim.")
        result = trace.synthesis
        all_retrieved = result.all_retrieved_ids
        for cited in result.cited_passage_ids:
            assert cited in all_retrieved, f"Citation {cited} not in retrieved set"

    def test_trace_to_dict_serializable(self):
        import json
        agent = self._agent()
        trace = agent.check_with_trace("Claim.")
        d = trace.to_dict()
        serialized = json.dumps(d, default=str)
        assert '"original_claim"' in serialized

    def test_check_batch(self):
        agent = self._agent()
        results = agent.check_batch(["Claim 1.", "Claim 2."])
        assert len(results) == 2
        assert all(isinstance(r, SynthesisResult) for r in results)

    def test_latency_is_positive(self):
        agent = self._agent()
        trace = agent.check_with_trace("Claim.")
        assert trace.total_latency_ms > 0

    def test_no_hallucinated_citations(self):
        agent = self._agent()
        result = agent.check("Claim.")
        assert result.hallucinated_citations == []
