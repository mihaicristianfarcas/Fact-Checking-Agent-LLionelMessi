from src.data_ingestion.datasets.base import EvidencePassage
from src.data_ingestion.retriever.evidence_retriever import RetrievalResult
from src.data_ingestion.retriever.hybrid_retriever import HybridEvidenceRetriever


def _result(pid, score, rank, source="Page", method=None, metadata=None):
    metadata = dict(metadata or {})
    if method:
        metadata["retrieval_method"] = method
    return RetrievalResult(
        passage=EvidencePassage(
            id=pid,
            text=f"Evidence text for {pid}",
            source=source,
            dataset="fever",
            metadata=metadata,
        ),
        score=score,
        rank=rank,
    )


class FakeDenseRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=None, **kwargs):
        return self.results[:top_k]


class FakeTitleRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=50):
        return self.results[:top_k]


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs):
        return self.scores[: len(pairs)]


def test_hybrid_retriever_dedupes_and_preserves_methods():
    dense = FakeDenseRetriever([
        _result("p1", 0.40, 1, method="dense"),
        _result("p2", 0.30, 2, method="dense"),
    ])
    title = FakeTitleRetriever([
        _result("p1", 0.80, 1, method="title"),
        _result("p3", 0.70, 2, method="title"),
    ])
    retriever = HybridEvidenceRetriever(
        dense_retriever=dense,
        title_retriever=title,
        enable_title_retrieval=True,
        enable_reranker=False,
        candidate_k=5,
    )

    results = retriever.retrieve("Claim.", top_k=5)
    ids = [r.passage.id for r in results]

    assert ids.count("p1") == 1
    assert results[0].passage.id == "p1"
    assert set(results[0].passage.metadata["retrieval_methods"]) == {"dense", "title"}


def test_hybrid_reranker_preserves_ids_and_scores():
    dense = FakeDenseRetriever([
        _result("p1", 0.90, 1, method="dense"),
        _result("p2", 0.80, 2, method="dense"),
    ])
    retriever = HybridEvidenceRetriever(
        dense_retriever=dense,
        enable_title_retrieval=False,
        enable_reranker=True,
        candidate_k=2,
        reranker=FakeReranker([-2.0, 2.0]),
    )

    results = retriever.retrieve("Claim.", top_k=2)

    assert [r.passage.id for r in results] == ["p2", "p1"]
    assert results[0].passage.metadata["rerank_score_raw"] == 2.0
    assert 0.0 <= results[0].score <= 1.0


def test_hybrid_retriever_scores_source_relevance_for_title_quality():
    title = FakeTitleRetriever([
        _result(
            "exact",
            0.90,
            1,
            source="Shane_Black",
            method="title",
            metadata={"title_score": 1.0, "title_match_type": "exact"},
        ),
        _result(
            "fuzzy",
            0.80,
            2,
            source="Shane_Blackett",
            method="title",
            metadata={"title_score": 0.35, "title_match_type": "token_fts"},
        ),
    ])
    retriever = HybridEvidenceRetriever(
        dense_retriever=FakeDenseRetriever([]),
        title_retriever=title,
        enable_dense_retrieval=False,
        enable_title_retrieval=True,
        enable_reranker=False,
        candidate_k=5,
    )

    results = retriever.retrieve("Shane Black was born in 1961.", top_k=2)
    relevance = {
        r.passage.id: r.passage.metadata["source_relevance"] for r in results
    }

    assert relevance["exact"] > relevance["fuzzy"]
    assert relevance["fuzzy"] < 0.55


def test_retrieve_batch_applies_dataset_filter_to_title_results():
    """Batch path must apply dataset_filter to title results, matching the
    single-call retrieve() path. Otherwise non-FEVER passages leak in."""
    politifact_passage = RetrievalResult(
        passage=EvidencePassage(
            id="politifact-1",
            text="Politifact passage.",
            source="politifact",
            dataset="politifact",
        ),
        score=0.9,
        rank=1,
    )
    fever_passage = _result("fever-1", 0.8, 1, method="title")

    title = FakeTitleRetriever([politifact_passage, fever_passage])
    retriever = HybridEvidenceRetriever(
        dense_retriever=FakeDenseRetriever([]),
        title_retriever=title,
        enable_dense_retrieval=False,
        enable_title_retrieval=True,
        enable_reranker=False,
        candidate_k=5,
    )

    batch = retriever.retrieve_batch(["claim 1"], top_k=5, dataset_filter="fever")
    ids = [r.passage.id for r in batch[0]]

    assert "politifact-1" not in ids
    assert "fever-1" in ids
