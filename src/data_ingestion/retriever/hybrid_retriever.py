"""Hybrid evidence retrieval with optional FEVER title lookup and reranking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from loguru import logger
from tqdm import tqdm

from src.data_ingestion.datasets.base import EvidencePassage
from src.data_ingestion.retriever.evidence_retriever import (
    EvidenceRetriever,
    RetrievalResult,
)
from src.data_ingestion.retriever.fever_title_retriever import (
    FeverTitleRetriever,
    extract_title_spans,
    normalize_title,
)


class _RetrieverProtocol(Protocol):
    def retrieve(self, query: str, top_k: int | None = None, **kwargs) -> list:
        ...


@dataclass
class RetrievalDebug:
    """Candidate/final retrieval metadata for diagnostics."""

    query: str
    candidates: list[dict] = field(default_factory=list)
    final_results: list[dict] = field(default_factory=list)
    reranker_enabled: bool = False


class HybridEvidenceRetriever:
    """Union dense Chroma retrieval, FEVER title retrieval, and rerank.

    The class preserves the ``EvidenceRetriever`` public shape: callers use
    ``retrieve(query, top_k=...)`` and receive ``RetrievalResult`` objects whose
    passage IDs remain the only citation IDs downstream components can cite.
    """

    def __init__(
        self,
        *,
        dense_retriever: _RetrieverProtocol | None = None,
        enable_dense_retrieval: bool = True,
        enable_title_retrieval: bool = False,
        enable_reranker: bool = False,
        candidate_k: int = 50,
        title_candidate_k: int = 50,
        title_candidate_pages: int = 20,
        title_passages_per_page: int = 3,
        title_index_path: str | Path | None = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        title_retriever: FeverTitleRetriever | None = None,
        reranker=None,
    ) -> None:
        self.enable_dense_retrieval = enable_dense_retrieval
        self.enable_title_retrieval = enable_title_retrieval
        self.enable_reranker = enable_reranker
        self.candidate_k = candidate_k
        self.title_candidate_k = title_candidate_k
        self.title_candidate_pages = title_candidate_pages
        self.title_passages_per_page = title_passages_per_page
        self.reranker_model = reranker_model

        self.dense_retriever = dense_retriever
        self.title_retriever = title_retriever
        self.title_index_path = title_index_path
        self._reranker = reranker
        self._last_debug_by_query: dict[str, RetrievalDebug] = {}

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        dataset_filter: str | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve final evidence candidates for a claim."""
        final_top_k = top_k or 5
        dense_top_k = max(final_top_k, self.candidate_k)
        title_top_k = max(final_top_k, self.title_candidate_k)

        candidates: list[RetrievalResult] = []

        if self.enable_dense_retrieval:
            dense_retriever = self._dense_retriever()
            dense_results = dense_retriever.retrieve(
                query,
                top_k=dense_top_k,
                dataset_filter=dataset_filter,
                source_filter=source_filter,
            )
            candidates.extend(_tag_results(dense_results, method="dense"))

        if self.enable_title_retrieval and not source_filter:
            title_results = self._title_retriever().retrieve(query, top_k=title_top_k)
            if dataset_filter:
                title_results = [
                    r for r in title_results if r.passage.dataset == dataset_filter
                ]
            candidates.extend(_tag_results(title_results, method="title"))

        merged = annotate_source_relevance(query, merge_retrieval_results(candidates))
        debug = RetrievalDebug(
            query=query,
            candidates=[_summarize_result(r) for r in merged],
            reranker_enabled=self.enable_reranker,
        )

        if self.enable_reranker and merged:
            ranked = annotate_source_relevance(query, self._rerank(query, merged))
        else:
            ranked = sorted(merged, key=lambda r: r.score, reverse=True)

        limited = ranked[:final_top_k]
        final = [
            RetrievalResult(passage=r.passage, score=r.score, rank=i + 1)
            for i, r in enumerate(limited)
        ]
        debug.final_results = [_summarize_result(r) for r in final]
        self._last_debug_by_query[query] = debug
        return final

    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int | None = None,
    ) -> list[list[RetrievalResult]]:
        """Retrieve for multiple claims.

        Dense-only mode delegates to the underlying batch API. Hybrid/reranked
        modes run sequentially so candidate traces stay simple and explicit.
        """
        if (
            self.enable_dense_retrieval
            and not self.enable_title_retrieval
            and not self.enable_reranker
            and self.dense_retriever is not None
            and hasattr(self.dense_retriever, "retrieve_batch")
        ):
            return self.dense_retriever.retrieve_batch(queries, top_k=top_k)

        final_top_k = top_k or 5
        dense_top_k = max(final_top_k, self.candidate_k)
        title_top_k = max(final_top_k, self.title_candidate_k)

        dense_batches: list[list[RetrievalResult]] = [[] for _ in queries]
        if self.enable_dense_retrieval:
            dense_retriever = self._dense_retriever()
            if hasattr(dense_retriever, "retrieve_batch"):
                dense_batches = dense_retriever.retrieve_batch(
                    queries,
                    top_k=dense_top_k,
                )
            else:
                dense_batches = [
                    dense_retriever.retrieve(query, top_k=dense_top_k)
                    for query in tqdm(queries, desc="Dense retrieval")
                ]

        all_results: list[list[RetrievalResult]] = []
        iterator = tqdm(
            list(enumerate(queries)),
            desc="Hybrid retrieval",
            disable=len(queries) < 10,
        )
        for idx, query in iterator:
            candidates: list[RetrievalResult] = []
            if self.enable_dense_retrieval:
                candidates.extend(_tag_results(dense_batches[idx], method="dense"))
            if self.enable_title_retrieval:
                title_results = self._title_retriever().retrieve(
                    query,
                    top_k=title_top_k,
                )
                candidates.extend(_tag_results(title_results, method="title"))

            merged = annotate_source_relevance(
                query,
                merge_retrieval_results(candidates),
            )
            debug = RetrievalDebug(
                query=query,
                candidates=[_summarize_result(r) for r in merged],
                reranker_enabled=self.enable_reranker,
            )

            if self.enable_reranker and merged:
                ranked = annotate_source_relevance(query, self._rerank(query, merged))
            else:
                ranked = sorted(merged, key=lambda r: r.score, reverse=True)

            limited = ranked[:final_top_k]
            final = [
                RetrievalResult(passage=r.passage, score=r.score, rank=i + 1)
                for i, r in enumerate(limited)
            ]
            debug.final_results = [_summarize_result(r) for r in final]
            self._last_debug_by_query[query] = debug
            all_results.append(final)

        return all_results

    def get_last_debug(self, query: str) -> RetrievalDebug | None:
        """Return the latest retrieval debug trace for a query, if available."""
        return self._last_debug_by_query.get(query)

    def get_corpus_stats(self) -> dict:
        """Proxy corpus stats for scripts that expect EvidenceRetriever."""
        return self._dense_retriever().get_corpus_stats()

    def _dense_retriever(self):
        if self.dense_retriever is None:
            self.dense_retriever = EvidenceRetriever()
        return self.dense_retriever

    def _title_retriever(self) -> FeverTitleRetriever:
        if self.title_retriever is None:
            self.title_retriever = FeverTitleRetriever(
                self.title_index_path,
                candidate_pages=self.title_candidate_pages,
                passages_per_page=self.title_passages_per_page,
            )
        return self.title_retriever

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker {self.reranker_model}")
            self._reranker = CrossEncoder(self.reranker_model)
        return self._reranker

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        reranker = self._load_reranker()
        pairs = [(query, candidate.passage.text) for candidate in candidates]
        raw_scores = reranker.predict(pairs)

        reranked: list[RetrievalResult] = []
        for candidate, raw_score in zip(candidates, raw_scores):
            raw = float(raw_score)
            normalized = _sigmoid(raw)
            metadata = dict(candidate.passage.metadata)
            metadata["rerank_score_raw"] = raw
            metadata["rerank_score"] = normalized
            passage = EvidencePassage(
                id=candidate.passage.id,
                text=candidate.passage.text,
                source=candidate.passage.source,
                dataset=candidate.passage.dataset,
                metadata=metadata,
            )
            reranked.append(
                RetrievalResult(passage=passage, score=normalized, rank=candidate.rank)
            )

        reranked.sort(
            key=lambda r: (
                r.passage.metadata.get("rerank_score_raw", r.score),
                r.score,
            ),
            reverse=True,
        )
        return [
            RetrievalResult(passage=r.passage, score=r.score, rank=i + 1)
            for i, r in enumerate(reranked)
        ]


def merge_retrieval_results(results: Iterable[RetrievalResult]) -> list[RetrievalResult]:
    """De-duplicate candidate results by passage ID while merging metadata."""
    merged: dict[str, RetrievalResult] = {}
    for result in results:
        existing = merged.get(result.passage.id)
        if existing is None:
            merged[result.passage.id] = result
            continue

        existing_metadata = dict(existing.passage.metadata)
        result_metadata = dict(result.passage.metadata)
        methods = set(_as_list(existing_metadata.get("retrieval_methods")))
        methods.update(_as_list(existing_metadata.get("retrieval_method")))
        methods.update(_as_list(result_metadata.get("retrieval_methods")))
        methods.update(_as_list(result_metadata.get("retrieval_method")))

        combined_metadata = existing_metadata
        combined_metadata.update(result_metadata)
        method = result.passage.metadata.get("retrieval_method")
        if method:
            methods.add(str(method))
        if methods:
            combined_metadata["retrieval_methods"] = sorted(methods)

        best = result if result.score > existing.score else existing
        passage = EvidencePassage(
            id=best.passage.id,
            text=best.passage.text,
            source=best.passage.source,
            dataset=best.passage.dataset,
            metadata=combined_metadata,
        )
        merged[result.passage.id] = RetrievalResult(
            passage=passage,
            score=max(existing.score, result.score),
            rank=min(existing.rank, result.rank),
        )

    ranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)
    return [
        RetrievalResult(passage=r.passage, score=r.score, rank=i + 1)
        for i, r in enumerate(ranked)
    ]


def annotate_source_relevance(
    query: str,
    results: Iterable[RetrievalResult],
) -> list[RetrievalResult]:
    """Attach a claim/title relevance score used by stance synthesis."""
    annotated: list[RetrievalResult] = []
    for result in results:
        metadata = dict(result.passage.metadata)
        relevance, reason = _source_relevance(query, result, metadata)
        metadata["source_relevance"] = round(relevance, 4)
        metadata["source_relevance_reason"] = reason
        passage = EvidencePassage(
            id=result.passage.id,
            text=result.passage.text,
            source=result.passage.source,
            dataset=result.passage.dataset,
            metadata=metadata,
        )
        annotated.append(
            RetrievalResult(passage=passage, score=result.score, rank=result.rank)
        )
    return annotated


def _tag_results(
    results: Iterable[RetrievalResult],
    *,
    method: str,
) -> list[RetrievalResult]:
    tagged: list[RetrievalResult] = []
    for result in results:
        metadata = dict(result.passage.metadata)
        metadata["retrieval_method"] = method
        score_key = f"{method}_score"
        if score_key in metadata:
            metadata[f"{method}_retrieval_score"] = result.score
        else:
            metadata[score_key] = result.score
        metadata[f"{method}_rank"] = result.rank
        methods = set(_as_list(metadata.get("retrieval_methods")))
        methods.add(method)
        metadata["retrieval_methods"] = sorted(methods)
        passage = EvidencePassage(
            id=result.passage.id,
            text=result.passage.text,
            source=result.passage.source,
            dataset=result.passage.dataset,
            metadata=metadata,
        )
        tagged.append(
            RetrievalResult(passage=passage, score=result.score, rank=result.rank)
        )
    return tagged


def _summarize_result(result: RetrievalResult) -> dict:
    metadata = result.passage.metadata
    return {
        "passage_id": result.passage.id,
        "source": result.passage.source,
        "dataset": result.passage.dataset,
        "rank": result.rank,
        "score": result.score,
        "retrieval_methods": _as_list(metadata.get("retrieval_methods"))
        or _as_list(metadata.get("retrieval_method")),
        "dense_score": metadata.get("dense_score"),
        "title_score": metadata.get("title_score"),
        "title_match_type": metadata.get("title_match_type"),
        "rerank_score": metadata.get("rerank_score"),
        "source_relevance": metadata.get("source_relevance"),
        "source_relevance_reason": metadata.get("source_relevance_reason"),
        "text_preview": result.passage.text[:160],
    }


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _source_relevance(
    query: str,
    result: RetrievalResult,
    metadata: dict,
) -> tuple[float, str]:
    source_norm = normalize_title(result.passage.source)
    if not source_norm:
        return 0.50, "missing_source"

    query_norm = normalize_title(query)
    claim_tokens = set(_content_tokens(query_norm))
    source_tokens = set(_content_tokens(source_norm))
    if not source_tokens:
        return 0.50, "empty_source_tokens"

    span_norms = [normalize_title(span) for span in extract_title_spans(query)]
    span_norms = [span for span in span_norms if span]
    span_token_sets = [set(_content_tokens(span)) for span in span_norms]
    span_token_sets = [tokens for tokens in span_token_sets if tokens]

    exact_span_match = source_norm in span_norms
    subset_span_match = any(
        len(tokens) >= 2 and tokens <= source_tokens for tokens in span_token_sets
    )
    overlap = len(claim_tokens & source_tokens)
    coverage = overlap / max(1, len(source_tokens))
    claim_coverage = overlap / max(1, len(claim_tokens))

    methods = set(_as_list(metadata.get("retrieval_methods")))
    methods.update(_as_list(metadata.get("retrieval_method")))
    title_score = _safe_float(metadata.get("title_score"))
    rerank_score = _safe_float(metadata.get("rerank_score"))
    match_type = str(metadata.get("title_match_type") or "")

    if exact_span_match or match_type == "exact":
        relevance = max(0.95, title_score)
        reason = "exact_title"
    elif subset_span_match:
        relevance = max(0.78, 0.65 + 0.20 * coverage + 0.10 * title_score)
        reason = "title_span_subset"
    elif match_type == "phrase_fts":
        relevance = max(0.68, 0.55 + 0.25 * title_score + 0.15 * claim_coverage)
        reason = "phrase_title"
    elif "title" in methods:
        relevance = max(0.20, 0.25 + 0.45 * title_score + 0.20 * claim_coverage)
        reason = "token_title"
    else:
        relevance = max(0.20, 0.35 + 0.45 * claim_coverage + 0.10 * coverage)
        reason = "dense_title_overlap"

    if "title" in methods and "dense" in methods:
        relevance += 0.05
    if rerank_score > 0:
        relevance += 0.04 * rerank_score

    extra_tokens = source_tokens - claim_tokens
    extra_penalty = min(0.20, 0.04 * len(extra_tokens))
    relevance -= extra_penalty

    low_signal = source_tokens & {
        "category",
        "discography",
        "disambiguation",
        "filmography",
        "index",
        "list",
        "lists",
    }
    if low_signal and not (claim_tokens & low_signal):
        relevance *= 0.55
        reason += "_low_signal_page"

    return max(0.05, min(1.0, relevance)), reason


def _content_tokens(text: str) -> list[str]:
    stopwords = {"a", "an", "and", "by", "for", "in", "of", "on", "the", "to"}
    return [tok for tok in text.split() if tok and tok not in stopwords]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
