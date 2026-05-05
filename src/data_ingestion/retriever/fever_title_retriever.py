"""FEVER Wikipedia title and page retrieval.

The dense Chroma index can miss the correct FEVER page entirely when the local
index is filtered. This module adds a lightweight title lookup over the full
FEVER wiki-page title list, then fetches candidate page sentences from the
HuggingFace wiki cache.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from loguru import logger

from src.config import settings
from src.data_ingestion.datasets.base import EvidencePassage
from src.data_ingestion.preprocessing import clean_text
from src.data_ingestion.retriever.evidence_retriever import RetrievalResult

FEVER_TITLE_INDEX_FILENAME = "fever_titles.sqlite"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CAPITALIZED_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Za-z0-9'’-]*$")
_LOWER_CONNECTORS = {"a", "an", "and", "de", "del", "of", "the", "to", "van", "von"}
_TITLE_STOPWORDS = {"a", "an", "and", "by", "for", "in", "of", "on", "the", "to"}
_LOW_SIGNAL_TITLE_TOKENS = {
    "category",
    "discography",
    "disambiguation",
    "filmography",
    "index",
    "list",
    "lists",
}


@dataclass(frozen=True)
class FeverPageMatch:
    """One title-index search result."""

    title: str
    row_index: int
    score: float
    match_type: str
    query: str


@dataclass(frozen=True)
class FeverSentence:
    """Parsed sentence from a FEVER wiki page."""

    sentence_id: int
    text: str


def default_title_index_path() -> Path:
    """Default persistent path for the FEVER title SQLite index."""
    return settings.get_absolute_path(settings.index_dir) / FEVER_TITLE_INDEX_FILENAME


def normalize_title(text: str) -> str:
    """Normalize FEVER page titles and claim spans for exact/FTS matching."""
    text = (
        text.replace("_", " ")
        .replace("-LRB-", " ")
        .replace("-RRB-", " ")
        .replace("-LSB-", " ")
        .replace("-RSB-", " ")
    )
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def title_search_text(title: str) -> str:
    """Return an FTS-friendly title string."""
    return normalize_title(title)


def extract_title_spans(claim: str, max_tokens: int = 6) -> list[str]:
    """Extract likely Wikipedia title spans from a claim.

    This intentionally stays dependency-free. FEVER claims are title-heavy, so
    capitalized token runs plus their sub-spans are a useful first pass.
    """
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", claim)
    spans: list[list[str]] = []
    current: list[str] = []

    for token in tokens:
        is_titleish = bool(_CAPITALIZED_TOKEN_RE.match(token))
        is_connector = token.lower() in _LOWER_CONNECTORS

        if is_titleish or (is_connector and current):
            current.append(token)
            continue

        if current:
            while current and current[-1].lower() in _LOWER_CONNECTORS:
                current.pop()
            if current:
                spans.append(current)
            current = []

    if current:
        while current and current[-1].lower() in _LOWER_CONNECTORS:
            current.pop()
        if current:
            spans.append(current)

    candidates: list[str] = []
    for span_tokens in spans:
        n = len(span_tokens)
        for start in range(n):
            for end in range(min(n, start + max_tokens), start, -1):
                sub = span_tokens[start:end]
                if not sub or all(tok.lower() in _LOWER_CONNECTORS for tok in sub):
                    continue
                candidates.append(" ".join(sub))

    # Add quoted/title-cased fragments split by common punctuation.
    for piece in re.split(r"[,.;:!?()\\[\\]\"]+", claim):
        piece = piece.strip()
        if piece and any(ch.isupper() for ch in piece):
            candidates.append(piece)

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        normalized = normalize_title(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(candidate)

    return ordered


def initialize_title_index(conn: sqlite3.Connection) -> None:
    """Create the title-index schema if it does not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY,
            row_index INTEGER NOT NULL UNIQUE,
            title TEXT NOT NULL UNIQUE,
            normalized_title TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pages_normalized_title "
        "ON pages(normalized_title)"
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS page_titles_fts
        USING fts5(title, normalized_title, page_id UNINDEXED)
        """
    )
    conn.commit()


def upsert_title_page(
    conn: sqlite3.Connection,
    *,
    row_index: int,
    title: str,
) -> None:
    """Insert or update one FEVER wiki page title in the SQLite index."""
    normalized = normalize_title(title)
    if not normalized:
        return

    conn.execute(
        """
        INSERT INTO pages(row_index, title, normalized_title)
        VALUES (?, ?, ?)
        ON CONFLICT(title) DO UPDATE SET
            row_index = excluded.row_index,
            normalized_title = excluded.normalized_title
        """,
        (row_index, title, normalized),
    )
    page_id = conn.execute(
        "SELECT id FROM pages WHERE title = ?",
        (title,),
    ).fetchone()[0]
    conn.execute("DELETE FROM page_titles_fts WHERE page_id = ?", (page_id,))
    conn.execute(
        """
        INSERT INTO page_titles_fts(page_id, title, normalized_title)
        VALUES (?, ?, ?)
        """,
        (page_id, title_search_text(title), normalized),
    )


class FeverTitleIndex:
    """SQLite-backed full-FEVER title lookup."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_title_index_path()
        self._conn: sqlite3.Connection | None = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "FeverTitleIndex":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _connect(self) -> sqlite3.Connection:
        if not self.path.exists():
            raise FileNotFoundError(
                f"FEVER title index not found: {self.path}. "
                "Build it with `python -m src.scripts.build_fever_title_index`."
            )
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def search(self, query: str, limit: int = 20) -> list[FeverPageMatch]:
        """Search by exact normalized title first, then FTS title matches."""
        normalized = normalize_title(query)
        if not normalized:
            return []

        conn = self._connect()
        matches: list[FeverPageMatch] = []
        seen_titles: set[str] = set()

        for row in conn.execute(
            """
            SELECT title, row_index
            FROM pages
            WHERE normalized_title = ?
            LIMIT ?
            """,
            (normalized, limit),
        ):
            matches.append(
                FeverPageMatch(
                    title=row["title"],
                    row_index=int(row["row_index"]),
                    score=1.0,
                    match_type="exact",
                    query=query,
                )
            )
            seen_titles.add(row["title"])

        remaining = max(0, limit - len(matches))
        if remaining == 0:
            return matches

        fts_queries = _fts_queries(normalized)
        if not fts_queries:
            return matches

        for match_type, fts_query in fts_queries:
            try:
                rows = conn.execute(
                    """
                    SELECT p.title, p.row_index, bm25(page_titles_fts) AS rank_score
                    FROM page_titles_fts
                    JOIN pages p ON p.id = page_titles_fts.page_id
                    WHERE page_titles_fts MATCH ?
                    ORDER BY rank_score
                    LIMIT ?
                    """,
                    (fts_query, remaining * 4),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                logger.debug(f"FTS title search failed for {query!r}: {exc}")
                continue

            for row in rows:
                title = row["title"]
                if title in seen_titles:
                    continue
                raw_rank = float(row["rank_score"])
                score = _title_match_score(
                    normalized,
                    normalize_title(title),
                    match_type=match_type,
                    raw_rank=raw_rank,
                )
                if score <= 0.05:
                    continue
                matches.append(
                    FeverPageMatch(
                        title=title,
                        row_index=int(row["row_index"]),
                        score=score,
                        match_type=match_type,
                        query=query,
                    )
                )
                seen_titles.add(title)
                if len(matches) >= limit:
                    return _rank_page_matches(matches)

        return _rank_page_matches(matches)


class FeverTitleRetriever:
    """Retrieve FEVER page sentences via title/entity matching."""

    def __init__(
        self,
        title_index_path: str | Path | None = None,
        *,
        candidate_pages: int = 20,
        passages_per_page: int = 3,
        wiki_pages: Sequence[dict] | None = None,
    ) -> None:
        self.title_index = FeverTitleIndex(title_index_path)
        self.candidate_pages = candidate_pages
        self.passages_per_page = passages_per_page
        self._wiki_pages = wiki_pages
        self._loaded_dataset = None

    def retrieve(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        """Return title-matched FEVER sentence/window passages for a claim."""
        page_matches = self._find_page_matches(query)
        if not page_matches:
            return []

        query_tokens = set(_tokens(normalize_title(query)))
        candidates: list[RetrievalResult] = []

        for match in page_matches:
            page = self._get_page(match.row_index)
            if not page:
                continue
            sentences = parse_fever_lines(page.get("lines", ""))
            ranked = rank_sentences(query_tokens, sentences)
            for sentence, sentence_score in ranked[: self.passages_per_page]:
                text = clean_text(sentence.text)
                if not text:
                    continue
                score = _combine_scores(match.score, sentence_score)
                passage = EvidencePassage(
                    id=f"fever_{match.title}_{sentence.sentence_id}",
                    text=text,
                    source=match.title,
                    dataset="fever",
                    metadata={
                        "sentence_id": sentence.sentence_id,
                        "retrieval_method": "title",
                        "title_query": match.query,
                        "title_match_type": match.match_type,
                        "title_score": match.score,
                        "title_sentence_score": sentence_score,
                    },
                )
                candidates.append(
                    RetrievalResult(passage=passage, score=score, rank=0)
                )

        candidates = _dedupe_results(candidates)
        candidates.sort(key=lambda r: r.score, reverse=True)
        limited = candidates[:top_k]
        return [
            RetrievalResult(passage=r.passage, score=r.score, rank=i + 1)
            for i, r in enumerate(limited)
        ]

    def _find_page_matches(self, query: str) -> list[FeverPageMatch]:
        searches = extract_title_spans(query)
        searches.append(query)

        matches: list[FeverPageMatch] = []
        seen: set[str] = set()
        per_query_limit = max(3, math.ceil(self.candidate_pages / max(1, len(searches))))
        for search in searches:
            for match in self.title_index.search(search, limit=per_query_limit):
                if match.title in seen:
                    continue
                seen.add(match.title)
                matches.append(match)
                if len(matches) >= self.candidate_pages:
                    return _rank_page_matches(matches)[: self.candidate_pages]
        return _rank_page_matches(matches)[: self.candidate_pages]

    def _get_page(self, row_index: int) -> dict | None:
        pages = self._load_wiki_pages()
        try:
            return pages[row_index]
        except (IndexError, TypeError):
            logger.debug(f"FEVER wiki row index {row_index} was not available")
            return None

    def _load_wiki_pages(self):
        if self._wiki_pages is not None:
            return self._wiki_pages
        if self._loaded_dataset is None:
            from datasets import load_dataset

            wiki = load_dataset(
                "fever/fever",
                "wiki_pages",
                trust_remote_code=True,
                verification_mode="no_checks",
            )
            self._loaded_dataset = wiki["wikipedia_pages"]
        return self._loaded_dataset


def parse_fever_lines(lines: str) -> list[FeverSentence]:
    """Parse FEVER wiki ``lines`` into clean sentence records."""
    sentences: list[FeverSentence] = []
    for raw_line in lines.splitlines():
        if "\t" not in raw_line:
            continue
        parts = raw_line.split("\t")
        try:
            sentence_id = int(parts[0])
        except ValueError:
            continue
        text = parts[1].strip() if len(parts) > 1 else ""
        if text:
            sentences.append(FeverSentence(sentence_id=sentence_id, text=text))
    return sentences


def rank_sentences(
    query_tokens: set[str],
    sentences: Sequence[FeverSentence],
) -> list[tuple[FeverSentence, float]]:
    """Rank page sentences by lexical overlap with the claim."""
    ranked: list[tuple[FeverSentence, float]] = []
    for sentence in sentences:
        sentence_tokens = set(_tokens(normalize_title(sentence.text)))
        if not sentence_tokens:
            score = 0.0
        else:
            overlap = len(query_tokens & sentence_tokens)
            score = overlap / max(1, len(query_tokens))
            if sentence.sentence_id == 0:
                score += 0.05
        ranked.append((sentence, min(1.0, score)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _tokens(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) > 1]


def _fts_queries(normalized_query: str) -> list[tuple[str, str]]:
    terms = _tokens(normalized_query)
    if not terms:
        return []
    if len(terms) == 1:
        return [("token_fts", f"{terms[0]}*")]

    phrase = " ".join(terms)
    token_query = " OR ".join(f"{term}*" for term in terms)
    return [("phrase_fts", f'"{phrase}"'), ("token_fts", token_query)]


def _rank_page_matches(matches: Iterable[FeverPageMatch]) -> list[FeverPageMatch]:
    return sorted(
        matches,
        key=lambda match: (
            match.score,
            {"exact": 3, "phrase_fts": 2, "token_fts": 1}.get(match.match_type, 0),
            -len(_content_tokens(normalize_title(match.title))),
        ),
        reverse=True,
    )


def _title_match_score(
    query_norm: str,
    title_norm: str,
    *,
    match_type: str,
    raw_rank: float,
) -> float:
    query_tokens = set(_content_tokens(query_norm))
    title_tokens = set(_content_tokens(title_norm))
    if not query_tokens or not title_tokens:
        return 0.0

    overlap = len(query_tokens & title_tokens)
    coverage = overlap / max(1, len(query_tokens))
    precision = overlap / max(1, len(title_tokens))
    extra_tokens = len(title_tokens - query_tokens)
    extra_penalty = min(0.25, 0.05 * extra_tokens)

    if match_type == "exact":
        base = 1.0
    elif match_type == "phrase_fts":
        base = 0.86 * coverage + 0.08 * precision
    else:
        bm25_signal = 1.0 / (1.0 + max(0.0, raw_rank + 10.0))
        base = 0.55 * coverage + 0.25 * precision + 0.10 * bm25_signal

    low_signal_penalty = 0.0
    low_signal_terms = title_tokens & _LOW_SIGNAL_TITLE_TOKENS
    if low_signal_terms and not (query_tokens & low_signal_terms):
        low_signal_penalty = 0.25

    return max(0.0, min(1.0, base - extra_penalty - low_signal_penalty))


def _content_tokens(text: str) -> list[str]:
    return [tok for tok in _tokens(text) if tok not in _TITLE_STOPWORDS]


def _combine_scores(title_score: float, sentence_score: float) -> float:
    return max(0.0, min(1.0, 0.65 * title_score + 0.35 * sentence_score))


def _dedupe_results(results: Iterable[RetrievalResult]) -> list[RetrievalResult]:
    best: dict[str, RetrievalResult] = {}
    for result in results:
        existing = best.get(result.passage.id)
        if existing is None or result.score > existing.score:
            best[result.passage.id] = result
    return list(best.values())
