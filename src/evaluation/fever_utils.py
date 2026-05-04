"""Shared FEVER evaluation helpers.

These utilities keep the baseline and full-pipeline evaluation scripts aligned
on label normalization, dev-set de-duplication, exact train-overlap exclusion,
and page-level retrieval diagnostics.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from loguru import logger

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
}
LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def normalize_claim_text(text: str) -> str:
    """Normalize claim text for exact leakage/decontamination checks."""
    return re.sub(r"\s+", " ", text.strip().lower())


def load_train_claim_texts(path: str | Path) -> set[str]:
    """Load normalized training claim texts from a JSONL triples file."""
    train_path = Path(path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training triples file not found: {train_path}")

    claims: set[str] = set()
    for line in train_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        claim_text = row.get("claim_text")
        if claim_text:
            claims.add(normalize_claim_text(claim_text))
    return claims


def load_fever_dev_claims(
    max_claims: int | None,
    exclude_claim_texts: set[str] | None = None,
) -> tuple[list[dict], int]:
    """Load and de-duplicate FEVER labelled_dev claims.

    Returns one row per unique FEVER claim id:
        {id, claim, label, gold_pages}

    ``gold_pages`` is a set of cited Wikipedia page titles and is empty for
    NOT ENOUGH INFO claims.
    """
    logger.info("Loading FEVER labelled_dev...")
    ds = load_dataset(
        "fever/fever",
        "v1.0",
        split="labelled_dev",
        trust_remote_code=True,
        verification_mode="no_checks",
    )

    grouped: dict[int, dict] = {}
    for row in ds:
        cid = row["id"]
        if cid not in grouped:
            grouped[cid] = {
                "id": cid,
                "claim": row["claim"],
                "label": LABEL_MAP[row["label"]],
                "gold_pages": set(),
            }
        if row["evidence_wiki_url"]:
            grouped[cid]["gold_pages"].add(row["evidence_wiki_url"])

    claims = list(grouped.values())
    excluded_count = 0
    if exclude_claim_texts:
        before = len(claims)
        claims = [
            claim
            for claim in claims
            if normalize_claim_text(claim["claim"]) not in exclude_claim_texts
        ]
        excluded_count = before - len(claims)
        logger.info(
            f"Excluded {excluded_count} dev claims that exactly overlap training triples"
        )

    if max_claims and max_claims > 0:
        claims = claims[:max_claims]

    logger.info(f"Evaluating on {len(claims)} claims")
    return claims, excluded_count


def recall_at_k(results: list, gold_pages: set[str], k: int) -> bool:
    """True if any of the top-k passages comes from a gold Wikipedia page."""
    if not gold_pages:
        return False
    return any(r.passage.source in gold_pages for r in results[:k])


def page_sources_from_results(results: Iterable) -> set[str]:
    """Collect source/page titles from retrieval results."""
    return {r.passage.source for r in results if getattr(r, "passage", None)}


class ChromaPagePresence:
    """Fast page-title presence checks against Chroma's SQLite metadata."""

    def __init__(self, chroma_sqlite_path: str | Path) -> None:
        self.path = Path(chroma_sqlite_path)
        self._conn: sqlite3.Connection | None = None
        self._cache: dict[str, bool] = {}

    @classmethod
    def from_index_dir(cls, index_dir: str | Path) -> "ChromaPagePresence":
        return cls(Path(index_dir) / "chroma.sqlite3")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ChromaPagePresence":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _connect(self) -> sqlite3.Connection | None:
        if not self.path.exists():
            return None
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path))
        return self._conn

    def has_page(self, page: str) -> bool:
        """Return True when Chroma metadata contains a source page title."""
        if page in self._cache:
            return self._cache[page]

        conn = self._connect()
        if conn is None:
            self._cache[page] = False
            return False

        row = conn.execute(
            """
            SELECT 1
            FROM embedding_metadata
            WHERE key = 'source' AND string_value = ?
            LIMIT 1
            """,
            (page,),
        ).fetchone()
        present = row is not None
        self._cache[page] = present
        return present

    def any_present(self, pages: Iterable[str]) -> bool:
        return any(self.has_page(page) for page in pages)


def gold_page_presence_stats(
    claims: Iterable[dict],
    page_presence: ChromaPagePresence,
) -> dict:
    """Compute gold-page presence statistics for claims with evidence."""
    with_gold = [claim for claim in claims if claim.get("gold_pages")]
    present_count = sum(
        1 for claim in with_gold if page_presence.any_present(claim["gold_pages"])
    )
    return {
        "claims_with_gold_pages": len(with_gold),
        "claims_with_gold_page_present": present_count,
        "gold_page_present_rate": (
            present_count / len(with_gold) if with_gold else None
        ),
    }


def serialize_gold_pages(gold_pages: Iterable[str]) -> list[str]:
    """Return deterministic JSON-friendly gold page lists."""
    return sorted(str(page) for page in gold_pages)
