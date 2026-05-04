#!/usr/bin/env python3
"""Build a full FEVER Wikipedia title lookup index.

This index stores only wiki page titles and row offsets from the public FEVER
wiki-pages split. It does not read labelled_dev labels or evidence annotations.

Usage:
    python -m src.scripts.build_fever_title_index
    python -m src.scripts.build_fever_title_index --output data/index/fever_titles.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from src.data_ingestion.retriever.fever_title_retriever import (
    default_title_index_path,
    initialize_title_index,
    normalize_title,
    title_search_text,
)


def _configure_fast_sqlite(conn: sqlite3.Connection) -> None:
    """Use faster settings for a local rebuildable title index."""
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -200000")


def _clear_title_index(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS page_titles_fts")
    conn.execute("DROP TABLE IF EXISTS pages")
    initialize_title_index(conn)


def _flush_batch(
    conn: sqlite3.Connection,
    page_rows: list[tuple[int, int, str, str]],
    fts_rows: list[tuple[int, str, str]],
) -> None:
    if not page_rows:
        return

    conn.executemany(
        """
        INSERT OR REPLACE INTO pages(id, row_index, title, normalized_title)
        VALUES (?, ?, ?, ?)
        """,
        page_rows,
    )
    conn.executemany(
        """
        INSERT INTO page_titles_fts(page_id, title, normalized_title)
        VALUES (?, ?, ?)
        """,
        fts_rows,
    )
    conn.commit()


def build_title_index(
    output: str | Path,
    limit: int | None = None,
    batch_size: int = 50_000,
    resume: bool = False,
) -> int:
    """Build the SQLite title index and return the number of indexed pages."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FEVER wiki_pages title source...")
    wiki = load_dataset(
        "fever/fever",
        "wiki_pages",
        trust_remote_code=True,
        verification_mode="no_checks",
    )
    pages = wiki["wikipedia_pages"]

    conn = sqlite3.connect(str(output_path))
    _configure_fast_sqlite(conn)
    if resume:
        initialize_title_index(conn)
    else:
        _clear_title_index(conn)

    count = 0
    page_rows: list[tuple[int, int, str, str]] = []
    fts_rows: list[tuple[int, str, str]] = []
    try:
        for row_index, page in enumerate(tqdm(pages, desc="Indexing FEVER titles")):
            title = page.get("id", "")
            normalized = normalize_title(title)
            if not normalized:
                continue

            page_id = row_index + 1
            page_rows.append((page_id, row_index, title, normalized))
            fts_rows.append((page_id, title_search_text(title), normalized))
            count += 1

            if len(page_rows) >= batch_size:
                _flush_batch(conn, page_rows, fts_rows)
                page_rows = []
                fts_rows = []

            if limit and count >= limit:
                break
        _flush_batch(conn, page_rows, fts_rows)
        conn.commit()
    finally:
        conn.close()

    logger.info(f"Indexed {count} FEVER page titles into {output_path}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FEVER title SQLite index")
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_title_index_path()),
        help="SQLite output path. Default: data/index/fever_titles.sqlite",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional page-title limit for smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="SQLite insert batch size.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append/update an existing index instead of clearing it first.",
    )
    args = parser.parse_args()

    build_title_index(
        args.output,
        limit=args.limit,
        batch_size=args.batch_size,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
