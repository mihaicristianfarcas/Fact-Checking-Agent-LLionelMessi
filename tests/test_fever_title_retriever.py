import sqlite3

from src.data_ingestion.retriever.fever_title_retriever import (
    FeverTitleIndex,
    FeverTitleRetriever,
    extract_title_spans,
    initialize_title_index,
    normalize_title,
    upsert_title_page,
)


def _build_title_index(path):
    conn = sqlite3.connect(path)
    initialize_title_index(conn)
    upsert_title_page(conn, row_index=0, title="Soul_Food_-LRB-film-RRB-")
    upsert_title_page(conn, row_index=1, title="Telemundo")
    upsert_title_page(conn, row_index=2, title="Shane_Black")
    upsert_title_page(conn, row_index=3, title="Shane_Blackett")
    conn.commit()
    conn.close()


def test_normalize_title_handles_fever_tokens():
    assert normalize_title("Soul_Food_-LRB-film-RRB-") == "soul food film"


def test_extract_title_spans_finds_entity_runs():
    spans = extract_title_spans("Fox 2000 Pictures released the film Soul Food.")
    normalized = {normalize_title(span) for span in spans}

    assert "fox 2000 pictures" in normalized
    assert "soul food" in normalized


def test_title_index_exact_and_fts_lookup(tmp_path):
    path = tmp_path / "titles.sqlite"
    _build_title_index(path)

    index = FeverTitleIndex(path)
    exact = index.search("Telemundo", limit=3)
    fts = index.search("Soul Food", limit=3)

    assert exact[0].title == "Telemundo"
    assert any(match.title == "Soul_Food_-LRB-film-RRB-" for match in fts)


def test_title_index_ranks_exact_title_above_fuzzy_same_name(tmp_path):
    path = tmp_path / "titles.sqlite"
    _build_title_index(path)

    index = FeverTitleIndex(path)
    matches = index.search("Shane Black", limit=5)

    assert matches[0].title == "Shane_Black"
    fuzzy = next(match for match in matches if match.title == "Shane_Blackett")
    assert matches[0].score > fuzzy.score


def test_title_retriever_returns_page_passages_without_gold_labels(tmp_path):
    path = tmp_path / "titles.sqlite"
    _build_title_index(path)
    wiki_pages = [
        {
            "id": "Soul_Food_-LRB-film-RRB-",
            "lines": "0\tSoul Food is a 1997 American comedy-drama film.\n"
            "1\tFox 2000 Pictures released Soul Food in the United States.",
        },
        {
            "id": "Telemundo",
            "lines": "0\tTelemundo is an American Spanish-language television network.",
        },
    ]
    retriever = FeverTitleRetriever(
        path,
        candidate_pages=5,
        passages_per_page=2,
        wiki_pages=wiki_pages,
    )

    results = retriever.retrieve(
        "Fox 2000 Pictures released the film Soul Food.",
        top_k=5,
    )

    assert results
    assert any(r.passage.source == "Soul_Food_-LRB-film-RRB-" for r in results)
    assert all("gold" not in r.passage.metadata for r in results)
