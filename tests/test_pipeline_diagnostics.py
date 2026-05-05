from src.scripts.evaluate_pipeline import classify_failure_category


def test_failure_category_gold_page_absent_from_current_index():
    category = classify_failure_category(
        gold_label="SUPPORTED",
        predicted_label="NOT_ENOUGH_INFO",
        gold_pages={"Soul_Food_-LRB-film-RRB-"},
        gold_page_present_in_chroma=False,
        retrieval_rows=[],
        candidate_rows=[],
        stance_rows=[],
    )

    assert category == "gold_page_absent_from_current_index"


def test_failure_category_rerank_dropped_gold_page():
    category = classify_failure_category(
        gold_label="SUPPORTED",
        predicted_label="NOT_ENOUGH_INFO",
        gold_pages={"Telemundo"},
        gold_page_present_in_chroma=True,
        retrieval_rows=[{"source": "Other"}],
        candidate_rows=[{"source": "Telemundo"}],
        stance_rows=[],
    )

    assert category == "rerank_dropped_gold_page"


def test_failure_category_synthesis_wrong_when_stance_was_correct():
    category = classify_failure_category(
        gold_label="REFUTED",
        predicted_label="NOT_ENOUGH_INFO",
        gold_pages={"Telemundo"},
        gold_page_present_in_chroma=True,
        retrieval_rows=[{"source": "Telemundo"}],
        candidate_rows=[],
        stance_rows=[{"source": "Telemundo", "stance": "REFUTING"}],
    )

    assert category == "synthesis_wrong"
