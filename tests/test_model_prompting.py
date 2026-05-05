"""Tests for trained-model prompt parsing helpers."""

from src.model_training.inference import (
    DEFAULT_ADAPTER_ID,
    DEFAULT_BASE_MODEL_ID,
    _guarded_text,
)
from src.model_training.prompting import (
    build_user_prompt,
    extract_citations,
    parse_model_output,
)


def test_default_model_points_to_published_huggingface_adapter():
    assert DEFAULT_ADAPTER_ID == "andreiungureanu/Fact-Checking-Agent-LLionelMessi"
    assert DEFAULT_BASE_MODEL_ID == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_build_user_prompt_preserves_evidence_ids():
    prompt = build_user_prompt(
        "Water is H2O.",
        [{"id": "chem_1", "text": "Water has the formula H2O."}],
    )

    assert "Claim: Water is H2O." in prompt
    assert "[chem_1]: Water has the formula H2O." in prompt


def test_extract_citations_deduplicates_in_order():
    assert extract_citations("[p2] and [p1] then [p2]") == ["p2", "p1"]


def test_parse_model_output_handles_not_enough_info_aliases():
    parsed = parse_model_output(
        "Verdict: NOT ENOUGH INFO\nExplanation: Evidence is missing."
    )

    assert parsed.verdict == "NOT_ENOUGH_INFO"
    assert parsed.explanation == "Evidence is missing."


def test_parse_model_output_extracts_verdict_and_citations():
    parsed = parse_model_output(
        "Verdict: REFUTED\nExplanation: The claim is contradicted by [doc_1]."
    )

    assert parsed.verdict == "REFUTED"
    assert parsed.citations == ["doc_1"]


def test_guardrail_converts_irrelevant_cited_support_to_nei():
    raw_text = "Verdict: SUPPORTED\nExplanation: The evidence [doc_geo] confirms it."
    guarded = _guarded_text(
        claim="Mount Everest is the tallest mountain on Earth.",
        raw_text=raw_text,
        parsed=parse_model_output(raw_text),
        hallucinated_citations=[],
        evidence_by_id={
            "doc_geo": "The Mariana Trench is the deepest oceanic trench on Earth."
        },
    )

    assert "NOT_ENOUGH_INFO" in guarded


def test_guardrail_allows_relevant_cited_support():
    raw_text = "Verdict: SUPPORTED\nExplanation: The evidence [doc_science] confirms it."
    guarded = _guarded_text(
        claim="Water is composed of two hydrogen atoms and one oxygen atom.",
        raw_text=raw_text,
        parsed=parse_model_output(raw_text),
        hallucinated_citations=[],
        evidence_by_id={
            "doc_science": (
                "A water molecule has the chemical formula H2O, meaning it "
                "contains one oxygen and two hydrogen atoms."
            )
        },
    )

    assert guarded == raw_text
