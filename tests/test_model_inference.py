"""
Integration Test for Trained SFT & DPO Fact-Checking Models.

Run with:
    RUN_MODEL_INFERENCE_TESTS=1 pytest tests/test_model_inference.py -v -s
"""

import os
import pytest

from src.model_training.inference import DEFAULT_ADAPTER_ID, FactCheckerInference


@pytest.fixture(scope="module")
def fact_checker_pipeline():
    """
    Loads the published DPO adapter and tokenizer only once.

    This is intentionally opt-in because it downloads TinyLlama plus the LoRA
    adapter and is too heavy for the normal unit-test loop.
    """
    if os.getenv("RUN_MODEL_INFERENCE_TESTS") != "1":
        pytest.skip("Set RUN_MODEL_INFERENCE_TESTS=1 to download and test the HF model.")

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    adapter_id = os.getenv("FACTCHECK_MODEL_REPO_ID", DEFAULT_ADAPTER_ID)
    generator = FactCheckerInference.from_pretrained(adapter_id=adapter_id)
    
    def generate_verdict(claim, evidence):
        """Helper inference wrapper using the production citation guardrail."""
        result = generator.generate_verdict(
            claim,
            evidence,
            max_new_tokens=80,
            do_sample=False,
        )
        return result.raw_text
        
    return generate_verdict


def test_fact_checker_supported_claim(fact_checker_pipeline):
    """
    Test if the model correctly asserts a SUPPORTED claim 
    and importantly, correctly handles parsing evidence citations.
    """
    claim = "Water is composed of two hydrogen atoms and one oxygen atom."
    list_of_evidence = [
        {"id": "doc_science", "text": "A water molecule has the chemical formula H2O, meaning it contains one oxygen and two hydrogen atoms structurally bound together."}
    ]
    
    result = fact_checker_pipeline(claim, list_of_evidence)
    
    # Verification Rules
    assert "SUPPORTED" in result.upper(), f"Model hallucinated or failed. Expected SUPPORTED. Dump: {result}"
    assert "doc_science" in result, "Model failed to cite its sources correctly from the prompt."


def test_fact_checker_dpo_abstention(fact_checker_pipeline):
    """
    Test if the guarded inference path suppresses hallucinated confidence.
    Even though the claim is true in real life, because the evidence doesn't answer it,
    it MUST output NOT_ENOUGH_INFO. 
    """
    claim = "Mount Everest is the tallest mountain on Earth."
    # The provided evidence discusses the ocean, not mountains.
    list_of_evidence = [
        {"id": "doc_geo", "text": "The Mariana Trench is the deepest oceanic trench on Earth."}
    ]
    
    result = fact_checker_pipeline(claim, list_of_evidence)
    
    # Verification Rules
    assert "NOT_ENOUGH_INFO" in result.upper(), (
        "Guarded inference failed to suppress a known claim out-of-context. "
        f"Dump: {result}"
    )

def test_fact_checker_refuted_claim(fact_checker_pipeline):
    """
    Test if the model correctly asserts a REFUTED claim when evidence explicitly contradicts it.
    """
    claim = "The Moon is made entirely of green cheese."
    list_of_evidence = [
        {"id": "doc_astro", "text": "The Moon consists primarily of solid rock and dust, with a core of iron and nickel. There is no biological matter or cheese on the Moon."}
    ]
    
    result = fact_checker_pipeline(claim, list_of_evidence)
    
    # Verification Rules
    assert "REFUTED" in result.upper(), f"Expected REFUTED. Dump: {result}"
    assert "doc_astro" in result, "Model failed to cite its contradictory source."


def test_fact_checker_missing_evidence(fact_checker_pipeline):
    """
    Test how the model behaves when it receives absolutely zero evidence.
    It MUST abstain, prioritizing our safety guidelines over hallucinations.
    """
    claim = "There is a secret alien base on the dark side of the moon."
    list_of_evidence = [] # Zero retrieved documents
    
    result = fact_checker_pipeline(claim, list_of_evidence)
    
    # Verification Rules
    assert "NOT_ENOUGH_INFO" in result.upper(), f"Expected robust abstention when given zero evidence. Dump: {result}"


def test_fact_checker_complex_claim(fact_checker_pipeline):
    """
    Test a real-world multi-part claim to ensure citing multiple documents holds up.
    """
    claim = "Albert Einstein won the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect."
    list_of_evidence = [
        {"id": "hist_1", "text": "The Nobel Prize in Physics 1921 was awarded to Albert Einstein."},
        {"id": "hist_2", "text": "He was awarded it especially for his discovery of the law of the photoelectric effect."}
    ]
    
    result = fact_checker_pipeline(claim, list_of_evidence)
    
    assert "SUPPORTED" in result.upper(), f"Expected SUPPORTED on complex claim. Dump: {result}"
    assert "hist_1" in result or "hist_2" in result, "Failed to cite proper historical documents."
