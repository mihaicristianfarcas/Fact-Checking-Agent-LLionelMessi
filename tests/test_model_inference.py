"""
Integration Test for Trained SFT & DPO Fact-Checking Models.

Run with:
    pytest tests/test_model_training.py -v -s
"""

import os
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.model_training.data_prep import build_user_prompt, SYSTEM_PROMPT


MODEL_DIR = "./models/fact_checker_dpo"


@pytest.fixture(scope="module")
def fact_checker_pipeline():
    """
    Loads the trained DPO model and tokenizer only once for the entire test module.
    If the model hasn't been trained yet, it skips all tests in this file.
    """
    if not os.path.exists(MODEL_DIR):
        pytest.skip(f"Trained model not found at {MODEL_DIR}. Please run the training pipeline first.")
        
    has_cuda = torch.cuda.is_available()
    
    # 1. Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto" if has_cuda else "cpu",
        torch_dtype=torch.float16 if has_cuda else torch.float32,
        trust_remote_code=True
    )
    
    # 2. Append Custom Model Weights (LoRA)
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    def generate_verdict(claim, evidence):
        """Helper inference wrapper"""
        prompt = build_user_prompt(claim, evidence)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Format strings matching HuggingFace ChatML logic
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Factual generation logic via greedy decoding
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return response
        
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
    Test if the DPO pipeline successfully penalized hallucinated confidence.
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
        "Model hallucinated! The DPO pipeline failed to suppress it from answering a known claim out-of-context. "
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
