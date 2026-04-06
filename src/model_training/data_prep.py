import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert fact-checking agent. Your task is to review a given Claim along with a list of Evidence passages. "
    "You must output a Verdict of either SUPPORTED, REFUTED, or NOT_ENOUGH_INFO. "
    "You must also provide a short explanation, citing the specific evidence IDs used (e.g., [source_id]). "
    "If the evidence does not clearly support or refute the claim, you must choose NOT_ENOUGH_INFO and explain why."
)

def load_jsonl(filepath: str | Path) -> List[Dict]:
    """Loads a JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def build_user_prompt(claim: str, evidence_passages: List[Dict]) -> str:
    """Builds the user prompt containing the claim and evidence."""
    prompt = f"Claim: {claim}\n\nEvidence:\n"
    if not evidence_passages:
        prompt += "None.\n"
    else:
        for ev in evidence_passages:
            evidence_id = ev.get("id", "unknown")
            text = ev.get("text", "")
            prompt += f"[{evidence_id}]: {text}\n"
    return prompt

def build_assistant_response(verdict: str, evidence_passages: List[Dict]) -> str:
    """Builds the gold assistant response for SFT."""
    response = f"Verdict: {verdict}\nExplanation: "
    if verdict == "NOT_ENOUGH_INFO":
        response += "The provided evidence does not contain sufficient information to verify or refute this claim."
    else:
        # Just cite everything for the baseline
        # A more advanced version would use Person B's stance classifier to only cite SUPPORTING/REFUTING
        ids = [f"[{ev['id']}]" for ev in evidence_passages if 'id' in ev]
        citation_str = " ".join(ids) if ids else "[No Citation]"
        explanation = "The claim is " + ("supported" if verdict == "SUPPORTED" else "refuted") + f" by the provided evidence {citation_str}."
        response += explanation
    return response

def prepare_sft_dataset(input_filepath: str | Path, tokenizer=None, max_samples: int = None) -> Dataset:
    """
    Converts raw triples into a HuggingFace Dataset ready for SFT.
    We return standard Chat Dicts. The SFT Trainer will use apply_chat_template natively.
    """
    raw_data = load_jsonl(input_filepath)
    if max_samples:
        raw_data = raw_data[:max_samples]
        
    formatted_data = []
    for item in raw_data:
        claim = item.get("claim_text", "")
        evidence = item.get("evidence_passages", [])
        verdict = item.get("verdict", "NOT_ENOUGH_INFO")
        
        user_text = build_user_prompt(claim, evidence)
        assistant_text = build_assistant_response(verdict, evidence)
        
        # Standard chat format list
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]
        formatted_data.append({"messages": messages})
        
    logger.info(f"Prepared SFT dataset with {len(formatted_data)} samples.")
    return Dataset.from_list(formatted_data)

def prepare_dpo_dataset(input_filepath: str | Path, max_samples: int = None) -> Dataset:
    """
    Creates pairs of Chosen vs Rejected responses for Preference Learning (DPO) 
    to heavily penalize hallucination.
    """
    raw_data = load_jsonl(input_filepath)
    if max_samples:
        raw_data = raw_data[:max_samples]
        
    dpo_data = []
    for item in raw_data:
        claim = item.get("claim_text", "")
        evidence = item.get("evidence_passages", [])
        verdict = item.get("verdict", "NOT_ENOUGH_INFO")
        
        user_text = build_user_prompt(claim, evidence)
        
        # We only generate pairs if we can craft a "bad" response.
        # 1. Hallucinated Support Failure: When the answer is NEI, punish "Expert Guessing"
        if verdict == "NOT_ENOUGH_INFO":
            chosen = build_assistant_response(verdict, evidence)
            rejected = "Verdict: SUPPORTED\nExplanation: The claim is definitely true because the context implies it. [hallucinated_source_1]"
            
            dpo_data.append({
                "system": SYSTEM_PROMPT,
                "prompt": user_text,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}]
            })
        
        # 2. Negation Failure: When the answer is REFUTED, punish "Blind Agreement"
        elif verdict == "REFUTED":
            chosen = build_assistant_response(verdict, evidence)
            # Create a "Lazy" rejected response that says supported just because keywords matched
            rejected = "Verdict: SUPPORTED\nExplanation: The claim is supported because the evidence mentions the subject. [unknown_id]"
            
            dpo_data.append({
                "system": SYSTEM_PROMPT,
                "prompt": user_text,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}]
            })
            
    logger.info(f"Prepared DPO dataset with {len(dpo_data)} preference pairs.")
    return Dataset.from_list(dpo_data)

if __name__ == "__main__":
    test_path = Path("data/processed/val.jsonl")
    if test_path.exists():
        print("Testing SFT Prep (first 2 samples):")
        ds_sft = prepare_sft_dataset(test_path, max_samples=2)
        print(ds_sft[0]['messages'])
        
        print("\nTesting DPO Prep (first 2 samples):")
        ds_dpo = prepare_dpo_dataset(test_path, max_samples=20) # Need enough to find some NEI samples
        if len(ds_dpo) > 0:
            print("Prompt:", ds_dpo[0]['prompt'])
            print("Chosen:", ds_dpo[0]['chosen'])
            print("Rejected:", ds_dpo[0]['rejected'])
