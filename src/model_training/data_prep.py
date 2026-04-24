import json
import logging
import random
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

def _has_evidence(item: Dict) -> bool:
    """Returns True if the sample has at least one evidence passage."""
    return len(item.get("evidence_passages", [])) > 0


def prepare_sft_dataset(input_filepath: str | Path, tokenizer=None, max_samples: int = None) -> Dataset:
    """
    Converts raw triples into a HuggingFace Dataset ready for SFT.
    We return standard Chat Dicts. The SFT Trainer will use apply_chat_template natively.

    Filtering: SUPPORTED and REFUTED samples with no evidence passages are dropped.
    Training the model to assign confident verdicts without any evidence undermines
    the "abstention over fabrication" goal. NOT_ENOUGH_INFO samples with empty
    evidence are kept — they correctly teach the model to abstain when nothing is found.
    """
    raw_data = load_jsonl(input_filepath)

    # Filter before capping so max_samples reflects usable samples, not raw rows.
    raw_data = [
        item for item in raw_data
        if item.get("verdict") == "NOT_ENOUGH_INFO" or _has_evidence(item)
    ]
    logger.info(f"After evidence filter: {len(raw_data)} samples retained.")

    if max_samples:
        random.seed(42)
        random.shuffle(raw_data)
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

# Varied rejection templates per verdict class.
# Using round-robin selection (index % len) keeps generation deterministic.
_NEI_REJECTIONS = [
    "Verdict: SUPPORTED\nExplanation: The claim is definitely true because the context implies it. [hallucinated_source_1]",
    "Verdict: REFUTED\nExplanation: This claim is incorrect based on general knowledge alone. [hallucinated_source_2]",
    "Verdict: SUPPORTED\nExplanation: Based on common knowledge this claim is accurate, so the evidence is not needed. [assumed_source]",
    "Verdict: REFUTED\nExplanation: The phrasing of the claim sounds suspicious, so it must be false. [unknown_id]",
]

_REFUTED_REJECTIONS = [
    "Verdict: SUPPORTED\nExplanation: The claim is supported because the evidence mentions the subject. [unknown_id]",
    "Verdict: SUPPORTED\nExplanation: This appears to be correct based on the information provided. [source_1]",
    "Verdict: SUPPORTED\nExplanation: The evidence is consistent with the claim. [evidence_0]",
    "Verdict: SUPPORTED\nExplanation: The subject of the claim appears in the evidence, confirming it. [evidence_1]",
]

_SUPPORTED_REJECTIONS = [
    "Verdict: REFUTED\nExplanation: The claim contradicts the evidence based on keyword analysis. [unknown_id]",
    "Verdict: NOT_ENOUGH_INFO\nExplanation: There is not sufficient evidence to support this claim.",
    "Verdict: REFUTED\nExplanation: This claim appears to be false based on general knowledge. [hallucinated_source]",
    "Verdict: NOT_ENOUGH_INFO\nExplanation: The evidence is ambiguous and does not clearly confirm the claim.",
]


def prepare_dpo_dataset(input_filepath: str | Path, max_samples: int = None) -> Dataset:
    """
    Creates pairs of Chosen vs Rejected responses for Preference Learning (DPO)
    to heavily penalize hallucination.

    Covers all three verdict classes (NOT_ENOUGH_INFO, REFUTED, SUPPORTED) with
    varied rejection templates to prevent the model from pattern-matching on a
    single fixed rejected string.
    """
    raw_data = load_jsonl(input_filepath)

    # Same evidence filter as SFT — don't train preferences on evidence-free verdicts.
    raw_data = [
        item for item in raw_data
        if item.get("verdict") == "NOT_ENOUGH_INFO" or _has_evidence(item)
    ]
    logger.info(f"DPO — after evidence filter: {len(raw_data)} samples retained.")

    if max_samples:
        random.seed(42)
        random.shuffle(raw_data)
        raw_data = raw_data[:max_samples]

    dpo_data = []
    nei_idx = refuted_idx = supported_idx = 0

    for item in raw_data:
        claim = item.get("claim_text", "")
        evidence = item.get("evidence_passages", [])
        verdict = item.get("verdict", "NOT_ENOUGH_INFO")

        user_text = build_user_prompt(claim, evidence)
        chosen = build_assistant_response(verdict, evidence)

        if verdict == "NOT_ENOUGH_INFO":
            # Punish overconfident hallucination in both directions.
            rejected = _NEI_REJECTIONS[nei_idx % len(_NEI_REJECTIONS)]
            nei_idx += 1

        elif verdict == "REFUTED":
            # Punish blind agreement (lazy SUPPORTED when evidence refutes).
            rejected = _REFUTED_REJECTIONS[refuted_idx % len(_REFUTED_REJECTIONS)]
            refuted_idx += 1

        else:  # SUPPORTED
            # Punish false refutation and spurious abstention.
            rejected = _SUPPORTED_REJECTIONS[supported_idx % len(_SUPPORTED_REJECTIONS)]
            supported_idx += 1

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
