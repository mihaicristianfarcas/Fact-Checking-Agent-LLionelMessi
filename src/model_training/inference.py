"""Inference wrapper for the published fact-checking LoRA adapter."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.model_training.prompting import (
    NOT_ENOUGH_INFO,
    ParsedModelOutput,
    REFUTED,
    SYSTEM_PROMPT,
    SUPPORTED,
    build_user_prompt,
    parse_model_output,
)

logger = logging.getLogger(__name__)

DEFAULT_ADAPTER_ID = "andreiungureanu/Fact-Checking-Agent-LLionelMessi"
DEFAULT_BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@dataclass(frozen=True)
class FactCheckGeneration:
    """Generated verdict plus citation audit details."""

    raw_text: str
    parsed: ParsedModelOutput
    hallucinated_citations: list[str]
    original_text: str | None = None
    guardrail_applied: bool = False

    @property
    def verdict(self) -> str | None:
        return self.parsed.verdict

    @property
    def citations(self) -> list[str]:
        return self.parsed.citations


class FactCheckerInference:
    """Small adapter around Transformers + PEFT for fact-checker inference."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        adapter_id: str = DEFAULT_ADAPTER_ID,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
        *,
        tokenizer_id: str | None = None,
        merge_adapter: bool = False,
        trust_remote_code: bool = True,
        **model_kwargs: Any,
    ) -> "FactCheckerInference":
        """Load the TinyLlama base model and attach the published LoRA adapter.

        The default adapter is the Hugging Face repository published by the
        team.  Heavy ML packages are imported lazily so tests can exercise
        prompt parsing without downloading a model.
        """
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_kwargs = _default_model_kwargs(torch)
        resolved_kwargs.update(model_kwargs)

        tokenizer = _load_tokenizer(
            AutoTokenizer,
            tokenizer_id=tokenizer_id or base_model_id,
            fallback_tokenizer_id=adapter_id,
            trust_remote_code=trust_remote_code,
        )

        logger.info("Loading base model %s", base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=trust_remote_code,
            **resolved_kwargs,
        )

        logger.info("Loading LoRA adapter %s", adapter_id)
        model = PeftModel.from_pretrained(base_model, adapter_id)
        if merge_adapter and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        model.eval()

        return cls(model=model, tokenizer=tokenizer)

    def generate_text(
        self,
        claim: str,
        evidence_passages: Sequence[Mapping[str, Any] | Any],
        *,
        max_new_tokens: int = 96,
        do_sample: bool = False,
        **generate_kwargs: Any,
    ) -> str:
        """Generate the model's raw verdict response for one claim."""
        import torch

        prompt = build_user_prompt(claim, evidence_passages)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        model_device = getattr(self.model, "device", None)
        if model_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(model_device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **generate_kwargs,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_length:]
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    def generate_verdict(
        self,
        claim: str,
        evidence_passages: Sequence[Mapping[str, Any] | Any],
        *,
        max_new_tokens: int = 96,
        **generate_kwargs: Any,
    ) -> FactCheckGeneration:
        """Generate and parse a verdict, with citation hallucination checks.

        The published LoRA adapter is small and can over-commit when evidence
        is merely topically adjacent.  This method therefore applies a
        deterministic guardrail: non-NEI verdicts must cite retrieved IDs and
        at least one cited passage must share meaningful content with the claim.
        Use ``generate_text`` when you need the raw model completion.
        """
        raw_text = self.generate_text(
            claim,
            evidence_passages,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        parsed = parse_model_output(raw_text)
        evidence_by_id = _evidence_by_id(evidence_passages)
        retrieved_ids = set(evidence_by_id)
        hallucinated = [
            citation for citation in parsed.citations if citation not in retrieved_ids
        ]
        guarded_text = _guarded_text(
            claim=claim,
            raw_text=raw_text,
            parsed=parsed,
            hallucinated_citations=hallucinated,
            evidence_by_id=evidence_by_id,
        )
        guardrail_applied = guarded_text != raw_text
        if guardrail_applied:
            parsed = parse_model_output(guarded_text)
            hallucinated = []

        return FactCheckGeneration(
            raw_text=guarded_text,
            parsed=parsed,
            hallucinated_citations=hallucinated,
            original_text=raw_text if guardrail_applied else None,
            guardrail_applied=guardrail_applied,
        )


def load_fact_checker(
    adapter_id: str = DEFAULT_ADAPTER_ID,
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    tokenizer_id: str | None = None,
    **kwargs: Any,
) -> FactCheckerInference:
    """Convenience function for loading the published fact-checker adapter."""
    return FactCheckerInference.from_pretrained(
        adapter_id=adapter_id,
        base_model_id=base_model_id,
        tokenizer_id=tokenizer_id,
        **kwargs,
    )


def _load_tokenizer(
    auto_tokenizer: Any,
    *,
    tokenizer_id: str,
    fallback_tokenizer_id: str,
    trust_remote_code: bool,
) -> Any:
    """Load tokenizer with a fallback for adapter repos with flaky tokenizer files."""
    attempts = [
        {"pretrained_model_name_or_path": tokenizer_id},
        {"pretrained_model_name_or_path": fallback_tokenizer_id},
        {"pretrained_model_name_or_path": tokenizer_id, "use_fast": False},
    ]
    last_error: Exception | None = None

    for attempt in attempts:
        model_id = attempt["pretrained_model_name_or_path"]
        try:
            logger.info("Loading tokenizer from %s", model_id)
            tokenizer = auto_tokenizer.from_pretrained(
                trust_remote_code=trust_remote_code,
                **attempt,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as exc:
            last_error = exc
            logger.warning("Tokenizer load failed for %s: %s", model_id, exc)

    raise OSError("Could not load a tokenizer for fact-checker inference.") from last_error


def _guarded_text(
    *,
    claim: str,
    raw_text: str,
    parsed: ParsedModelOutput,
    hallucinated_citations: Sequence[str],
    evidence_by_id: Mapping[str, str],
) -> str:
    """Return a conservative NEI when the generated verdict is not grounded."""
    if parsed.verdict not in {SUPPORTED, REFUTED}:
        return raw_text

    if not parsed.citations:
        return _guardrail_nei("The model produced a decisive verdict without citations.")

    if hallucinated_citations:
        return _guardrail_nei("The model cited evidence that was not retrieved.")

    cited_passages = [
        evidence_by_id[citation]
        for citation in parsed.citations
        if citation in evidence_by_id
    ]
    if not cited_passages:
        return _guardrail_nei("No cited evidence passage was available for audit.")

    if not any(_is_relevant_to_claim(claim, passage) for passage in cited_passages):
        return _guardrail_nei(
            "The cited evidence is not about the same verifiable claim."
        )

    return raw_text


def _guardrail_nei(reason: str) -> str:
    return (
        f"Verdict: {NOT_ENOUGH_INFO}\n"
        "Explanation: The provided evidence does not contain sufficient "
        f"information to verify or refute this claim. {reason}"
    )


def _evidence_by_id(
    evidence_passages: Sequence[Mapping[str, Any] | Any],
) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for passage in evidence_passages:
        if isinstance(passage, Mapping):
            passage_id = str(passage.get("id", ""))
            text = str(passage.get("text", ""))
        else:
            passage_id = str(getattr(passage, "id", ""))
            text = str(getattr(passage, "text", ""))
        if passage_id:
            evidence[passage_id] = text
    return evidence


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def _is_relevant_to_claim(claim: str, passage: str) -> bool:
    claim_terms = _content_terms(claim)
    passage_terms = _content_terms(passage)
    if not claim_terms or not passage_terms:
        return False

    overlap = claim_terms & passage_terms
    return len(overlap) >= 2


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _default_model_kwargs(torch_module: Any) -> dict[str, Any]:
    if torch_module.cuda.is_available():
        return {"device_map": "auto", "torch_dtype": torch_module.float16}

    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return {"device_map": {"": "mps"}, "torch_dtype": torch_module.float16}

    return {"torch_dtype": torch_module.float32}
