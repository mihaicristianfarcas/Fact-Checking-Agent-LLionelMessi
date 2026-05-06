"""Tests for FEVER decontamination helpers."""

import json
from pathlib import Path

import pytest

from src.evaluation.fever_utils import (
    load_train_claim_texts,
    normalize_claim_text,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_load_train_claim_texts_accepts_sft_triples_key(tmp_path):
    """SFT triples use the 'claim_text' key."""
    triples = tmp_path / "train.jsonl"
    _write_jsonl(triples, [
        {"claim_text": "Lionel Messi is Argentine.", "verdict": "SUPPORTED"},
        {"claim_text": "Water is wet.", "verdict": "SUPPORTED"},
    ])

    claims = load_train_claim_texts(triples)

    assert normalize_claim_text("Lionel Messi is Argentine.") in claims
    assert len(claims) == 2


def test_load_train_claim_texts_accepts_verifier_train_key(tmp_path):
    """Verifier-train JSONL (built by build_fever_verifier_dataset.py) uses 'claim'."""
    verifier_train = tmp_path / "fever_verifier_train.jsonl"
    _write_jsonl(verifier_train, [
        {"claim": "FEVER claim 1.", "label": "SUPPORTED"},
        {"claim": "FEVER claim 2.", "label": "REFUTED"},
    ])

    claims = load_train_claim_texts(verifier_train)

    assert normalize_claim_text("FEVER claim 1.") in claims
    assert normalize_claim_text("FEVER claim 2.") in claims
    assert len(claims) == 2


def test_load_train_claim_texts_accepts_multiple_paths(tmp_path):
    """Caller can union exclusions from both SFT triples and verifier train."""
    sft = tmp_path / "sft.jsonl"
    _write_jsonl(sft, [{"claim_text": "alpha"}])
    verifier = tmp_path / "verifier.jsonl"
    _write_jsonl(verifier, [{"claim": "beta"}, {"claim": "alpha"}])

    claims = load_train_claim_texts([sft, verifier])

    assert claims == {normalize_claim_text("alpha"), normalize_claim_text("beta")}


def test_load_train_claim_texts_raises_on_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_train_claim_texts(tmp_path / "nonexistent.jsonl")
