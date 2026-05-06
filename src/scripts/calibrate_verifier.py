"""Fit temperature scaling for the trained FEVER verifier.

Reads the verifier-train dev JSONL, runs the trained model to collect logits,
fits a single scalar T by minimizing NLL, and writes ``temperature.json`` next
to the model weights. ``FeverVerdictVerifier`` will pick up the sidecar
automatically on next load.

Usage:
    python -m src.scripts.calibrate_verifier \
        --model-path models/fever_verifier_deberta_base_20k_fast \
        --dev-file data/processed/fever_verifier_dev_2k_fast.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from src.claim_processing.verdict_verifier import LABEL_TO_ID
from src.model_training.calibration import (
    fit_temperature,
    write_temperature_sidecar,
)


def _read_dev_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _collect_logits(
    model_path: str,
    rows: list[dict],
    *,
    max_length: int,
    batch_size: int,
) -> tuple[list[list[float]], list[int]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Loading {} on {}", model_path, device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device).eval()

    label2id = getattr(model.config, "label2id", None) or LABEL_TO_ID

    all_logits: list[list[float]] = []
    all_labels: list[int] = []
    skipped = 0

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [row["text"] for row in batch]
        labels: list[int] = []
        for row in batch:
            label = row.get("label")
            if label not in label2id:
                skipped += 1
                labels.append(-1)
                continue
            labels.append(int(label2id[label]))

        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits.detach().cpu().tolist()

        for logit, label in zip(logits, labels):
            if label < 0:
                continue
            all_logits.append(logit)
            all_labels.append(label)

    if skipped:
        logger.warning("Skipped {} rows with unknown labels", skipped)
    return all_logits, all_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--text-key",
        default="text",
        help="JSONL key for the verifier input text (default 'text').",
    )
    args = parser.parse_args()

    dev_path = Path(args.dev_file)
    rows = _read_dev_jsonl(dev_path)
    if args.text_key != "text":
        rows = [{**row, "text": row[args.text_key]} for row in rows]
    logger.info("Loaded {} dev rows from {}", len(rows), dev_path)

    logits, labels = _collect_logits(
        args.model_path,
        rows,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    logger.info("Collected logits for {} dev examples", len(logits))

    T = fit_temperature(logits, labels)
    logger.info("Fitted temperature T = {:.4f}", T)

    sidecar_path = write_temperature_sidecar(
        args.model_path,
        T,
        metadata={
            "dev_file": str(dev_path),
            "dev_size": len(logits),
            "max_length": args.max_length,
        },
    )
    logger.info("Wrote {}", sidecar_path)


if __name__ == "__main__":
    main()
