"""Train a DeBERTa FEVER verdict verifier.

Input JSONL rows are produced by:

    python -m src.scripts.build_fever_verifier_dataset ...

Each row must contain:
    input_text: "Claim: ... Evidence: ..."
    label: SUPPORTED | REFUTED | NOT_ENOUGH_INFO
"""

from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.claim_processing.verdict_verifier import ID_TO_LABEL, LABEL_TO_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load verifier JSONL rows."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_rows(rows: list[dict[str, Any]], *, path: str | Path) -> None:
    """Fail early on malformed training rows."""
    if not rows:
        raise ValueError(f"No rows found in {path}")

    for idx, row in enumerate(rows[:10]):
        if not row.get("input_text"):
            raise ValueError(f"Row {idx} in {path} is missing input_text")
        label = row.get("label")
        if label not in LABEL_TO_ID:
            raise ValueError(
                f"Row {idx} in {path} has unsupported label {label!r}; "
                f"expected one of {sorted(LABEL_TO_ID)}"
            )


def maybe_split_train_eval(
    train_rows: list[dict[str, Any]],
    *,
    eval_rows: list[dict[str, Any]] | None,
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Use explicit eval rows or split a small validation set from train JSONL."""
    if eval_rows is not None:
        return train_rows, eval_rows
    if eval_fraction <= 0:
        return train_rows, None

    labels = [row["label"] for row in train_rows]
    train_split, eval_split = train_test_split(
        train_rows,
        test_size=eval_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return list(train_split), list(eval_split)


def make_dataset(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    max_length: int,
) -> Dataset:
    """Create a tokenized HuggingFace Dataset."""
    dataset = Dataset.from_list(
        [
            {
                "input_text": row["input_text"],
                "labels": LABEL_TO_ID[row["label"]],
            }
            for row in rows
        ]
    )

    def tokenize(batch):
        return tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, desc="Tokenizing verifier data")
    return tokenized.remove_columns(["input_text"])


def load_model_and_tokenizer(model_name: str):
    """Load base model/tokenizer with the verifier label mapping."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {
        "num_labels": len(LABEL_TO_ID),
        "id2label": ID_TO_LABEL,
        "label2id": LABEL_TO_ID,
        "ignore_mismatched_sizes": True,
    }
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            dtype=torch.float32,
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            **model_kwargs,
        )
    model.config.problem_type = "single_label_classification"
    return model, tokenizer


def training_args_from_cli(args: argparse.Namespace, *, has_eval: bool) -> TrainingArguments:
    """Build TrainingArguments across transformers 4.x/5.x naming differences."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"

    fp16 = bool(args.fp16)
    if fp16 and not torch.cuda.is_available():
        logger.warning("--fp16 was requested but CUDA is not available; disabling fp16")
        fp16 = False

    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size or args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
        "fp16": fp16,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": args.optim,
    }

    if args.max_steps is not None:
        kwargs["max_steps"] = args.max_steps
    if args.warmup_steps is not None:
        kwargs["warmup_steps"] = args.warmup_steps
    else:
        kwargs["warmup_ratio"] = args.warmup_ratio

    if has_eval:
        kwargs[eval_key] = "epoch"
        kwargs["save_strategy"] = "epoch"
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "macro_f1"
        kwargs["greater_is_better"] = True
    else:
        kwargs[eval_key] = "no"
        kwargs["save_strategy"] = "epoch"

    supported = {key: value for key, value in kwargs.items() if key in params}
    return TrainingArguments(**supported)


def compute_metrics(eval_pred) -> dict[str, float]:
    """Trainer metrics for three-way FEVER verdict classification."""
    logits = (
        eval_pred.predictions
        if hasattr(eval_pred, "predictions")
        else eval_pred[0]
    )
    labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def maybe_limit(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    """Deterministically cap rows for quick debug runs."""
    if not limit or limit <= 0 or limit >= len(rows):
        return rows
    sampled = rows[:]
    random.Random(seed).shuffle(sampled)
    return sampled[:limit]


def label_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count labels with deterministic key order."""
    counts = Counter(row["label"] for row in rows)
    return {label: int(counts.get(label, 0)) for label in LABEL_TO_ID}


def compute_class_weights(rows: list[dict[str, Any]]) -> torch.Tensor:
    """Inverse-frequency CE weights, normalized to mean=1."""
    counts = np.zeros(len(LABEL_TO_ID), dtype=np.float32)
    for row in rows:
        counts[LABEL_TO_ID[row["label"]]] += 1

    if np.any(counts == 0):
        missing = [ID_TO_LABEL[idx] for idx, count in enumerate(counts) if count == 0]
        raise ValueError(
            "Training set is missing one or more FEVER labels; missing="
            f"{missing}. Rebuild data with all three labels present."
        )

    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class ClassWeightedTrainer(Trainer):
    """Trainer with optional class-weighted cross-entropy."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)

        if labels is None:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        weights = (
            self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a FEVER DeBERTa verifier")
    parser.add_argument("--train-file", type=str, required=False)
    parser.add_argument("--eval-file", type=str, default=None)
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.0,
        help="Split this fraction from --train-file when --eval-file is omitted.",
    )
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/fever_verifier_deberta_base",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument(
        "--class-weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inverse-frequency class weights to reduce majority-class collapse.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/cache tokenizer and model, then exit without training.",
    )
    args = parser.parse_args()

    logger.info(f"Loading tokenizer/model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    if args.download_only:
        logger.info("Download-only requested; model/tokenizer are cached.")
        return 0

    if not args.train_file:
        raise ValueError("--train-file is required unless --download-only is used")

    train_rows = load_jsonl(args.train_file)
    validate_rows(train_rows, path=args.train_file)
    train_rows = maybe_limit(train_rows, args.max_train_samples, args.seed)

    eval_rows = None
    if args.eval_file:
        eval_rows = load_jsonl(args.eval_file)
        validate_rows(eval_rows, path=args.eval_file)
        eval_rows = maybe_limit(eval_rows, args.max_eval_samples, args.seed)

    train_rows, eval_rows = maybe_split_train_eval(
        train_rows,
        eval_rows=eval_rows,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )

    logger.info(f"Training rows: {len(train_rows):,}")
    logger.info(f"Training label distribution: {label_distribution(train_rows)}")
    if eval_rows is not None:
        logger.info(f"Evaluation rows: {len(eval_rows):,}")
        logger.info(f"Evaluation label distribution: {label_distribution(eval_rows)}")

    train_dataset = make_dataset(train_rows, tokenizer, max_length=args.max_length)
    eval_dataset = (
        make_dataset(eval_rows, tokenizer, max_length=args.max_length)
        if eval_rows is not None
        else None
    )

    training_args = training_args_from_cli(args, has_eval=eval_dataset is not None)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics if eval_dataset is not None else None,
    }

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    class_weights = compute_class_weights(train_rows) if args.class_weighting else None
    if class_weights is not None:
        logger.info(
            "Using class weights: %s",
            {
                ID_TO_LABEL[idx]: round(float(weight), 4)
                for idx, weight in enumerate(class_weights.tolist())
            },
        )

    trainer = ClassWeightedTrainer(
        **trainer_kwargs,
        class_weights=class_weights,
    )

    logger.info("Starting FEVER verifier training...")
    trainer.train()

    metrics = trainer.evaluate() if eval_dataset is not None else {}
    logger.info(f"Final eval metrics: {metrics}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    summary = {
        "model_name": args.model_name,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows) if eval_rows is not None else 0,
        "max_length": args.max_length,
        "labels": LABEL_TO_ID,
        "metrics": metrics,
    }
    (output_dir / "verifier_training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Saved verifier to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
