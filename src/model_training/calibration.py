"""Temperature scaling for the trained FEVER verifier.

Temperature scaling is a single-parameter post-hoc calibration: divide the
pre-softmax logits by a learned scalar T > 0 before softmax. T is fit by
minimizing NLL on a held-out dev split. It does not change the argmax
prediction, only the confidence.

Usage:
    fit_temperature(logits, labels) -> T
    apply_temperature(logits, T)    -> calibrated probabilities
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


SIDE_CAR_FILENAME = "temperature.json"


def fit_temperature(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    max_iter: int = 200,
    lr: float = 0.05,
    tol: float = 1e-6,
) -> float:
    """Fit a single scalar temperature T by minimizing NLL via L-BFGS.

    Args:
        logits: NxC pre-softmax logits.
        labels: N integer class indices in [0, C).

    Returns:
        Best T as a float, clamped to [0.05, 10.0].
    """
    import torch

    if len(logits) == 0:
        raise ValueError("Cannot fit temperature on an empty dev set.")
    if len(logits) != len(labels):
        raise ValueError("logits and labels must have the same length.")

    logits_t = torch.tensor(logits, dtype=torch.float64)
    labels_t = torch.tensor(labels, dtype=torch.long)

    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_t],
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=tol,
        tolerance_change=tol,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure() -> "torch.Tensor":
        optimizer.zero_grad()
        T = torch.exp(log_t)
        loss = loss_fn(logits_t / T, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(torch.exp(log_t).item())
    return max(0.05, min(10.0, T))


def apply_temperature(
    logits: Sequence[float],
    temperature: float,
) -> list[float]:
    """Return softmax(logits / T) as a Python list."""
    import torch

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    logits_t = torch.tensor(logits, dtype=torch.float64)
    probs = torch.softmax(logits_t / float(temperature), dim=-1)
    return [float(p) for p in probs.tolist()]


def write_temperature_sidecar(
    model_dir: str | Path,
    temperature: float,
    *,
    metadata: dict | None = None,
) -> Path:
    """Write temperature.json next to the model weights."""
    path = Path(model_dir) / SIDE_CAR_FILENAME
    payload = {"temperature": float(temperature)}
    if metadata:
        payload["metadata"] = metadata
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_temperature_sidecar(model_dir: str | Path) -> float | None:
    """Read temperature from a model directory; return None if absent."""
    path = Path(model_dir) / SIDE_CAR_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    T = payload.get("temperature")
    if T is None:
        return None
    return float(T)
