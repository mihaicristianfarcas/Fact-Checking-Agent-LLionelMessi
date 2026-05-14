"""Device-selection helpers shared across training and inference modules."""

from __future__ import annotations


def pick_device() -> str:
    """Return the best available torch device string: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
