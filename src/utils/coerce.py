"""Defensive type-coercion helpers used at module boundaries."""

from __future__ import annotations

from typing import Any


def as_float(value: Any, default: float) -> float:
    """Coerce `value` to float, falling back to `default` on TypeError/ValueError."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metadata_float(metadata: dict, key: str, default: float) -> float:
    """Look up `key` in `metadata` and coerce to float (or return `default`)."""
    return as_float(metadata.get(key, default), default)
