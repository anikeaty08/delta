"""Equivalence checking utilities (delta-only vs full retrain)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _to_numpy_confusion(conf: Any) -> np.ndarray:
    a = np.asarray(conf)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Confusion matrix must be square; got shape {a.shape}")
    return a.astype(np.float64, copy=False)


def confusion_cosine_similarity(conf_a: Any, conf_b: Any) -> float:
    """Cosine similarity between flattened, sum-normalized confusion matrices."""
    a = _to_numpy_confusion(conf_a)
    b = _to_numpy_confusion(conf_b)
    if a.shape != b.shape:
        n = min(a.shape[0], b.shape[0])
        a = a[:n, :n]
        b = b[:n, :n]

    a = a / max(float(a.sum()), 1.0)
    b = b / max(float(b.sum()), 1.0)

    av = a.reshape(-1)
    bv = b.reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 1.0
    return float(np.dot(av, bv) / denom)


def compute_savings_percent(t_full_s: float, t_delta_s: float) -> float:
    t_full_s = float(t_full_s)
    t_delta_s = float(t_delta_s)
    if t_full_s <= 0:
        return 0.0
    return float(((t_full_s - t_delta_s) / t_full_s) * 100.0)


def summarize_equivalence(
    *,
    full_metrics: Dict[str, Any],
    delta_metrics: Dict[str, Any],
    timing_full_s: float,
    timing_delta_s: float,
    mem_full_mb: float = 0.0,
    mem_delta_mb: float = 0.0,
    acc_key: str = "top1",
    equivalence_threshold: float = 0.005,
) -> Dict[str, Any]:
    """Return an equivalence summary dict for UI/JSON consumption."""
    acc_full = float(full_metrics.get(acc_key, 0.0))
    acc_delta = float(delta_metrics.get(acc_key, 0.0))

    ece_full = float(full_metrics.get("ece", 0.0))
    ece_delta = float(delta_metrics.get("ece", 0.0))

    conf_full = full_metrics.get("confusion_matrix")
    conf_delta = delta_metrics.get("confusion_matrix")
    confusion_similarity: Optional[float]
    if conf_full is None or conf_delta is None:
        confusion_similarity = None
    else:
        confusion_similarity = confusion_cosine_similarity(conf_full, conf_delta)

    gap = abs(acc_full - acc_delta)

    return {
        "equivalence_gap": float(gap),
        "is_equivalent": bool(gap < float(equivalence_threshold)),
        "calibration_diff": float(abs(ece_full - ece_delta)),
        "confusion_similarity": confusion_similarity,
        "timing_full_s": float(timing_full_s),
        "timing_delta_s": float(timing_delta_s),
        "compute_savings_percent": float(compute_savings_percent(timing_full_s, timing_delta_s)),
        "mem_full_mb": float(mem_full_mb),
        "mem_delta_mb": float(mem_delta_mb),
    }


