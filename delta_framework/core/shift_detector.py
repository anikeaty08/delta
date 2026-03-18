"""Distribution / representation shift detection utilities.

We model per-class embedding distributions as diagonal Gaussians and compute
closed-form KL divergence between the same class before vs after an update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ShiftResult:
    shift_score: float
    shift_detected: bool
    per_class_drift: Dict[int, float]


def fit_diag_gaussian(x: np.ndarray, *, min_var: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Fit diagonal Gaussian parameters (mean, var) for x[N, D]."""
    if x.size == 0:
        raise ValueError("Cannot fit Gaussian: empty input.")
    mu = x.mean(axis=0)
    var = x.var(axis=0) + float(min_var)
    return mu, var


def kl_diag_gaussian(
    mu_p: np.ndarray,
    var_p: np.ndarray,
    mu_q: np.ndarray,
    var_q: np.ndarray,
) -> float:
    """KL( P || Q ) for diagonal Gaussians P=N(mu_p,var_p), Q=N(mu_q,var_q)."""
    # 0.5 * sum( log(var_q/var_p) + (var_p + (mu_p-mu_q)^2)/var_q - 1 )
    var_ratio = var_q / var_p
    diff = mu_p - mu_q
    term = np.log(var_ratio) + (var_p + diff * diff) / var_q - 1.0
    return float(0.5 * np.sum(term))


def detect_shift_from_embeddings(
    *,
    before_by_class: Dict[int, np.ndarray],
    after_by_class: Dict[int, np.ndarray],
    threshold: float = 0.3,
    aggregate: str = "mean",
) -> ShiftResult:
    """Compute per-class KL drift and an aggregate shift score."""
    common = sorted(set(before_by_class.keys()) & set(after_by_class.keys()))
    per_class: Dict[int, float] = {}
    for c in common:
        mu_b, var_b = fit_diag_gaussian(before_by_class[c])
        mu_a, var_a = fit_diag_gaussian(after_by_class[c])
        per_class[c] = kl_diag_gaussian(mu_b, var_b, mu_a, var_a)

    if not per_class:
        score = 0.0
    else:
        vals = np.asarray(list(per_class.values()), dtype=np.float64)
        if aggregate == "max":
            score = float(vals.max())
        elif aggregate == "mean":
            score = float(vals.mean())
        else:
            raise ValueError("aggregate must be 'mean' or 'max'")

    return ShiftResult(
        shift_score=float(score),
        shift_detected=bool(score > threshold),
        per_class_drift=per_class,
    )


def extract_embeddings_by_class(
    *,
    model: Any,
    data_loader: Any,
    device: Any,
    class_ids: Optional[Sequence[int]] = None,
    max_per_class: int = 256,
) -> Dict[int, np.ndarray]:
    """Extract penultimate embeddings (model.extract_vector) grouped by class."""
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("`torch` is required for embedding extraction.") from e

    model.eval()
    selected = set(class_ids) if class_ids is not None else None

    out: Dict[int, List[np.ndarray]] = {}
    counts: Dict[int, int] = {}

    with torch.no_grad():
        for images, targets, _task_ids in data_loader:
            images = images.to(device, non_blocking=True)
            feats = model.extract_vector(images).detach().cpu().numpy()
            t = targets.detach().cpu().numpy().astype(np.int64)

            for i in range(len(t)):
                c = int(t[i])
                if selected is not None and c not in selected:
                    continue
                cur = counts.get(c, 0)
                if cur >= max_per_class:
                    continue
                out.setdefault(c, []).append(feats[i : i + 1])
                counts[c] = cur + 1

    return {c: np.concatenate(v, axis=0) for c, v in out.items()}


def detect_shift_for_models(
    *,
    before_model: Any,
    after_model: Any,
    dataset: Any,
    device: Any,
    class_ids: Sequence[int],
    batch_size: int = 128,
    num_workers: int = 2,
    max_per_class: int = 256,
    threshold: float = 0.3,
    aggregate: str = "mean",
) -> ShiftResult:
    """Convenience wrapper: compute drift on a dataset for specified classes."""
    try:
        from torch.utils.data import DataLoader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("`torch` is required for shift detection on models/datasets.") from e

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=getattr(device, "type", None) == "cuda",
    )
    before = extract_embeddings_by_class(
        model=before_model,
        data_loader=loader,
        device=device,
        class_ids=class_ids,
        max_per_class=max_per_class,
    )
    after = extract_embeddings_by_class(
        model=after_model,
        data_loader=loader,
        device=device,
        class_ids=class_ids,
        max_per_class=max_per_class,
    )
    return detect_shift_from_embeddings(
        before_by_class=before,
        after_by_class=after,
        threshold=threshold,
        aggregate=aggregate,
    )


