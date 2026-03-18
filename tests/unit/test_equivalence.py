import numpy as np
import pytest

from delta_framework.core.equivalence import (
    compute_savings_percent,
    confusion_cosine_similarity,
    summarize_equivalence,
)


def test_confusion_cosine_similarity_identical_is_one():
    cm = np.eye(3, dtype=np.float64)
    assert confusion_cosine_similarity(cm, cm) == 1.0


def test_confusion_cosine_similarity_shape_mismatch_is_handled():
    a = np.eye(4, dtype=np.float64)
    b = np.eye(3, dtype=np.float64)
    sim = confusion_cosine_similarity(a, b)
    assert 0.99 <= sim <= 1.0


def test_compute_savings_percent_basic():
    assert compute_savings_percent(100.0, 25.0) == 75.0
    assert compute_savings_percent(0.0, 10.0) == 0.0


def test_summarize_equivalence_keys_and_threshold():
    full = {
        "top1": 0.9,
        "ece": 0.02,
        "confusion_matrix": np.eye(2),
        "per_class_acc": [0.9, 0.8],
    }
    delta = {
        "top1": 0.897,
        "ece": 0.03,
        "confusion_matrix": np.eye(2),
        "per_class_acc": [0.89, 0.78],
    }
    out = summarize_equivalence(
        full_metrics=full,
        delta_metrics=delta,
        timing_full_s=10.0,
        timing_delta_s=2.0,
        equivalence_threshold=0.01,
    )
    assert out["equivalence_gap"] == abs(0.9 - 0.897)
    assert out["is_equivalent"] is True
    assert out["compute_savings_percent"] == 80.0
    assert out["confusion_similarity"] == pytest.approx(1.0, abs=1e-12)
    assert out["worst_class_acc_gap"] == pytest.approx(0.02)
    assert out["mean_class_acc_gap"] == pytest.approx(0.015)
