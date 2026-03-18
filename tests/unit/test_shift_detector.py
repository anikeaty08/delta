import numpy as np

from delta_framework.core.shift_detector import (
    detect_shift_from_embeddings,
    fit_diag_gaussian,
    kl_diag_gaussian,
)


def test_fit_diag_gaussian_shapes():
    x = np.zeros((5, 3), dtype=np.float64)
    mu, var = fit_diag_gaussian(x)
    assert mu.shape == (3,)
    assert var.shape == (3,)
    assert np.all(var > 0.0)


def test_kl_diag_gaussian_is_zero_for_identical_distributions():
    mu = np.zeros((4,), dtype=np.float64)
    var = np.ones((4,), dtype=np.float64)
    assert kl_diag_gaussian(mu, var, mu, var) == 0.0


def test_detect_shift_from_embeddings_flags_shift():
    rng = np.random.default_rng(0)
    before = {0: rng.normal(0.0, 1.0, size=(200, 8))}
    after_small = {0: rng.normal(0.0, 1.0, size=(200, 8))}
    after_big = {0: rng.normal(2.5, 1.0, size=(200, 8))}

    res_small = detect_shift_from_embeddings(
        before_by_class=before, after_by_class=after_small, threshold=0.3, aggregate="mean"
    )
    assert res_small.shift_detected is False

    res_big = detect_shift_from_embeddings(
        before_by_class=before, after_by_class=after_big, threshold=0.3, aggregate="mean"
    )
    assert res_big.shift_detected is True

