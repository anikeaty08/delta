import math

from delta_framework.core.bounds import pac_bound


def test_pac_bound_zero_new_data_defaults_to_one_plus_correction():
    out = pac_bound(N_old=100, N_new=0, delta=0.05, correction_term=0.25)
    assert math.isclose(out["guaranteed_epsilon"], 1.25, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(out["confidence_level"], 0.95, rel_tol=0.0, abs_tol=1e-12)


def test_pac_bound_decreases_with_more_new_samples():
    small = pac_bound(N_old=0, N_new=10, delta=0.1, correction_term=0.0)["guaranteed_epsilon"]
    large = pac_bound(N_old=0, N_new=10_000, delta=0.1, correction_term=0.0)["guaranteed_epsilon"]
    assert large < small


def test_pac_bound_clamps_delta_into_valid_range():
    out_lo = pac_bound(N_old=0, N_new=100, delta=0.0, correction_term=0.0)
    out_hi = pac_bound(N_old=0, N_new=100, delta=2.0, correction_term=0.0)
    assert 0.0 < out_lo["confidence_level"] < 1.0
    assert 0.0 < out_hi["confidence_level"] < 1.0

