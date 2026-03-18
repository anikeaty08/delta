import pytest

from delta_framework import api


def test_api_imports_without_optional_ml_stack():
    assert api.pac_bound(N_old=0, N_new=10, delta=0.1)["guaranteed_epsilon"] > 0.0
    decision = api.decide_deployment(
        shift_detected=False,
        shift_score=0.0,
        bound_epsilon=0.001,
        policy=api.DecisionPolicyConfig(),
        equivalence_gap=0.0,
    )
    assert decision.recommended_action == "delta_update"


def test_api_ml_runtime_functions_raise_without_optional_stack(monkeypatch):
    def _raise_import_error():
        raise ImportError("optional ML runtime missing")

    monkeypatch.setattr(api, "_load_ml_runtime", _raise_import_error)
    with pytest.raises(ImportError, match="optional ML runtime missing"):
        api.get_device()
