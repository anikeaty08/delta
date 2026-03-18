from delta_framework.core.policy import DecisionPolicyConfig, decide_deployment


def test_policy_accepts_delta_when_shift_and_bound_are_safe():
    decision = decide_deployment(
        shift_detected=False,
        shift_score=0.1,
        bound_epsilon=0.003,
        policy=DecisionPolicyConfig(equivalence_threshold=0.01, max_bound_epsilon=0.01),
        equivalence_gap=0.004,
    )
    assert decision.recommended_action == "delta_update"
    assert decision.hindsight_equivalent is True
    assert decision.unsafe_delta is False
    assert decision.conservative_retrain is False


def test_policy_falls_back_to_retrain_when_shift_detected():
    decision = decide_deployment(
        shift_detected=True,
        shift_score=0.9,
        bound_epsilon=0.003,
        policy=DecisionPolicyConfig(equivalence_threshold=0.01, max_bound_epsilon=0.01),
        equivalence_gap=0.004,
    )
    assert decision.recommended_action == "full_retrain"
    assert "distribution_shift_detected" in decision.reasons
    assert decision.conservative_retrain is True


def test_policy_marks_unsafe_delta_in_hindsight():
    decision = decide_deployment(
        shift_detected=False,
        shift_score=0.1,
        bound_epsilon=0.003,
        policy=DecisionPolicyConfig(equivalence_threshold=0.01, max_bound_epsilon=0.01),
        equivalence_gap=0.05,
    )
    assert decision.recommended_action == "delta_update"
    assert decision.hindsight_equivalent is False
    assert decision.unsafe_delta is True
