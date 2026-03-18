"""Decision-policy helpers for choosing delta updates vs full retraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DecisionPolicyConfig:
    """Thresholds used to decide whether a delta update is safe to deploy."""

    equivalence_threshold: float = 0.005
    max_bound_epsilon: float = 0.01


@dataclass(frozen=True)
class DeploymentDecision:
    """Policy decision and hindsight label for a benchmarked task."""

    recommended_action: str
    reasons: tuple[str, ...]
    shift_detected: bool
    shift_score: float
    bound_epsilon: float
    equivalence_gap: Optional[float]
    hindsight_equivalent: Optional[bool]
    conservative_retrain: bool
    unsafe_delta: bool

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["reasons"] = list(self.reasons)
        return out


def decide_deployment(
    *,
    shift_detected: bool,
    shift_score: float,
    bound_epsilon: float,
    policy: DecisionPolicyConfig,
    equivalence_gap: Optional[float] = None,
) -> DeploymentDecision:
    """Choose whether to deploy the delta update or fall back to full retraining."""

    reasons: list[str] = []
    if shift_detected:
        reasons.append("distribution_shift_detected")
    if float(bound_epsilon) > float(policy.max_bound_epsilon):
        reasons.append("bound_exceeds_threshold")

    recommended_action = "delta_update" if not reasons else "full_retrain"

    hindsight_equivalent: Optional[bool]
    if equivalence_gap is None:
        hindsight_equivalent = None
    else:
        hindsight_equivalent = float(equivalence_gap) <= float(policy.equivalence_threshold)

    conservative_retrain = bool(
        recommended_action == "full_retrain" and hindsight_equivalent is True
    )
    unsafe_delta = bool(recommended_action == "delta_update" and hindsight_equivalent is False)

    return DeploymentDecision(
        recommended_action=recommended_action,
        reasons=tuple(reasons),
        shift_detected=bool(shift_detected),
        shift_score=float(shift_score),
        bound_epsilon=float(bound_epsilon),
        equivalence_gap=None if equivalence_gap is None else float(equivalence_gap),
        hindsight_equivalent=hindsight_equivalent,
        conservative_retrain=conservative_retrain,
        unsafe_delta=unsafe_delta,
    )
