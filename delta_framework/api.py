"""Public, stable-ish API for the Delta Framework library.

The `core/` modules contain the implementation details. This module provides a
small surface area that downstream code can import without depending on internal
file layout.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from delta_framework.core.benchmarker import BenchmarkConfig, run_benchmark
from delta_framework.core.bounds import pac_bound
from delta_framework.core.equivalence import summarize_equivalence
from delta_framework.core.policy import DecisionPolicyConfig, decide_deployment
from delta_framework.core.shift_detector import (
    ShiftResult,
    detect_shift_for_models,
    detect_shift_from_embeddings,
)
from delta_framework.core.trainer import TrainConfig, build_scenarios, get_device, set_seed

__all__ = [
    "BenchmarkConfig",
    "DecisionPolicyConfig",
    "TrainConfig",
    "ShiftResult",
    "build_scenarios",
    "decide_deployment",
    "detect_shift_for_models",
    "detect_shift_from_embeddings",
    "get_device",
    "pac_bound",
    "run_benchmark",
    "set_seed",
    "summarize_equivalence",
]


def run(
    config: BenchmarkConfig,
    *,
    results_path: str = "results.json",
) -> Dict[str, Any]:
    """Convenience wrapper around `run_benchmark`."""

    return run_benchmark(config, results_path=results_path)


def device(prefer_cuda: bool = True) -> torch.device:
    """Convenience wrapper around `get_device`."""

    return get_device(prefer_cuda=prefer_cuda)


def scenarios(
    *,
    dataset: str,
    data_path: str,
    classes_per_task: int,
    seed: int,
    num_tasks: int,
    input_size: int = 32,
    class_order: Optional[Sequence[int]] = None,
) -> Tuple[Any, Any, int, list[int]]:
    """Convenience wrapper around `build_scenarios`."""

    return build_scenarios(
        dataset=dataset,
        data_path=data_path,
        classes_per_task=classes_per_task,
        seed=seed,
        num_tasks=num_tasks,
        input_size=input_size,
        class_order=class_order,
    )
