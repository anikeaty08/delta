"""Public API for the Delta Framework library.

Pure-Python helpers stay importable without the optional ML stack. Functions
that require the training runtime load those dependencies lazily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from delta_framework.core.bounds import pac_bound
from delta_framework.core.equivalence import summarize_equivalence
from delta_framework.core.policy import DecisionPolicyConfig, decide_deployment
from delta_framework.core.shift_detector import (
    ShiftResult,
    detect_shift_for_models,
    detect_shift_from_embeddings,
)

if TYPE_CHECKING:  # pragma: no cover
    from delta_framework.core.benchmarker import BenchmarkConfig
    from delta_framework.core.trainer import TrainConfig


__all__ = [
    "BenchmarkConfig",
    "DecisionPolicyConfig",
    "TrainConfig",
    "ShiftResult",
    "build_scenarios",
    "decide_deployment",
    "detect_shift_for_models",
    "detect_shift_from_embeddings",
    "device",
    "get_device",
    "pac_bound",
    "run",
    "run_benchmark",
    "scenarios",
    "set_seed",
    "summarize_equivalence",
]


def _load_ml_runtime():
    try:
        from delta_framework.core.benchmarker import BenchmarkConfig, run_benchmark
        from delta_framework.core.trainer import TrainConfig, build_scenarios, get_device, set_seed
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "The optional ML runtime is not available. Install the project with "
            "`pip install -e .[ml]` or `pip install -e .[app]` before using training APIs."
        ) from exc

    return BenchmarkConfig, TrainConfig, run_benchmark, build_scenarios, get_device, set_seed


def __getattr__(name: str):
    if name in {"BenchmarkConfig", "TrainConfig"}:
        benchmark_config_cls, train_config_cls, *_rest = _load_ml_runtime()
        return {"BenchmarkConfig": benchmark_config_cls, "TrainConfig": train_config_cls}[name]
    raise AttributeError(name)


def run_benchmark(config: Any, *, results_path: str = "results.json") -> dict[str, Any]:
    """Run the benchmark, loading the ML runtime on demand."""

    _benchmark_config_cls, _train_config_cls, run_benchmark_fn, *_rest = _load_ml_runtime()
    return run_benchmark_fn(config, results_path=results_path)


def build_scenarios(
    *,
    dataset: str,
    data_path: str,
    classes_per_task: int,
    seed: int,
    num_tasks: int,
    input_size: int = 32,
    class_order: Optional[Sequence[int]] = None,
) -> tuple[Any, Any, int, list[int]]:
    """Load incremental-learning scenarios using the optional ML runtime."""

    _benchmark_config_cls, _train_config_cls, _run_benchmark_fn, build_scenarios_fn, *_rest = (
        _load_ml_runtime()
    )
    return build_scenarios_fn(
        dataset=dataset,
        data_path=data_path,
        classes_per_task=classes_per_task,
        seed=seed,
        num_tasks=num_tasks,
        input_size=input_size,
        class_order=class_order,
    )


def get_device(prefer_cuda: bool = True) -> Any:
    """Return the preferred training device using the optional ML runtime."""

    _benchmark_config_cls, _train_config_cls, _run_benchmark_fn, _build_scenarios_fn, get_device_fn, _set_seed = (
        _load_ml_runtime()
    )
    return get_device_fn(prefer_cuda=prefer_cuda)


def set_seed(seed: int) -> None:
    """Set NumPy / Torch random seeds using the optional ML runtime."""

    _benchmark_config_cls, _train_config_cls, _run_benchmark_fn, _build_scenarios_fn, _get_device_fn, set_seed_fn = (
        _load_ml_runtime()
    )
    set_seed_fn(seed)


def run(config: Any, *, results_path: str = "results.json") -> dict[str, Any]:
    """Convenience alias for :func:`run_benchmark`."""

    return run_benchmark(config, results_path=results_path)


def device(prefer_cuda: bool = True) -> Any:
    """Convenience alias for :func:`get_device`."""

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
) -> tuple[Any, Any, int, list[int]]:
    """Convenience alias for :func:`build_scenarios`."""

    return build_scenarios(
        dataset=dataset,
        data_path=data_path,
        classes_per_task=classes_per_task,
        seed=seed,
        num_tasks=num_tasks,
        input_size=input_size,
        class_order=class_order,
    )
