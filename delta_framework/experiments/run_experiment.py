"""CLI entrypoint for running a benchmark experiment.

Used by the Streamlit UI via a subprocess.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _load_runtime_dependencies() -> Tuple[type, type, Callable[..., dict]]:
    try:
        from delta_framework.core.benchmarker import BenchmarkConfig, run_benchmark
        from delta_framework.core.trainer import TrainConfig
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "The optional ML runtime is not available. Install the project with "
            "`pip install -e .[ml]` or `pip install -e .[app]` before running the benchmark CLI."
        ) from exc

    return BenchmarkConfig, TrainConfig, run_benchmark


def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Config JSON must be an object/dict at top-level.")
    return obj


def _benchmark_config_from_dict(obj: dict, benchmark_config_cls: type, train_config_cls: type) -> Any:
    train_obj = obj.get("train", {}) or {}
    if not isinstance(train_obj, dict):
        raise ValueError("`train` must be an object/dict.")

    train_cfg = train_config_cls(**train_obj)
    cfg_fields = dict(obj)
    cfg_fields["train"] = train_cfg
    if isinstance(cfg_fields.get("ablation_variants"), list):
        cfg_fields["ablation_variants"] = tuple(cfg_fields["ablation_variants"])
    return benchmark_config_cls(**cfg_fields)


def _dump_config(path: str, cfg: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Delta-only vs full-retrain benchmark runner")

    p.add_argument(
        "--config",
        default=None,
        help="Path to a JSON config file. If provided, CLI hyperparameter flags are ignored.",
    )
    p.add_argument(
        "--dump-config",
        default=None,
        help="Write the resolved config JSON to this path and exit.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity.",
    )

    p.add_argument(
        "--dataset",
        default="CIFAR-100",
        choices=["CIFAR-100", "CIFAR-10", "TinyImageNet"],
        help="Dataset name. TinyImageNet requires a local ImageFolder-style path.",
    )
    p.add_argument("--data-path", default="./data", help="Dataset root path.")
    p.add_argument("--num-tasks", type=int, default=5)
    p.add_argument("--classes-per-task", type=int, default=20)
    p.add_argument("--old-fraction", type=float, default=0.2, help="Replay fraction for delta training mix.")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefer-cuda", action="store_true")

    p.add_argument("--backbone", default="resnet32", choices=["resnet20", "resnet32", "resnet44", "resnet56"])
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)

    p.add_argument("--lambda-kd", type=float, default=0.5)
    p.add_argument("--kd-temperature", type=float, default=2.0)

    p.add_argument("--memory-size", type=int, default=2000)
    p.add_argument("--herding-method", default="barycenter")
    p.add_argument("--fixed-memory", action="store_true")
    p.add_argument("--disable-replay", action="store_true")
    p.add_argument("--disable-kd", action="store_true")
    p.add_argument("--disable-weight-align", action="store_true")

    p.add_argument("--shift-threshold", type=float, default=0.3)
    p.add_argument("--equivalence-threshold", type=float, default=0.005)
    p.add_argument("--policy-max-bound-epsilon", type=float, default=0.01)
    p.add_argument("--run-ablations", action="store_true")
    p.add_argument(
        "--ablation-variants",
        nargs="*",
        default=["naive_new_data", "replay_only", "replay_kd"],
        help="Optional ablation variants to evaluate alongside the main method.",
    )

    p.add_argument("--results-path", default="results.json")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    benchmark_config_cls, train_config_cls, run_benchmark = _load_runtime_dependencies()

    if args.config is not None:
        cfg = _benchmark_config_from_dict(
            _load_json(args.config),
            benchmark_config_cls,
            train_config_cls,
        )
        logger.info("Loaded config from %s", args.config)
        if args.dump_config is not None:
            _dump_config(args.dump_config, cfg)
            logger.info("Wrote config to %s", args.dump_config)
            return
        run_benchmark(cfg, results_path=args.results_path)
        return

    train_cfg = train_config_cls(
        backbone=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lambda_kd=args.lambda_kd,
        kd_temperature=args.kd_temperature,
        old_fraction=args.old_fraction,
        memory_size=args.memory_size,
        herding_method=args.herding_method,
        fixed_memory=args.fixed_memory,
        use_replay=not bool(args.disable_replay),
        use_kd=not bool(args.disable_kd),
        use_weight_align=not bool(args.disable_weight_align),
    )

    cfg = benchmark_config_cls(
        dataset=args.dataset,
        data_path=args.data_path,
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        seed=args.seed,
        prefer_cuda=bool(args.prefer_cuda),
        train=train_cfg,
        shift_threshold=args.shift_threshold,
        equivalence_threshold=args.equivalence_threshold,
        policy_max_bound_epsilon=args.policy_max_bound_epsilon,
        run_ablations=bool(args.run_ablations),
        ablation_variants=tuple(args.ablation_variants),
    )

    if args.dump_config is not None:
        _dump_config(args.dump_config, cfg)
        logger.info("Wrote config to %s", args.dump_config)
        return

    run_benchmark(cfg, results_path=args.results_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        raise SystemExit(1) from None

