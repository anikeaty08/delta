"""CLI entrypoint for running a benchmark experiment.

Used by the Streamlit UI via a subprocess.
"""

from __future__ import annotations

import argparse
import sys

from delta_framework.core.benchmarker import BenchmarkConfig, run_benchmark
from delta_framework.core.trainer import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Delta-only vs full-retrain benchmark runner")

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

    p.add_argument("--shift-threshold", type=float, default=0.3)

    p.add_argument("--results-path", default="results.json")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    train_cfg = TrainConfig(
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
    )

    cfg = BenchmarkConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        seed=args.seed,
        prefer_cuda=bool(args.prefer_cuda),
        train=train_cfg,
        shift_threshold=args.shift_threshold,
    )

    run_benchmark(cfg, results_path=args.results_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        raise SystemExit(1)

