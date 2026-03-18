import json
from dataclasses import dataclass, field

import pytest

from delta_framework.experiments import run_experiment


@dataclass(frozen=True)
class FakeTrainConfig:
    backbone: str = "resnet32"
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 3
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lambda_kd: float = 0.5
    kd_temperature: float = 2.0
    old_fraction: float = 0.2
    memory_size: int = 2000
    herding_method: str = "barycenter"
    fixed_memory: bool = False
    use_replay: bool = True
    use_kd: bool = True
    use_weight_align: bool = True


@dataclass(frozen=True)
class FakeBenchmarkConfig:
    dataset: str = "CIFAR-100"
    data_path: str = "./data"
    num_tasks: int = 5
    classes_per_task: int = 20
    input_size: int = 32
    seed: int = 0
    prefer_cuda: bool = False
    shift_threshold: float = 0.3
    shift_aggregate: str = "mean"
    shift_max_per_class: int = 256
    bound_delta: float = 0.05
    bound_correction_term: float = 0.0
    equivalence_threshold: float = 0.005
    policy_max_bound_epsilon: float = 0.01
    run_ablations: bool = False
    ablation_variants: tuple[str, ...] = ("naive_new_data", "replay_only", "replay_kd")
    train: FakeTrainConfig = field(default_factory=FakeTrainConfig)


@pytest.mark.integration
def test_cli_dump_and_config_run_smoke(tmp_path, monkeypatch):
    calls = {}

    def fake_run_benchmark(cfg, *, results_path):
        calls["cfg"] = cfg
        calls["results_path"] = results_path
        return {"status": {"state": "completed"}}

    monkeypatch.setattr(
        run_experiment,
        "_load_runtime_dependencies",
        lambda: (FakeBenchmarkConfig, FakeTrainConfig, fake_run_benchmark),
    )

    config_path = tmp_path / "config.json"
    results_path = tmp_path / "results.json"

    run_experiment.main(
        [
            "--dataset",
            "CIFAR-10",
            "--num-tasks",
            "2",
            "--classes-per-task",
            "5",
            "--run-ablations",
            "--equivalence-threshold",
            "0.01",
            "--dump-config",
            str(config_path),
        ]
    )

    dumped = json.loads(config_path.read_text(encoding="utf-8"))
    assert dumped["dataset"] == "CIFAR-10"
    assert dumped["num_tasks"] == 2
    assert dumped["classes_per_task"] == 5
    assert dumped["run_ablations"] is True
    assert dumped["equivalence_threshold"] == 0.01

    run_experiment.main(
        [
            "--config",
            str(config_path),
            "--dataset",
            "TinyImageNet",
            "--results-path",
            str(results_path),
            "--log-level",
            "DEBUG",
        ]
    )

    assert calls["results_path"] == str(results_path)
    assert isinstance(calls["cfg"], FakeBenchmarkConfig)
    assert calls["cfg"].dataset == "CIFAR-10"
    assert calls["cfg"].num_tasks == 2
    assert calls["cfg"].run_ablations is True
