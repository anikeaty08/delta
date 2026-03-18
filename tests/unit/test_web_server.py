from pathlib import Path

from delta_framework.web.server import ExperimentManager, WebAppPaths, build_run_command


def test_build_run_command_respects_flags(tmp_path):
    cmd = build_run_command(
        {
            "dataset": "CIFAR-10",
            "num_tasks": 2,
            "classes_per_task": 5,
            "epochs": 1,
            "batch_size": 64,
            "num_workers": 0,
            "prefer_cuda": False,
            "use_replay": False,
            "use_kd": False,
            "use_weight_align": False,
            "run_ablations": True,
            "ablation_variants": ["naive_new_data", "replay_only"],
        },
        results_path=tmp_path / "results.json",
    )

    assert "--dataset" in cmd
    assert "CIFAR-10" in cmd
    assert "--disable-replay" in cmd
    assert "--disable-kd" in cmd
    assert "--disable-weight-align" in cmd
    assert "--run-ablations" in cmd
    assert "--prefer-cuda" not in cmd


def test_experiment_manager_reads_existing_state(tmp_path):
    results_path = tmp_path / "results.json"
    log_path = tmp_path / "experiment.log"
    results_path.write_text('{"status":{"state":"running"},"timeline":{"tasks":[]}}', encoding="utf-8")
    log_path.write_text("line 1\nline 2\n", encoding="utf-8")

    manager = ExperimentManager(
        WebAppPaths(
            workdir=tmp_path,
            results_path=results_path,
            log_path=log_path,
            static_dir=Path(__file__).resolve().parent,
        )
    )

    state = manager.read_state()
    assert state["running"] is False
    assert state["results"]["status"]["state"] == "running"
    assert "line 2" in state["logs"]
