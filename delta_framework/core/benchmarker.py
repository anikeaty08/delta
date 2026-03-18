"""Benchmark runner orchestration (delta-only vs full retrain)."""

from __future__ import annotations

import copy
import json
import logging
import os
import traceback
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from . import bounds, equivalence, policy, shift_detector, trainer

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str = "CIFAR-100"
    data_path: str = "./data"
    num_tasks: int = 5
    classes_per_task: int = 20
    input_size: int = 32
    seed: int = 0
    prefer_cuda: bool = True

    # Shift detection
    shift_threshold: float = 0.3
    shift_aggregate: str = "mean"  # mean|max
    shift_max_per_class: int = 256

    # Bound
    bound_delta: float = 0.05
    bound_correction_term: float = 0.0
    equivalence_threshold: float = 0.005
    policy_max_bound_epsilon: float = 0.01
    run_ablations: bool = False
    ablation_variants: tuple[str, ...] = ("naive_new_data", "replay_only", "replay_kd")

    # Training
    train: trainer.TrainConfig = field(default_factory=trainer.TrainConfig)


def _build_ablation_configs(
    train_config: trainer.TrainConfig,
    variants: Sequence[str],
) -> Dict[str, trainer.TrainConfig]:
    known = {
        "naive_new_data": replace(
            train_config,
            use_replay=False,
            use_kd=False,
            use_weight_align=False,
        ),
        "replay_only": replace(
            train_config,
            use_replay=True,
            use_kd=False,
            use_weight_align=False,
        ),
        "replay_kd": replace(
            train_config,
            use_replay=True,
            use_kd=True,
            use_weight_align=False,
        ),
    }

    out: Dict[str, trainer.TrainConfig] = {}
    for name in variants:
        if name not in known:
            raise ValueError(f"Unknown ablation variant: {name}")
        out[name] = known[name]
    return out


def _copy_model_if_present(
    model: Optional[trainer.CilModel],
    *,
    device: Any,
) -> Optional[trainer.CilModel]:
    if model is None:
        return None
    return model.copy().to(device)


def _init_results(config: BenchmarkConfig) -> Dict[str, Any]:
    return {
        "config": {
            "dataset": config.dataset,
            "data_path": config.data_path,
            "num_tasks": int(config.num_tasks),
            "classes_per_task": int(config.classes_per_task),
            "input_size": int(config.input_size),
            "seed": int(config.seed),
            "prefer_cuda": bool(config.prefer_cuda),
            "train": asdict(config.train),
            "shift_threshold": float(config.shift_threshold),
            "bound_delta": float(config.bound_delta),
            "equivalence_threshold": float(config.equivalence_threshold),
            "policy_max_bound_epsilon": float(config.policy_max_bound_epsilon),
            "run_ablations": bool(config.run_ablations),
            "ablation_variants": list(config.ablation_variants),
        },
        "status": {
            "state": "idle",
            "current_task": 0,
            "message": "",
            "started_at": None,
            "updated_at": _now_iso(),
        },
        "timeline": {"tasks": []},
        "final_summary": None,
    }


def run_benchmark(config: BenchmarkConfig, *, results_path: str = "results.json") -> Dict[str, Any]:
    """Run delta-only vs full-retrain per task and write incremental JSON results."""
    results = _init_results(config)
    results["status"] = {
        "state": "running",
        "current_task": 0,
        "message": "Starting experiment",
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    _atomic_write_json(results_path, results)

    trainer.set_seed(config.seed)
    device = trainer.get_device(prefer_cuda=config.prefer_cuda)
    logger.info(
        "Starting benchmark dataset=%s tasks=%s classes_per_task=%s device=%s results_path=%s",
        config.dataset,
        config.num_tasks,
        config.classes_per_task,
        device,
        results_path,
    )

    scenario_train, scenario_val, nb_classes, class_order_used = trainer.build_scenarios(
        dataset=config.dataset,
        data_path=config.data_path,
        classes_per_task=config.classes_per_task,
        seed=config.seed,
        num_tasks=config.num_tasks,
        input_size=config.input_size,
        class_order=None,
    )
    logger.debug(
        "Built scenarios with total_classes=%s class_order_length=%s",
        nb_classes,
        len(class_order_used),
    )

    # Delta-only state
    delta_model = trainer.CilModel(config.train.backbone, device=device).to(device)
    teacher_model: Optional[trainer.CilModel] = None
    known_classes = 0

    memory = trainer.rehearsal.RehearsalMemory(
        memory_size=config.train.memory_size,
        herding_method=config.train.herding_method,
        fixed_memory=config.train.fixed_memory,
    )

    # For bound calculation
    n_old_seen = 0

    try:
        for task_id, dataset_train in enumerate(scenario_train):
            if task_id >= config.num_tasks:
                break

            nb_new = min(config.classes_per_task, nb_classes - known_classes)
            dataset_val = scenario_val[: task_id + 1]
            logger.info(
                "Task %s/%s started known_classes=%s new_classes=%s train_samples=%s",
                task_id + 1,
                config.num_tasks,
                known_classes,
                nb_new,
                len(dataset_train),
            )

            results["status"] = {
                "state": "running",
                "current_task": int(task_id),
                "message": f"Training task {task_id + 1}/{config.num_tasks}",
                "started_at": results["status"]["started_at"],
                "updated_at": _now_iso(),
            }
            _atomic_write_json(results_path, results)

            # SHIFT (old classes) evaluated on previous-task validation data.
            before_model = teacher_model
            ablation_state = None
            if config.run_ablations:
                ablation_state = (
                    delta_model.copy().to(device),
                    _copy_model_if_present(teacher_model, device=device),
                    copy.deepcopy(memory),
                )

            # Delta-only incremental update.
            trainer.reset_peak_memory_stats(device)
            t0 = trainer.time.perf_counter()
            delta_out = trainer.train_one_task_delta(
                model=delta_model,
                teacher_model=teacher_model,
                memory=memory,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                task_id=task_id,
                nb_new_classes=nb_new,
                known_classes=known_classes,
                device=device,
                config=config.train,
                seed=config.seed,
            )
            delta_wall_s = trainer.time.perf_counter() - t0
            delta_peak_mb = trainer.measure_peak_memory_mb(device)
            delta_model, teacher_model, delta_metrics, _delta_artifacts = delta_out
            delta_metrics["wall_time_s"] = float(delta_wall_s)
            delta_metrics["peak_mem_mb"] = float(delta_peak_mb)

            # Full retrain baseline on all seen data.
            dataset_train_full = scenario_train[: task_id + 1]
            trainer.reset_peak_memory_stats(device)
            t1 = trainer.time.perf_counter()
            full_out = trainer.train_one_task_full_retrain(
                backbone=config.train.backbone,
                dataset_train_full=dataset_train_full,
                dataset_val=dataset_val,
                task_id=task_id,
                classes_per_task=config.classes_per_task,
                total_tasks_seen=task_id + 1,
                device=device,
                config=config.train,
                seed=config.seed,
            )
            full_wall_s = trainer.time.perf_counter() - t1
            full_peak_mb = trainer.measure_peak_memory_mb(device)
            full_model, full_metrics = full_out
            full_metrics["wall_time_s"] = float(full_wall_s)
            full_metrics["peak_mem_mb"] = float(full_peak_mb)

            # Equivalence summary.
            eq = equivalence.summarize_equivalence(
                full_metrics=full_metrics,
                delta_metrics=delta_metrics,
                timing_full_s=full_wall_s,
                timing_delta_s=delta_wall_s,
                mem_full_mb=full_peak_mb,
                mem_delta_mb=delta_peak_mb,
                equivalence_threshold=config.equivalence_threshold,
            )

            # Shift detection (representation drift for old classes).
            if before_model is None or known_classes == 0:
                shift = shift_detector.ShiftResult(0.0, False, {})
            else:
                old_val = scenario_val[:task_id]
                old_class_ids = list(range(known_classes))
                shift = shift_detector.detect_shift_for_models(
                    before_model=before_model,
                    after_model=delta_model,
                    dataset=old_val,
                    device=device,
                    class_ids=old_class_ids,
                    batch_size=config.train.batch_size,
                    num_workers=config.train.num_workers,
                    max_per_class=config.shift_max_per_class,
                    threshold=config.shift_threshold,
                    aggregate=config.shift_aggregate,
                )

            # Theoretical bound.
            n_new = len(dataset_train)
            bound = bounds.pac_bound(
                N_old=int(n_old_seen),
                N_new=int(n_new),
                delta=float(config.bound_delta),
                correction_term=float(config.bound_correction_term),
            )
            n_old_seen += int(n_new)

            deployment_decision = policy.decide_deployment(
                shift_detected=bool(shift.shift_detected),
                shift_score=float(shift.shift_score),
                bound_epsilon=float(bound.get("guaranteed_epsilon", 0.0)),
                policy=policy.DecisionPolicyConfig(
                    equivalence_threshold=config.equivalence_threshold,
                    max_bound_epsilon=config.policy_max_bound_epsilon,
                ),
                equivalence_gap=float(eq.get("equivalence_gap", 0.0)),
            )
            selected_source = deployment_decision.recommended_action
            selected_metrics = delta_metrics if selected_source == "delta_update" else full_metrics
            selected_time_s = delta_wall_s if selected_source == "delta_update" else full_wall_s
            selected_peak_mb = delta_peak_mb if selected_source == "delta_update" else full_peak_mb

            ablations: Dict[str, Any] = {}
            if ablation_state is not None:
                ablation_model, ablation_teacher, ablation_memory = ablation_state
                for name, ablation_config in _build_ablation_configs(
                    config.train,
                    config.ablation_variants,
                ).items():
                    trainer.reset_peak_memory_stats(device)
                    t_ab = trainer.time.perf_counter()
                    ablation_out = trainer.train_one_task_delta(
                        model=ablation_model.copy().to(device),
                        teacher_model=_copy_model_if_present(ablation_teacher, device=device),
                        memory=copy.deepcopy(ablation_memory),
                        dataset_train=dataset_train,
                        dataset_val=dataset_val,
                        task_id=task_id,
                        nb_new_classes=nb_new,
                        known_classes=known_classes,
                        device=device,
                        config=ablation_config,
                        seed=config.seed,
                    )
                    ablation_wall_s = trainer.time.perf_counter() - t_ab
                    ablation_peak_mb = trainer.measure_peak_memory_mb(device)
                    _ab_model, _ab_teacher, ablation_metrics, ablation_artifacts = ablation_out
                    ablation_metrics["wall_time_s"] = float(ablation_wall_s)
                    ablation_metrics["peak_mem_mb"] = float(ablation_peak_mb)
                    ablation_eq = equivalence.summarize_equivalence(
                        full_metrics=full_metrics,
                        delta_metrics=ablation_metrics,
                        timing_full_s=full_wall_s,
                        timing_delta_s=ablation_wall_s,
                        mem_full_mb=full_peak_mb,
                        mem_delta_mb=ablation_peak_mb,
                        equivalence_threshold=config.equivalence_threshold,
                    )
                    ablations[name] = {
                        "metrics": ablation_metrics,
                        "equivalence": ablation_eq,
                        "artifacts": ablation_artifacts,
                        "train": asdict(ablation_config),
                    }

            task_record = {
                "task_id": int(task_id),
                "seen_classes": int(known_classes + nb_new),
                "delta": delta_metrics,
                "delta_artifacts": _delta_artifacts,
                "full": full_metrics,
                "equivalence": eq,
                "shift": {
                    "shift_score": float(shift.shift_score),
                    "shift_detected": bool(shift.shift_detected),
                    "per_class_drift": {str(k): float(v) for k, v in shift.per_class_drift.items()},
                },
                "bound": bound,
                "deployment": {
                    **deployment_decision.to_dict(),
                    "selected_source": selected_source,
                    "selected_top1": float(selected_metrics.get("top1", 0.0)),
                    "selected_ece": float(selected_metrics.get("ece", 0.0)),
                    "selected_wall_time_s": float(selected_time_s),
                    "selected_peak_mem_mb": float(selected_peak_mb),
                },
                "ablations": ablations,
            }

            results["timeline"]["tasks"].append(task_record)
            results["status"]["updated_at"] = _now_iso()
            _atomic_write_json(results_path, results)
            logger.info(
                "Task %s/%s finished delta_top1=%.4f full_top1=%.4f gap=%.4f "
                "delta_time=%.2fs full_time=%.2fs shift=%.4f decision=%s",
                task_id + 1,
                config.num_tasks,
                float(delta_metrics.get("top1", 0.0)),
                float(full_metrics.get("top1", 0.0)),
                float(eq.get("equivalence_gap", 0.0)),
                delta_wall_s,
                full_wall_s,
                float(shift.shift_score),
                selected_source,
            )

            known_classes += nb_new

        # Final summary.
        tasks = results["timeline"]["tasks"]
        if tasks:
            final_full = tasks[-1]["full"]
            final_delta = tasks[-1]["delta"]
            final_deployment = tasks[-1].get("deployment", {}) or {}

            total_full_time = float(sum(t["full"]["wall_time_s"] for t in tasks))
            total_delta_time = float(sum(t["delta"]["wall_time_s"] for t in tasks))
            total_policy_time = float(
                sum((t.get("deployment", {}) or {}).get("selected_wall_time_s", 0.0) for t in tasks)
            )
            compute_saved = equivalence.compute_savings_percent(total_full_time, total_delta_time)
            policy_saved = equivalence.compute_savings_percent(total_full_time, total_policy_time)
            eq_gap = abs(float(final_full.get("top1", 0.0)) - float(final_delta.get("top1", 0.0)))
            delta_task_count = sum(
                1
                for t in tasks
                if (t.get("deployment", {}) or {}).get("selected_source") == "delta_update"
            )
            retrain_task_count = len(tasks) - delta_task_count
            conservative_retrain_count = sum(
                1
                for t in tasks
                if bool((t.get("deployment", {}) or {}).get("conservative_retrain", False))
            )
            unsafe_delta_count = sum(
                1
                for t in tasks
                if bool((t.get("deployment", {}) or {}).get("unsafe_delta", False))
            )

            results["final_summary"] = {
                "compute_savings_percent": float(compute_saved),
                "policy_compute_savings_percent": float(policy_saved),
                "final_top1_full": float(final_full.get("top1", 0.0)),
                "final_top1_delta": float(final_delta.get("top1", 0.0)),
                "final_equivalence_gap": float(eq_gap),
                "final_selected_source": final_deployment.get("selected_source"),
                "final_selected_top1": float(final_deployment.get("selected_top1", 0.0)),
                "delta_update_tasks": int(delta_task_count),
                "full_retrain_tasks": int(retrain_task_count),
                "conservative_retrains": int(conservative_retrain_count),
                "unsafe_delta_count": int(unsafe_delta_count),
            }
            final_ablations = tasks[-1].get("ablations", {}) or {}
            if final_ablations:
                results["final_summary"]["ablation_summary"] = {
                    name: {
                        "top1": float((record.get("metrics", {}) or {}).get("top1", 0.0)),
                        "equivalence_gap": float(
                            (record.get("equivalence", {}) or {}).get("equivalence_gap", 0.0)
                        ),
                        "compute_savings_percent": float(
                            (record.get("equivalence", {}) or {}).get("compute_savings_percent", 0.0)
                        ),
                    }
                    for name, record in final_ablations.items()
                }
            logger.info(
                "Benchmark completed compute_saved=%.2f%% policy_saved=%.2f%% "
                "final_top1_full=%.4f final_top1_delta=%.4f final_gap=%.4f",
                compute_saved,
                policy_saved,
                float(final_full.get("top1", 0.0)),
                float(final_delta.get("top1", 0.0)),
                eq_gap,
            )

        results["status"] = {
            "state": "completed",
            "current_task": int(config.num_tasks),
            "message": "Experiment completed",
            "started_at": results["status"]["started_at"],
            "updated_at": _now_iso(),
        }
        _atomic_write_json(results_path, results)
        return results

    except Exception as e:
        results["status"] = {
            "state": "failed",
            "current_task": int(results["status"].get("current_task", 0)),
            "message": f"Experiment failed: {type(e).__name__}: {e}",
            "started_at": results["status"]["started_at"],
            "updated_at": _now_iso(),
        }
        results["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        _atomic_write_json(results_path, results)
        logger.exception("Benchmark failed")
        raise


