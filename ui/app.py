"""Streamlit UI entrypoint.

Run:
  streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

RESULTS_PATH = "results.json"
LOG_PATH = "experiment.log"
RUNNER_MODULE = "delta_framework.experiments.run_experiment"


def _load_results(path: str = RESULTS_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_process_running(proc: Optional[subprocess.Popen]) -> bool:
    if proc is None:
        return False
    return proc.poll() is None


def _start_experiment(cmd: list[str]) -> subprocess.Popen:
    # Stream logs to a file for UI debugging.
    # Keep a handle in session_state so we can close it on stop/exit.
    log_f = open(LOG_PATH, "w", encoding="utf-8", buffering=1)
    st.session_state["log_f"] = log_f
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)


def _tail_text(path: str, max_lines: int = 200) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception:
        return ""


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_elapsed(started_at: Any) -> str:
    started = _parse_iso_timestamp(started_at)
    if started is None:
        return "n/a"
    try:
        elapsed = max(0.0, time.time() - started.timestamp())
    except Exception:
        return "n/a"
    if elapsed < 60:
        return f"{elapsed:.0f}s"
    minutes, seconds = divmod(int(elapsed), 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _phase_label(status: Dict[str, Any], total_tasks: int) -> str:
    phase = str(status.get("phase") or "").strip()
    current_task = int(status.get("current_task") or 0)
    if phase == "delta_training":
        return f"Delta training (task {current_task}/{max(1, total_tasks)})"
    if phase == "full_retrain":
        return f"Full retrain baseline (task {current_task}/{max(1, total_tasks)})"
    if phase == "setup":
        return "Preparing experiment"
    if phase == "completed":
        return "Completed"
    if phase == "failed":
        return "Failed"
    message = str(status.get("message") or "").strip()
    return message or "Waiting"


def _render_run_status(res: Dict[str, Any]) -> None:
    status = res.get("status", {}) or {}
    cfg = res.get("config", {}) or {}
    total_tasks = int(cfg.get("num_tasks", 0) or 0)
    completed = len(res.get("timeline", {}).get("tasks", []))
    cols = st.columns(4)
    cols[0].metric("Run State", str(status.get("state", "idle")).title())
    cols[1].metric("Completed Tasks", f"{completed}/{max(1, total_tasks)}")
    cols[2].metric("Current Step", _phase_label(status, total_tasks))
    cols[3].metric("Elapsed", _format_elapsed(status.get("started_at")))

    phase_elapsed = _format_elapsed(status.get("phase_started_at"))
    if phase_elapsed != "n/a":
        st.caption(f"Current step time: {phase_elapsed}")


def _render_partial_task(res: Dict[str, Any]) -> None:
    partial = res.get("partial_task") or {}
    delta = partial.get("delta", {}) or {}
    artifacts = partial.get("delta_artifacts", {}) or {}
    if not partial or not delta:
        return

    st.subheader("Current Partial Results")
    cols = st.columns(4)
    cols[0].metric("Task", str(int(partial.get("task_id", 0)) + 1))
    cols[1].metric("Seen Classes", str(int(partial.get("seen_classes", 0))))
    cols[2].metric("Delta Top-1", f"{float(delta.get('top1', 0.0)) * 100:.2f}%")
    cols[3].metric("Delta Time", f"{float(delta.get('wall_time_s', 0.0)):.1f}s")
    st.caption(
        "Delta update is finished for the current task. The app is now waiting for the "
        "full-retrain baseline to finish so it can compare both methods."
    )
    if artifacts:
        st.json(artifacts)


def _stop_experiment() -> None:
    proc = st.session_state.get("proc")
    if proc is not None and _is_process_running(proc):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    st.session_state["proc"] = None

    log_f = st.session_state.get("log_f")
    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass
    st.session_state["log_f"] = None


def _get_task_series(results: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tasks = results.get("timeline", {}).get("tasks", [])
    if not tasks:
        empty = pd.DataFrame(columns=["task", "top1", "top5", "ece", "wall_time_s"])
        return empty, empty

    rows_delta = []
    rows_full = []
    for t in tasks:
        tid = int(t.get("task_id", len(rows_delta)))
        d = t.get("delta", {}) or {}
        f = t.get("full", {}) or {}
        rows_delta.append(
            {
                "task": tid,
                "top1": float(d.get("top1", 0.0)),
                "top5": float(d.get("top5", 0.0)),
                "ece": float(d.get("ece", 0.0)),
                "wall_time_s": float(d.get("wall_time_s", 0.0)),
                "shift_score": float((t.get("shift", {}) or {}).get("shift_score", 0.0)),
                "shift_detected": bool((t.get("shift", {}) or {}).get("shift_detected", False)),
            }
        )
        rows_full.append(
            {
                "task": tid,
                "top1": float(f.get("top1", 0.0)),
                "top5": float(f.get("top5", 0.0)),
                "ece": float(f.get("ece", 0.0)),
                "wall_time_s": float(f.get("wall_time_s", 0.0)),
            }
        )

    return pd.DataFrame(rows_delta), pd.DataFrame(rows_full)


def _plot_confusion(cm: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    fig.tight_layout()
    return fig


def _badge_color(gap: float) -> str:
    if gap < 0.005:
        return "green"
    if gap < 0.01:
        return "orange"
    return "red"


def _ablation_frame(task: Dict[str, Any]) -> pd.DataFrame:
    ablations = task.get("ablations", {}) or {}
    rows = []
    for name, record in ablations.items():
        metrics = record.get("metrics", {}) or {}
        eq = record.get("equivalence", {}) or {}
        rows.append(
            {
                "variant": name,
                "top1": float(metrics.get("top1", 0.0)),
                "ece": float(metrics.get("ece", 0.0)),
                "equivalence_gap": float(eq.get("equivalence_gap", 0.0)),
                "compute_savings_percent": float(eq.get("compute_savings_percent", 0.0)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("top1", ascending=False)


def page_setup():
    st.header("Setup")

    dataset = st.selectbox("Dataset", ["CIFAR-100", "CIFAR-10", "TinyImageNet"], index=0)
    num_tasks = st.slider("Number of tasks/increments", min_value=2, max_value=10, value=5)
    classes_per_task = st.slider("Classes per task", min_value=1, max_value=50, value=20)
    old_fraction = st.slider("Old/New data split (old replay fraction)", 0.0, 0.9, value=0.2, step=0.05)
    st.caption("Tip: for a fast CPU demo, use CIFAR-10, 2 tasks, 1 epoch, and 0 data-loader workers.")

    if dataset == "TinyImageNet":
        data_path = st.text_input("TinyImageNet path (must contain train/ and val/ folders)", value="./tinyimagenet")
    else:
        data_path = st.text_input("Dataset path (downloaded if CIFAR)", value="./data")

    epochs = st.slider("Epochs per task (demo-friendly)", min_value=1, max_value=30, value=3)
    batch_size = st.select_slider("Batch size", options=[32, 64, 128, 256], value=128)
    num_workers = st.slider("Data loader workers", min_value=0, max_value=4, value=0 if os.name == "nt" else 2)

    prefer_cuda = st.checkbox("Use GPU if available", value=True)
    seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=0)
    run_ablations = st.checkbox("Run ablation baselines", value=False)
    equivalence_threshold = st.slider(
        "Equivalence gap threshold",
        min_value=0.001,
        max_value=0.05,
        value=0.005,
        step=0.001,
        format="%.3f",
    )
    policy_bound_epsilon = st.slider(
        "Policy max bound epsilon",
        min_value=0.001,
        max_value=0.05,
        value=0.01,
        step=0.001,
        format="%.3f",
    )

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        start = st.button("Start Experiment", type="primary")
    with col2:
        refresh = st.button("Refresh Results")
        _ = refresh

    stop = st.button("Stop Experiment")
    if stop:
        _stop_experiment()
        st.info("Experiment stopped.")
        return

    if start:
        if _is_process_running(st.session_state.get("proc")):
            st.warning("Experiment already running.")
            return

        cmd = [
            sys.executable,
            "-m",
            RUNNER_MODULE,
            "--dataset",
            dataset,
            "--data-path",
            data_path,
            "--num-tasks",
            str(int(num_tasks)),
            "--classes-per-task",
            str(int(classes_per_task)),
            "--old-fraction",
            str(float(old_fraction)),
            "--epochs",
            str(int(epochs)),
            "--batch-size",
            str(int(batch_size)),
            "--num-workers",
            str(int(num_workers)),
            "--seed",
            str(int(seed)),
            "--equivalence-threshold",
            str(float(equivalence_threshold)),
            "--policy-max-bound-epsilon",
            str(float(policy_bound_epsilon)),
            "--results-path",
            RESULTS_PATH,
        ]
        if prefer_cuda:
            cmd.append("--prefer-cuda")
        if run_ablations:
            cmd.append("--run-ablations")

        st.session_state["proc"] = _start_experiment(cmd)
        st.session_state["started_cmd"] = cmd
        st.success("Experiment started. Go to Live Training.")

    res = _load_results()
    if res is None:
        st.info("No `results.json` yet. Click Start Experiment to generate it.")
        return

    st.subheader("Current Status")
    _render_run_status(res)
    st.json(res.get("status", {}))

    if os.path.exists(LOG_PATH):
        with st.expander("Runner logs (tail)"):
            tail = _tail_text(LOG_PATH, max_lines=200)
            st.code(tail if tail else "(empty)")


def page_live_training():
    st.header("Live Training")

    auto = st.sidebar.checkbox("Auto-refresh (every 2s)", value=True)
    res = _load_results()
    if res is None:
        st.info("No `results.json` found yet.")
        return

    cfg = res.get("config", {})
    num_tasks = int(cfg.get("num_tasks", 1))
    tasks = res.get("timeline", {}).get("tasks", [])
    completed = len(tasks)
    status = res.get("status", {}) or {}
    phase = str(status.get("phase") or "")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Full Retrain")
        full_progress = completed / max(1, num_tasks)
        if phase == "full_retrain":
            full_progress = min(1.0, (completed + 0.5) / max(1, num_tasks))
        st.progress(min(1.0, full_progress))
    with colB:
        st.subheader("Delta Method")
        delta_progress = completed / max(1, num_tasks)
        if phase == "delta_training":
            delta_progress = min(1.0, (completed + 0.5) / max(1, num_tasks))
        elif phase == "full_retrain":
            delta_progress = min(1.0, (completed + 1.0) / max(1, num_tasks))
        st.progress(min(1.0, delta_progress))

    _render_run_status(res)
    if completed == 0 and str(status.get("state")) == "running":
        st.info(
            "The first task is still running. Results will appear after the task finishes, "
            "because each increment runs both delta training and a full-retrain baseline."
        )
    _render_partial_task(res)

    with st.expander("Runner logs (tail)"):
        tail = _tail_text(LOG_PATH, max_lines=200)
        if tail:
            st.code(tail)
        else:
            st.write("No logs yet.")

    df_delta, df_full = _get_task_series(res)
    if not df_delta.empty:
        chart = pd.DataFrame(
            {"full_top1": df_full["top1"].values, "delta_top1": df_delta["top1"].values},
            index=df_delta["task"].values,
        )
        st.line_chart(chart)

    full_time = float(df_full["wall_time_s"].sum()) if not df_full.empty else None
    delta_time = float(df_delta["wall_time_s"].sum()) if not df_delta.empty else None
    st.metric(
        "Total compute time (Full Retrain)",
        f"{full_time:.1f}s" if full_time is not None else "Pending",
    )
    st.metric(
        "Total compute time (Delta)",
        f"{delta_time:.1f}s" if delta_time is not None else "Pending",
    )

    latest_shift = None
    if tasks:
        latest_shift = tasks[-1].get("shift", {})
    if latest_shift:
        if bool(latest_shift.get("shift_detected", False)):
            st.error(
                f"Significant shift detected. KL={float(latest_shift.get('shift_score', 0.0)):.3f}"
            )
        else:
            st.success(
                f"Stable. KL={float(latest_shift.get('shift_score', 0.0)):.3f}"
            )

    latest_deployment = tasks[-1].get("deployment", {}) if tasks else {}
    if latest_deployment:
        action = latest_deployment.get("selected_source", "delta_update")
        if action == "delta_update":
            st.info("Policy decision: deploy delta update")
        else:
            st.warning("Policy decision: fall back to full retrain")
        reasons = latest_deployment.get("reasons", [])
        if reasons:
            st.caption("Policy reasons: " + ", ".join(str(reason) for reason in reasons))

    st.subheader("Raw status")
    st.json(res.get("status", {}))

    if auto and res.get("status", {}).get("state") == "running":
        time.sleep(2)
        st.rerun()


def page_results_dashboard():
    st.header("Results Dashboard")

    res = _load_results()
    if res is None:
        st.info("No `results.json` found yet.")
        return

    tasks = res.get("timeline", {}).get("tasks", [])
    if not tasks:
        _render_run_status(res)
        if str((res.get("status", {}) or {}).get("state")) == "running":
            st.info(
                "No tasks are completed yet. The current task still has to finish both the "
                "delta update and the full-retrain baseline before comparison charts appear."
            )
            _render_partial_task(res)
        elif str((res.get("status", {}) or {}).get("state")) == "failed":
            st.error(str((res.get("status", {}) or {}).get("message") or "Experiment failed."))
        else:
            st.info("No tasks completed yet.")
        return

    final = tasks[-1]
    final_full = final.get("full", {}) or {}
    final_delta = final.get("delta", {}) or {}
    final_eq = final.get("equivalence", {}) or {}
    final_deployment = final.get("deployment", {}) or {}
    final_summary = res.get("final_summary", {}) or {}
    cfg = res.get("config", {}) or {}
    gap_threshold = float(cfg.get("equivalence_threshold", 0.005))

    df_delta, df_full = _get_task_series(res)
    total_full = float(df_full["wall_time_s"].sum()) if not df_full.empty else 0.0
    total_delta = float(df_delta["wall_time_s"].sum()) if not df_delta.empty else 0.0

    compute_saved = 0.0
    if total_full > 0:
        compute_saved = ((total_full - total_delta) / total_full) * 100.0

    policy_saved = float(final_summary.get("policy_compute_savings_percent", 0.0))
    top_metrics = st.columns(4)
    top_metrics[0].metric("Delta Compute Saved", f"{compute_saved:.1f}%")
    top_metrics[1].metric("Policy Compute Saved", f"{policy_saved:.1f}%")
    top_metrics[2].metric("Policy Choice", str(final_deployment.get("selected_source", "n/a")))
    top_metrics[3].metric("Unsafe Delta Count", str(int(final_summary.get("unsafe_delta_count", 0))))

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Top-1 (Full)", f"{float(final_full.get('top1', 0.0))*100:.2f}%")
    col2.metric("Final Top-1 (Delta)", f"{float(final_delta.get('top1', 0.0))*100:.2f}%")
    gap = float(final_eq.get("equivalence_gap", abs(float(final_full.get("top1", 0.0)) - float(final_delta.get("top1", 0.0)))))
    col3.metric("Equivalence Gap", f"{gap*100:.2f}%")

    extra_metrics = st.columns(2)
    extra_metrics[0].metric(
        "Calibration Diff",
        f"{float(final_eq.get('calibration_diff', 0.0)) * 100:.2f}%",
    )
    extra_metrics[1].metric(
        "Worst Class Gap",
        f"{float(final_eq.get('worst_class_acc_gap') or 0.0) * 100:.2f}%",
    )

    st.markdown(
        f"**Equivalence badge:** :{_badge_color(gap)}[{'Equivalent' if gap < gap_threshold else 'Not equivalent'}]"
    )

    st.subheader("Accuracy over tasks")
    if not df_delta.empty:
        chart = pd.DataFrame(
            {"Full": df_full["top1"].values, "Delta": df_delta["top1"].values},
            index=df_delta["task"].values,
        )
        st.line_chart(chart)

    st.subheader("Per-class accuracy (final task)")
    per_full = np.asarray(final_full.get("per_class_acc", []), dtype=np.float32)
    per_delta = np.asarray(final_delta.get("per_class_acc", []), dtype=np.float32)
    n = int(min(len(per_full), len(per_delta)))
    if n > 0:
        df_pc = pd.DataFrame(
            {"Full": per_full[:n], "Delta": per_delta[:n]},
            index=[f"class_{i}" for i in range(n)],
        )
        st.bar_chart(df_pc)
    else:
        st.info("Per-class accuracy not available yet.")

    st.subheader("Theoretical bound (latest)")
    bound = final.get("bound", {}) or {}
    eps = float(bound.get("guaranteed_epsilon", 0.0)) * 100.0
    conf = float(bound.get("confidence_level", 0.95)) * 100.0
    st.write(f"Guaranteed within ε={eps:.2f}% with {conf:.1f}% confidence")

    if final_deployment:
        st.subheader("Deployment Decision")
        st.json(final_deployment)

    ablation_df = _ablation_frame(final)
    if not ablation_df.empty:
        st.subheader("Ablation Variants")
        st.dataframe(ablation_df, use_container_width=True)

    st.subheader("Confusion matrices (final)")
    cm_full = np.asarray(final_full.get("confusion_matrix", []))
    cm_delta = np.asarray(final_delta.get("confusion_matrix", []))
    colA, colB = st.columns(2)
    with colA:
        if cm_full.size:
            st.pyplot(_plot_confusion(cm_full, "Full Retrain"))
    with colB:
        if cm_delta.size:
            st.pyplot(_plot_confusion(cm_delta, "Delta Method"))


def page_shift_analysis():
    st.header("Shift Analysis")
    res = _load_results()
    if res is None:
        st.info("No `results.json` found yet.")
        return
    tasks = res.get("timeline", {}).get("tasks", [])
    if not tasks:
        st.info("No tasks completed yet.")
        return

    shift_scores = []
    triggers = []
    for t in tasks:
        s = t.get("shift", {}) or {}
        shift_scores.append(float(s.get("shift_score", 0.0)))
        triggers.append(bool(s.get("shift_detected", False)))

    st.subheader("KL divergence score across tasks")
    df = pd.DataFrame({"shift_score": shift_scores}, index=list(range(len(shift_scores))))
    st.line_chart(df)

    st.subheader("Tasks triggering shift detection")
    triggered = [i for i, tr in enumerate(triggers) if tr]
    st.write(triggered if triggered else "None")

    st.subheader("Per-class drift (latest task)")
    last = tasks[-1].get("shift", {}) or {}
    per = last.get("per_class_drift", {}) or {}
    if per:
        df_pc = pd.DataFrame(
            [{"class_id": k, "kl": float(v)} for k, v in per.items()]
        ).sort_values("kl", ascending=False)
        st.dataframe(df_pc, use_container_width=True)
    else:
        st.info("No per-class drift computed (likely first task).")


def main():
    st.set_page_config(page_title="Delta-Only Model Training Demo", layout="wide")
    st.title("Delta-Only Model Training: Incremental Learning Without Full Retraining")

    if "proc" not in st.session_state:
        st.session_state["proc"] = None
    if "log_f" not in st.session_state:
        st.session_state["log_f"] = None

    page = st.sidebar.radio("Pages", ["Setup", "Live Training", "Results Dashboard", "Shift Analysis"])

    if page == "Setup":
        page_setup()
    elif page == "Live Training":
        page_live_training()
    elif page == "Results Dashboard":
        page_results_dashboard()
    elif page == "Shift Analysis":
        page_shift_analysis()

    proc = st.session_state.get("proc")
    if proc is not None and not _is_process_running(proc):
        st.sidebar.info("Runner process finished.")
        log_f = st.session_state.get("log_f")
        if log_f is not None:
            try:
                log_f.close()
            except Exception:
                pass
        st.session_state["log_f"] = None


if __name__ == "__main__":
    main()

