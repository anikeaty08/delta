const POLL_INTERVAL_MS = 2500;

const PRESETS = {
  "fast-demo": {
    dataset: "CIFAR-10",
    data_path: "./data",
    num_tasks: 2,
    classes_per_task: 5,
    epochs: 1,
    batch_size: 64,
    num_workers: 0,
    seed: 0,
    old_fraction: 0.15,
    shift_threshold: 0.3,
    equivalence_threshold: 0.01,
    policy_max_bound_epsilon: 0.02,
    prefer_cuda: false,
    use_replay: true,
    use_kd: true,
    use_weight_align: true,
    run_ablations: false,
    fixed_memory: false,
  },
  balanced: {
    dataset: "CIFAR-100",
    data_path: "./data",
    num_tasks: 5,
    classes_per_task: 20,
    epochs: 2,
    batch_size: 128,
    num_workers: 0,
    seed: 0,
    old_fraction: 0.2,
    shift_threshold: 0.3,
    equivalence_threshold: 0.005,
    policy_max_bound_epsilon: 0.01,
    prefer_cuda: true,
    use_replay: true,
    use_kd: true,
    use_weight_align: true,
    run_ablations: false,
    fixed_memory: false,
  },
  "full-ps": {
    dataset: "CIFAR-100",
    data_path: "./data",
    num_tasks: 5,
    classes_per_task: 20,
    epochs: 3,
    batch_size: 128,
    num_workers: 0,
    seed: 0,
    old_fraction: 0.2,
    shift_threshold: 0.3,
    equivalence_threshold: 0.005,
    policy_max_bound_epsilon: 0.01,
    prefer_cuda: true,
    use_replay: true,
    use_kd: true,
    use_weight_align: true,
    run_ablations: true,
    fixed_memory: false,
  },
};

function byId(id) {
  return document.getElementById(id);
}

function setText(id, value) {
  const element = byId(id);
  if (element) {
    element.textContent = value;
  }
}

function fmtPercent(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function fmtNumber(value, digits = 2, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function fmtDurationFromIso(value) {
  if (!value) {
    return "n/a";
  }
  const started = new Date(value);
  if (Number.isNaN(started.getTime())) {
    return "n/a";
  }
  const seconds = Math.max(0, Math.floor((Date.now() - started.getTime()) / 1000));
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (minutes < 60) {
    return `${minutes}m ${remainingSeconds}s`;
  }
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${remainingMinutes}m`;
}

function lastItem(items) {
  return items && items.length ? items[items.length - 1] : null;
}

function phaseLabel(status, totalTasks) {
  const phase = status && status.phase ? status.phase : "";
  const currentTask = Number(status && status.current_task ? status.current_task : 0);
  if (phase === "delta_training") {
    return `Delta training (task ${currentTask}/${Math.max(1, totalTasks)})`;
  }
  if (phase === "full_retrain") {
    return `Full retrain baseline (task ${currentTask}/${Math.max(1, totalTasks)})`;
  }
  if (phase === "setup") {
    return "Preparing experiment";
  }
  if (phase === "completed") {
    return "Completed";
  }
  if (phase === "failed") {
    return "Failed";
  }
  return status && status.message ? status.message : "Waiting";
}

function updateRangeOutputs() {
  document.querySelectorAll("input[type='range']").forEach((input) => {
    const output = document.querySelector(`output[data-for="${input.name}"]`);
    if (output) {
      const step = Number(input.step || 1);
      const digits = step < 0.01 ? 3 : 2;
      output.textContent = Number(input.value).toFixed(digits);
    }
  });
}

function serializeForm(form) {
  const data = new FormData(form);
  return {
    dataset: data.get("dataset"),
    data_path: data.get("data_path"),
    num_tasks: Number(data.get("num_tasks")),
    classes_per_task: Number(data.get("classes_per_task")),
    epochs: Number(data.get("epochs")),
    batch_size: Number(data.get("batch_size")),
    num_workers: Number(data.get("num_workers")),
    seed: Number(data.get("seed")),
    old_fraction: Number(data.get("old_fraction")),
    shift_threshold: Number(data.get("shift_threshold")),
    equivalence_threshold: Number(data.get("equivalence_threshold")),
    policy_max_bound_epsilon: Number(data.get("policy_max_bound_epsilon")),
    prefer_cuda: Boolean(form.elements.prefer_cuda.checked),
    use_replay: Boolean(form.elements.use_replay.checked),
    use_kd: Boolean(form.elements.use_kd.checked),
    use_weight_align: Boolean(form.elements.use_weight_align.checked),
    run_ablations: Boolean(form.elements.run_ablations.checked),
    fixed_memory: Boolean(form.elements.fixed_memory.checked),
  };
}

function applyPreset(name) {
  const form = byId("config-form");
  const preset = PRESETS[name];
  if (!form || !preset) {
    return;
  }

  Object.entries(preset).forEach(([key, value]) => {
    const field = form.elements[key];
    if (!field) {
      return;
    }
    if (field.type === "checkbox") {
      field.checked = Boolean(value);
      return;
    }
    field.value = String(value);
  });
  updateRangeOutputs();

  document.querySelectorAll(".preset-card").forEach((button) => {
    button.classList.toggle("selected", button.dataset.preset === name);
  });
}

async function postJson(url, payload = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function renderSharedStatus(payload) {
  const results = payload && payload.results ? payload.results : null;
  const status = results && results.status ? results.status : {};
  const cfg = results && results.config ? results.config : {};
  const tasks = results && results.timeline && results.timeline.tasks ? results.timeline.tasks : [];
  const totalTasks = Number(cfg.num_tasks || 0);
  const completedTasks = tasks.length;
  setText("nav-run-state", (status.state || (payload.running ? "running" : "idle")).toUpperCase());
  setText("nav-task-progress", `${completedTasks}/${Math.max(1, totalTasks)}`);
}

function renderSetupPage(payload) {
  if (!document.body || document.body.dataset.page !== "setup") {
    return;
  }
  const results = payload && payload.results ? payload.results : null;
  const status = results && results.status ? results.status : {};
  const cfg = results && results.config ? results.config : {};
  const tasks = results && results.timeline && results.timeline.tasks ? results.timeline.tasks : [];
  const totalTasks = Number(cfg.num_tasks || 0);

  setText("setup-run-state", (status.state || (payload.running ? "running" : "idle")).toUpperCase());
  setText("setup-phase", phaseLabel(status, totalTasks));
  setText("setup-progress", `${tasks.length}/${Math.max(1, totalTasks)}`);
  setText("setup-elapsed", fmtDurationFromIso(status.started_at));
  setText(
    "setup-status-message",
    status.message || (payload.running ? "Run is active. Open Live Run to monitor it." : "No run active yet.")
  );
}

function renderProgress(payload) {
  const results = payload && payload.results ? payload.results : null;
  const status = results && results.status ? results.status : {};
  const cfg = results && results.config ? results.config : {};
  const tasks = results && results.timeline && results.timeline.tasks ? results.timeline.tasks : [];
  const totalTasks = Number(cfg.num_tasks || 0);
  const completedTasks = tasks.length;

  let deltaProgress = completedTasks / Math.max(1, totalTasks);
  let fullProgress = completedTasks / Math.max(1, totalTasks);
  if (status.phase === "delta_training") {
    deltaProgress = Math.min(1, (completedTasks + 0.5) / Math.max(1, totalTasks));
  } else if (status.phase === "full_retrain") {
    deltaProgress = Math.min(1, (completedTasks + 1) / Math.max(1, totalTasks));
    fullProgress = Math.min(1, (completedTasks + 0.5) / Math.max(1, totalTasks));
  }

  const deltaBar = byId("delta-progress");
  const fullBar = byId("full-progress");
  if (deltaBar) {
    deltaBar.style.width = `${deltaProgress * 100}%`;
  }
  if (fullBar) {
    fullBar.style.width = `${fullProgress * 100}%`;
  }
  setText("delta-progress-label", `${Math.round(deltaProgress * 100)}%`);
  setText("full-progress-label", `${Math.round(fullProgress * 100)}%`);
}

function renderDecision(containerId, deployment) {
  const panel = byId(containerId);
  const template = byId("decision-template");
  if (!panel || !template) {
    return;
  }

  if (!deployment) {
    panel.className = "decision-card muted";
    panel.innerHTML = "<h3>No decision yet</h3><p>Complete a task to populate the recommendation.</p>";
    return;
  }

  const fragment = template.content.cloneNode(true);
  fragment.querySelector("h3").textContent =
    deployment.selected_source === "delta_update" ? "Deploy delta update" : "Fallback to full retrain";
  fragment.querySelector(".decision-subtitle").textContent =
    `Shift score ${fmtNumber(deployment.shift_score, 3)} - Bound epsilon ${fmtPercent(deployment.bound_epsilon)} - Gap ${fmtPercent(deployment.equivalence_gap)}`;

  const list = fragment.querySelector(".reason-list");
  const reasons = deployment.reasons && deployment.reasons.length ? deployment.reasons : ["No reasons recorded."];
  reasons.forEach((reason) => {
    const li = document.createElement("li");
    li.textContent = reason;
    list.appendChild(li);
  });

  panel.className = deployment.selected_source === "delta_update" ? "decision-card safe" : "decision-card warn";
  panel.innerHTML = "";
  panel.appendChild(fragment);
}

function renderMonitorPage(payload) {
  if (!document.body || document.body.dataset.page !== "monitor") {
    return;
  }

  const results = payload && payload.results ? payload.results : null;
  const status = results && results.status ? results.status : {};
  const cfg = results && results.config ? results.config : {};
  const tasks = results && results.timeline && results.timeline.tasks ? results.timeline.tasks : [];
  const partial = results && results.partial_task ? results.partial_task : null;
  const totalTasks = Number(cfg.num_tasks || 0);
  const latestTask = lastItem(tasks);

  setText("monitor-run-state", (status.state || (payload.running ? "running" : "idle")).toUpperCase());
  setText("monitor-phase", phaseLabel(status, totalTasks));
  setText("monitor-progress", `${tasks.length}/${Math.max(1, totalTasks)}`);
  setText("monitor-elapsed", fmtDurationFromIso(status.started_at));
  setText(
    "monitor-status-message",
    status.message ||
      (payload.running ? "Experiment is active. Waiting for results to stream in." : "No active experiment.")
  );

  renderProgress(payload);

  const partialCard = byId("partial-card");
  if (partialCard) {
    if (partial && partial.delta) {
      partialCard.classList.remove("hidden");
      partialCard.innerHTML = `
        <strong>Current partial result</strong><br />
        Delta update for task ${Number(partial.task_id) + 1} finished at
        <b>${fmtPercent(partial.delta.top1)}</b> top-1 in <b>${fmtNumber(partial.delta.wall_time_s, 1, "s")}</b>.
        The app is waiting for the full-retrain baseline to finish this task.
      `;
    } else if (!tasks.length && payload.running) {
      partialCard.classList.remove("hidden");
      partialCard.innerHTML =
        "<strong>Heads up</strong><br />The first task only appears after both the delta update and the full baseline finish.";
    } else {
      partialCard.classList.add("hidden");
      partialCard.innerHTML = "";
    }
  }

  const latestDelta = latestTask && latestTask.delta ? latestTask.delta : {};
  const latestFull = latestTask && latestTask.full ? latestTask.full : {};
  const latestShift = latestTask && latestTask.shift ? latestTask.shift : {};
  const latestEq = latestTask && latestTask.equivalence ? latestTask.equivalence : {};

  setText("monitor-delta-top1", fmtPercent(latestDelta.top1));
  setText("monitor-full-top1", fmtPercent(latestFull.top1));
  setText("monitor-gap", fmtPercent(latestEq.equivalence_gap));
  setText("monitor-shift", fmtNumber(latestShift.shift_score, 3));
  renderDecision("decision-panel", latestTask && latestTask.deployment ? latestTask.deployment : null);

  const logView = byId("log-view");
  if (logView) {
    logView.textContent = payload.logs && payload.logs.trim() ? payload.logs : "No logs yet.";
  }
}

function renderAccuracyChart(tasks) {
  const host = byId("accuracy-chart");
  if (!host) {
    return;
  }
  if (!tasks || !tasks.length) {
    host.className = "chart-box empty";
    host.textContent = "No task data yet.";
    return;
  }

  host.className = "chart-box";
  const width = 620;
  const height = 220;
  const pad = 24;
  const valuesDelta = tasks.map((task) => Number((task.delta || {}).top1 || 0));
  const valuesFull = tasks.map((task) => Number((task.full || {}).top1 || 0));
  const maxVal = Math.max(0.01, ...valuesDelta, ...valuesFull);
  const minVal = Math.min(...valuesDelta, ...valuesFull, 0);
  const scaleX = (index) => pad + (index * (width - pad * 2)) / Math.max(1, tasks.length - 1);
  const scaleY = (value) =>
    height - pad - ((value - minVal) / Math.max(0.0001, maxVal - minVal)) * (height - pad * 2);
  const point = (x, y) => `${x.toFixed(1)},${y.toFixed(1)}`;
  const fullPoints = valuesFull.map((value, index) => point(scaleX(index), scaleY(value))).join(" ");
  const deltaPoints = valuesDelta.map((value, index) => point(scaleX(index), scaleY(value))).join(" ");

  host.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="Accuracy chart">
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="rgba(148,163,184,0.2)" />
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="rgba(148,163,184,0.2)" />
      <polyline fill="none" stroke="#22d3ee" stroke-width="3" points="${fullPoints}" />
      <polyline fill="none" stroke="#a78bfa" stroke-width="3" points="${deltaPoints}" />
    </svg>
    <div class="legend">
      <span><i style="background:#22d3ee"></i>Full retrain</span>
      <span><i style="background:#a78bfa"></i>Delta method</span>
    </div>
  `;
}

function renderClassChart(finalTask) {
  const host = byId("class-chart");
  if (!host) {
    return;
  }
  const full = finalTask && finalTask.full && finalTask.full.per_class_acc ? finalTask.full.per_class_acc : [];
  const delta = finalTask && finalTask.delta && finalTask.delta.per_class_acc ? finalTask.delta.per_class_acc : [];
  const count = Math.min(full.length, delta.length, 10);
  if (!count) {
    host.className = "chart-box empty";
    host.textContent = "No per-class metrics yet.";
    return;
  }

  host.className = "chart-box";
  const rows = [];
  for (let index = 0; index < count; index += 1) {
    rows.push(`
      <tr>
        <td>Class ${index}</td>
        <td>${(Number(full[index] || 0) * 100).toFixed(1)}%</td>
        <td>${(Number(delta[index] || 0) * 100).toFixed(1)}%</td>
      </tr>
    `);
  }
  host.innerHTML = `
    <table>
      <thead>
        <tr><th>Class</th><th>Full</th><th>Delta</th></tr>
      </thead>
      <tbody>${rows.join("")}</tbody>
    </table>
  `;
}

function renderAblations(finalTask) {
  const host = byId("ablations-table");
  if (!host) {
    return;
  }
  const ablations = finalTask && finalTask.ablations ? finalTask.ablations : {};
  const names = Object.keys(ablations);
  if (!names.length) {
    host.className = "table-wrap empty";
    host.textContent = "No ablation results yet.";
    return;
  }

  host.className = "table-wrap";
  host.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Variant</th>
          <th>Top-1</th>
          <th>Gap</th>
          <th>Compute saved</th>
        </tr>
      </thead>
      <tbody>
        ${names
          .map((name) => {
            const record = ablations[name] || {};
            return `
              <tr>
                <td>${name}</td>
                <td>${fmtPercent((record.metrics || {}).top1)}</td>
                <td>${fmtPercent((record.equivalence || {}).equivalence_gap)}</td>
                <td>${fmtNumber((record.equivalence || {}).compute_savings_percent, 2, "%")}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function renderResultsPage(payload) {
  if (!document.body || document.body.dataset.page !== "results") {
    return;
  }
  const results = payload && payload.results ? payload.results : null;
  const tasks = results && results.timeline && results.timeline.tasks ? results.timeline.tasks : [];
  const latestTask = lastItem(tasks);
  const summary = results && results.final_summary ? results.final_summary : {};
  const latestFull = latestTask && latestTask.full ? latestTask.full : {};
  const latestDelta = latestTask && latestTask.delta ? latestTask.delta : {};
  const latestEq = latestTask && latestTask.equivalence ? latestTask.equivalence : {};

  setText(
    "results-status-message",
    tasks.length
      ? "Completed task data is available below."
      : payload.running
        ? "Waiting for the first task to finish."
        : "No completed tasks yet."
  );
  setText("metric-compute-saved", fmtNumber(summary.compute_savings_percent, 2, "%"));
  setText("metric-policy-saved", fmtNumber(summary.policy_compute_savings_percent, 2, "%"));
  setText("metric-gap", fmtPercent(latestEq.equivalence_gap));
  setText("metric-choice", latestTask && latestTask.deployment ? latestTask.deployment.selected_source || "-" : "-");
  setText("metric-full-top1", fmtPercent(latestFull.top1));
  setText("metric-delta-top1", fmtPercent(latestDelta.top1));

  renderAccuracyChart(tasks);
  renderClassChart(latestTask);
  renderDecision("decision-panel", latestTask && latestTask.deployment ? latestTask.deployment : null);
  renderAblations(latestTask);
}

async function refreshState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    const payload = await response.json();
    renderSharedStatus(payload);
    renderSetupPage(payload);
    renderMonitorPage(payload);
    renderResultsPage(payload);
  } catch (error) {
    console.error(error);
    setText("setup-status-message", error.message);
    setText("monitor-status-message", error.message);
    setText("results-status-message", error.message);
  }
}

function bindCommonButtons() {
  const refreshButton = byId("refresh-btn");
  if (refreshButton) {
    refreshButton.addEventListener("click", refreshState);
  }

  const stopButton = byId("stop-btn");
  if (stopButton) {
    stopButton.addEventListener("click", async () => {
      try {
        await postJson("/api/stop");
        await refreshState();
      } catch (error) {
        console.error(error);
      }
    });
  }
}

function bindSetupPage() {
  const form = byId("config-form");
  if (!form) {
    return;
  }

  document.querySelectorAll(".preset-card").forEach((button) => {
    button.addEventListener("click", () => applyPreset(button.dataset.preset));
  });

  form.querySelectorAll("input[type='range']").forEach((input) => {
    input.addEventListener("input", updateRangeOutputs);
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await postJson("/api/run", serializeForm(form));
      await refreshState();
      window.location.href = "/monitor.html";
    } catch (error) {
      console.error(error);
      setText("setup-status-message", error.message);
    }
  });

  applyPreset("fast-demo");
}

function init() {
  bindCommonButtons();
  bindSetupPage();
  updateRangeOutputs();
  refreshState();
  window.setInterval(refreshState, POLL_INTERVAL_MS);
}

window.addEventListener("DOMContentLoaded", init);
