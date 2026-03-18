from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

DEFAULT_RESULTS_PATH = "results.json"
DEFAULT_LOG_PATH = "experiment.log"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail_text(path: Path, max_lines: int = 200) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def _bool_flag(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_run_command(config: Dict[str, Any], *, results_path: Path) -> list[str]:
    dataset = str(config.get("dataset", "CIFAR-100"))
    data_path = str(config.get("data_path", "./data"))
    num_tasks = int(config.get("num_tasks", 5))
    classes_per_task = int(config.get("classes_per_task", 20))
    old_fraction = float(config.get("old_fraction", 0.2))
    epochs = int(config.get("epochs", 3))
    batch_size = int(config.get("batch_size", 128))
    num_workers = int(config.get("num_workers", 0))
    seed = int(config.get("seed", 0))
    backbone = str(config.get("backbone", "resnet32"))
    lr = float(config.get("lr", 0.1))
    momentum = float(config.get("momentum", 0.9))
    weight_decay = float(config.get("weight_decay", 5e-4))
    lambda_kd = float(config.get("lambda_kd", 0.5))
    kd_temperature = float(config.get("kd_temperature", 2.0))
    memory_size = int(config.get("memory_size", 2000))
    herding_method = str(config.get("herding_method", "barycenter"))
    shift_threshold = float(config.get("shift_threshold", 0.3))
    equivalence_threshold = float(config.get("equivalence_threshold", 0.005))
    policy_max_bound_epsilon = float(config.get("policy_max_bound_epsilon", 0.01))

    cmd = [
        sys.executable,
        "-m",
        "delta_framework.experiments.run_experiment",
        "--dataset",
        dataset,
        "--data-path",
        data_path,
        "--num-tasks",
        str(num_tasks),
        "--classes-per-task",
        str(classes_per_task),
        "--old-fraction",
        str(old_fraction),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--seed",
        str(seed),
        "--backbone",
        backbone,
        "--lr",
        str(lr),
        "--momentum",
        str(momentum),
        "--weight-decay",
        str(weight_decay),
        "--lambda-kd",
        str(lambda_kd),
        "--kd-temperature",
        str(kd_temperature),
        "--memory-size",
        str(memory_size),
        "--herding-method",
        herding_method,
        "--shift-threshold",
        str(shift_threshold),
        "--equivalence-threshold",
        str(equivalence_threshold),
        "--policy-max-bound-epsilon",
        str(policy_max_bound_epsilon),
        "--results-path",
        str(results_path),
    ]

    if _bool_flag(config.get("prefer_cuda"), True):
        cmd.append("--prefer-cuda")
    if _bool_flag(config.get("fixed_memory"), False):
        cmd.append("--fixed-memory")
    if not _bool_flag(config.get("use_replay"), True):
        cmd.append("--disable-replay")
    if not _bool_flag(config.get("use_kd"), True):
        cmd.append("--disable-kd")
    if not _bool_flag(config.get("use_weight_align"), True):
        cmd.append("--disable-weight-align")
    if _bool_flag(config.get("run_ablations"), False):
        cmd.append("--run-ablations")
        ablations = config.get("ablation_variants", []) or []
        if isinstance(ablations, list) and ablations:
            cmd.extend(["--ablation-variants", *[str(item) for item in ablations]])

    return cmd


@dataclass
class WebAppPaths:
    workdir: Path
    results_path: Path
    log_path: Path
    static_dir: Path


class ExperimentManager:
    def __init__(self, paths: WebAppPaths) -> None:
        self.paths = paths
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen[str]] = None
        self._log_handle: Optional[Any] = None

    def _sync_process_state(self) -> None:
        if self._proc is not None and self._proc.poll() is not None:
            if self._log_handle is not None:
                try:
                    self._log_handle.close()
                except Exception:
                    pass
            self._proc = None
            self._log_handle = None

    def is_running(self) -> bool:
        with self._lock:
            self._sync_process_state()
            return self._proc is not None

    def start(self, config: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._sync_process_state()
            if self._proc is not None:
                raise RuntimeError("An experiment is already running.")

            self.paths.results_path.parent.mkdir(parents=True, exist_ok=True)
            self.paths.log_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = build_run_command(config, results_path=self.paths.results_path)
            self._log_handle = self.paths.log_path.open("w", encoding="utf-8", buffering=1)
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self.paths.workdir),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return {"started": True, "pid": int(self._proc.pid), "command": cmd}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            self._sync_process_state()
            if self._proc is None:
                return {"stopped": False, "message": "No experiment is currently running."}

            proc = self._proc
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            self._sync_process_state()
            return {"stopped": True}

    def read_state(self) -> Dict[str, Any]:
        with self._lock:
            self._sync_process_state()
            return {
                "running": self._proc is not None,
                "results": _read_json(self.paths.results_path),
                "logs": _tail_text(self.paths.log_path),
                "paths": {
                    "results_path": str(self.paths.results_path),
                    "log_path": str(self.paths.log_path),
                },
            }


def _send_json(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: Dict[str, Any]) -> None:
    raw = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _serve_static(handler: BaseHTTPRequestHandler, static_dir: Path, path: str) -> None:
    relative = "index.html" if path in {"/", ""} else path.lstrip("/")
    candidate = (static_dir / relative).resolve()
    if static_dir.resolve() not in candidate.parents and candidate != static_dir.resolve():
        handler.send_error(HTTPStatus.FORBIDDEN)
        return
    if not candidate.exists() or not candidate.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return

    mime, _ = mimetypes.guess_type(str(candidate))
    raw = candidate.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def create_handler(manager: ExperimentManager, static_dir: Path) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/state":
                _send_json(self, HTTPStatus.OK, manager.read_state())
                return
            _serve_static(self, static_dir, parsed.path)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len) if content_len else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                _send_json(self, HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body."})
                return

            try:
                if parsed.path == "/api/run":
                    _send_json(self, HTTPStatus.OK, manager.start(payload))
                    return
                if parsed.path == "/api/stop":
                    _send_json(self, HTTPStatus.OK, manager.stop())
                    return
            except RuntimeError as exc:
                _send_json(self, HTTPStatus.CONFLICT, {"error": str(exc)})
                return
            except Exception as exc:  # pragma: no cover
                _send_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            _send_json(self, HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return DashboardHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Delta Framework web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--workdir", default=".")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    base_dir = Path(__file__).resolve().parent
    paths = WebAppPaths(
        workdir=Path(args.workdir).resolve(),
        results_path=Path(args.results_path).resolve(),
        log_path=Path(args.log_path).resolve(),
        static_dir=base_dir / "static",
    )
    manager = ExperimentManager(paths)
    server = ThreadingHTTPServer((args.host, args.port), create_handler(manager, paths.static_dir))
    print(f"Delta Framework web app running on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()
        server.server_close()


if __name__ == "__main__":
    main()
