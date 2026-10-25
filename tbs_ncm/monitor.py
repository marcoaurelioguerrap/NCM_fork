"""Run bookkeeping and metric logging for the TBS/NCM pipeline.

Every data, training or eval stage writes one directory under
``artifacts/tbs/runs/``::

    runs/20261020-223140-generator/
        run.json        stage, argv, git commit, host, total_steps
        config.yaml     snapshot of tbs_ncm/config.yaml
        metrics.jsonl   one JSON event per line, flushed after every write
        samples.jsonl   target/generated pairs from the epoch evaluation
        status.json     running | done | crashed, pid, heartbeat
        best.pt         best checkpoint of the run
        last.pt         checkpoint from the last finished epoch
        tb/             TensorBoard events when the run was started with --tb

``python -m tbs_ncm.dashboard`` reads these files; nothing here depends on it,
and training must not fail because a dashboard is not running.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .common import ARTIFACT_DIR, CONFIG_PATH, NCM_ROOT

RUNS_DIR = ARTIFACT_DIR / "runs"
GPU_INTERVAL_S = 5.0

_GPU_CACHE: dict[str, object] = {"at": 0.0, "value": None}


def runs_root(path: Path | str | None = None) -> Path:
    return Path(path) if path else RUNS_DIR


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _iso(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(timestamp if timestamp is not None else time.time()).isoformat(
        timespec="seconds"
    )


def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    """Run a small helper command; return stdout or ``None`` when unavailable."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, creationflags=flags
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def git_revision(root: Path = NCM_ROOT) -> str | None:
    commit = _run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], timeout=5)
    if not commit:
        return None
    dirty = _run(["git", "-C", str(root), "status", "--porcelain"], timeout=5)
    return f"{commit}-dirty" if dirty else commit


def gpu_status(refresh: bool = False) -> dict[str, float] | None:
    """Utilization/memory/temperature from ``nvidia-smi``, cached for a few seconds."""
    now = time.time()
    if not refresh and now - float(_GPU_CACHE["at"]) < GPU_INTERVAL_S:
        cached = _GPU_CACHE["value"]
        return cached if isinstance(cached, dict) else None

    info: dict[str, float] | None = None
    fields = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    text = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if text:
        parts = [part.strip() for part in text.splitlines()[0].split(",")]
        if len(parts) >= 5:
            numbers: list[float | None] = []
            for part in parts[:5]:
                try:
                    numbers.append(float(part))
                except ValueError:
                    numbers.append(None)
            keys = ("gpu_util", "gpu_mem_mb", "gpu_mem_total_mb", "gpu_temp_c", "gpu_power_w")
            info = {key: value for key, value in zip(keys, numbers) if value is not None}
    _GPU_CACHE.update(at=now, value=info)
    return info


def torch_memory() -> dict[str, float]:
    """Allocated/reserved VRAM from torch, when CUDA is present."""
    try:
        import torch
    except Exception:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        mib = 2**20
        return {
            "torch_alloc_mb": round(torch.cuda.memory_allocated() / mib, 1),
            "torch_reserved_mb": round(torch.cuda.memory_reserved() / mib, 1),
        }
    except Exception:
        return {}


def _scalar(value: object) -> object:
    """JSON-safe value: tensors/arrays become scalars, NaN/inf become ``None``."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(f"{value:.9g}")
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _scalar(item())
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        return [_scalar(entry) for entry in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically so a dashboard never reads a half-written file."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path, default: object = None) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_metrics(run_dir: Path | str, limit: int | None = None) -> list[dict]:
    """Parse ``metrics.jsonl``, skipping a partially written trailing line."""
    path = Path(run_dir) / "metrics.jsonl"
    events: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return events
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    if limit is not None and len(events) > limit:
        events = events[-limit:]
    return events


def read_samples(run_dir: Path | str, limit: int = 40) -> list[dict]:
    path = Path(run_dir) / "samples.jsonl"
    samples: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return samples
    for line in lines[-limit:]:
        try:
            sample = json.loads(line)
        except ValueError:
            continue
        if isinstance(sample, dict):
            samples.append(sample)
    return samples


def _tail_events(run_dir: Path, lines: int = 40) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            handle.seek(max(0, end - 32768))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    events = []
    for line in text.splitlines()[-lines:]:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


class RunLogger:
    """Create one run directory and append metrics/samples/status to it."""

    def __init__(
        self,
        stage: str,
        *,
        runs_dir: Path | str | None = None,
        config_path: Path | str | None = CONFIG_PATH,
        argv: list[str] | None = None,
        total_steps: int | None = None,
        tensorboard: bool = False,
        label: str | None = None,
        verbose: bool = True,
    ):
        self.stage = stage
        self.verbose = verbose
        self.total_steps = int(total_steps) if total_steps and total_steps > 0 else None
        self.started = time.time()
        self._step: int | None = None
        self._epoch: int | None = None
        self._last_step: int | None = None
        self._last_t = self.started
        self._state = "running"
        self._tb = None
        self._tb_failed = False

        root = runs_root(runs_dir)
        base = f"{_stamp()}-{stage}" + (f"-{label}" if label else "")
        self.dir = root / base
        suffix = 1
        while self.dir.exists():
            suffix += 1
            self.dir = root / f"{base}-{suffix}"
        self.dir.mkdir(parents=True, exist_ok=True)

        if config_path is not None and Path(config_path).exists():
            shutil.copyfile(config_path, self.dir / "config.yaml")

        meta = {
            "stage": stage,
            "label": label,
            "started": _iso(self.started),
            "pid": os.getpid(),
            "argv": list(argv or sys.argv),
            "cwd": str(Path.cwd()),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "git": git_revision(),
            "total_steps": self.total_steps,
        }
        _write_json(self.dir / "run.json", meta)

        self.metrics = (self.dir / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
        self.samples = (self.dir / "samples.jsonl").open("a", encoding="utf-8", buffering=1)
        self._write_status()

        if tensorboard:
            self._enable_tensorboard()
        if self.verbose:
            print(f"run log: {self.dir}")

    # ------------------------------------------------------------------ status

    def _write_status(self, **extra: object) -> None:
        payload = {
            "state": self._state,
            "stage": self.stage,
            "pid": os.getpid(),
            "started": _iso(self.started),
            "heartbeat": time.time(),
            "heartbeat_iso": _iso(),
            "step": self._step,
            "epoch": self._epoch,
        }
        payload.update(extra)
        try:
            _write_json(self.dir / "status.json", payload)
        except OSError:
            pass

    def heartbeat(self) -> None:
        """Refresh the status file without appending a metric event."""
        self._write_status()

    def set_state(self, state: str, **extra: object) -> None:
        self._state = state
        self._write_status(**extra)

    # ------------------------------------------------------------------- write

    def _append(self, event: dict) -> None:
        try:
            self.metrics.write(json.dumps(event, separators=(",", ":")) + "\n")
            self.metrics.flush()
        except OSError:
            pass

    def log(self, *, kind: str = "step", step: int | None = None, epoch: int | None = None,
            gpu: bool = False, **metrics: object) -> dict:
        """Append one event.  ``step`` drives throughput and ETA."""
        now = time.time()
        event: dict[str, object] = {"t": round(now, 3), "kind": kind, "stage": self.stage}
        if step is not None:
            event["step"] = int(step)
        if epoch is not None:
            event["epoch"] = int(epoch)
        for key, value in metrics.items():
            clean = _scalar(value)
            if clean is not None:
                event[key] = clean

        if step is not None:
            if self.total_steps:
                event["progress"] = round(min(1.0, step / self.total_steps), 5)
            if self._last_step is not None and step > self._last_step and now > self._last_t:
                elapsed = now - self._last_t
                rate = (step - self._last_step) / elapsed if elapsed > 0 else 0.0
                if rate > 0:
                    event["steps_per_s"] = round(rate, 3)
                    if self.total_steps:
                        event["eta_s"] = round(max(0.0, (self.total_steps - step) / rate), 1)
            self._last_step = step
            self._last_t = now
            self._step = step
        if epoch is not None:
            self._epoch = epoch

        if gpu:
            info = gpu_status()
            if info:
                event.update(info)
            event.update(torch_memory())

        self._append(event)
        self._write_status()
        self._tb_scalars(event)
        return event

    def milestone(self, text: str, **metrics: object) -> dict:
        return self.log(kind="note", text=text, **metrics)

    def log_samples(self, samples: list[dict], *, step: int | None = None,
                    epoch: int | None = None, limit: int = 12) -> None:
        """Persist generated/expected pairs for the dashboard's diff table."""
        chosen = samples[:limit]
        for sample in chosen:
            payload = {"t": round(time.time(), 3), "stage": self.stage}
            if step is not None:
                payload["step"] = int(step)
            if epoch is not None:
                payload["epoch"] = int(epoch)
            for key, value in sample.items():
                clean = _scalar(value)
                if clean is not None:
                    payload[key] = clean
            try:
                self.samples.write(json.dumps(payload, separators=(",", ":")) + "\n")
                self.samples.flush()
            except OSError:
                pass
        if self._tb is not None and chosen:
            lines = ["| target | generated | ok |", "|---|---|---|"]
            for sample in chosen:
                target = str(sample.get("target", "")).replace("|", "\\|")
                generated = str(sample.get("generated", "")).replace("|", "\\|")
                lines.append(f"| `{target}` | `{generated}` | {sample.get('exact', '')} |")
            self._tb.add_text(
                f"{self.stage}/samples", "\n".join(lines), step or epoch or 0
            )

    def finish(self, state: str = "done", **metrics: object) -> None:
        self._state = state
        payload = {key: _scalar(value) for key, value in metrics.items()}
        event = {"t": round(time.time(), 3), "kind": "final", "stage": self.stage}
        event.update({key: value for key, value in payload.items() if value is not None})
        if self._step is not None:
            event["step"] = self._step
        if self._epoch is not None:
            event["epoch"] = self._epoch
        event["duration_s"] = round(time.time() - self.started, 1)
        self._append(event)
        self._write_status(**{key: value for key, value in payload.items() if value is not None})
        self._tb_scalars(event)
        self.close()

    def close(self) -> None:
        for handle in (getattr(self, "metrics", None), getattr(self, "samples", None)):
            try:
                if handle is not None and not handle.closed:
                    handle.flush()
                    handle.close()
            except OSError:
                pass
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            except Exception:
                pass
            self._tb = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._state == "running":
            self.finish("crashed" if exc_type is not None else "done")
        if exc_type is not None:
            print(f"run failed: {exc_type.__name__}: {exc}", file=sys.stderr)
        return False

    # ------------------------------------------------------------- tensorboard

    def _enable_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:  # tensorboard is optional
            self._tb_failed = True
            if self.verbose:
                print(f"tensorboard unavailable ({exc.__class__.__name__}); writing JSONL only")
            return
        try:
            self._tb = SummaryWriter(log_dir=str(self.dir / "tb"))
        except Exception as exc:
            self._tb_failed = True
            if self.verbose:
                print(f"tensorboard writer failed ({exc}); writing JSONL only")

    def _tb_scalars(self, event: dict) -> None:
        if self._tb is None:
            return
        x = event.get("step") or event.get("epoch") or 0
        skip = {"t", "kind", "stage", "step", "epoch", "text", "progress"}
        try:
            for key, value in event.items():
                if key in skip or not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                self._tb.add_scalar(f"{self.stage}/{key}", value, x)
            text = event.get("text")
            if isinstance(text, str):
                self._tb.add_text(f"{self.stage}/notes", text, x)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# read-side helpers shared with the dashboard
# ---------------------------------------------------------------------------


def summarize_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "run.json", {}) or {}
    status = read_json(run_dir / "status.json", {}) or {}
    tail = _tail_events(run_dir)
    last = tail[-1] if tail else None
    best = None
    for event in reversed(tail):
        if event.get("best"):
            best = event
            break
    heartbeat = status.get("heartbeat")
    age = round(time.time() - float(heartbeat), 1) if isinstance(heartbeat, (int, float)) else None
    return {
        "id": run_dir.name,
        "stage": meta.get("stage") or status.get("stage"),
        "label": meta.get("label"),
        "state": status.get("state"),
        "started": meta.get("started") or status.get("started"),
        "heartbeat_age_s": age,
        "total_steps": meta.get("total_steps"),
        "last": last,
        "best": best,
    }


def list_runs(runs_dir: Path | str | None = None) -> list[dict]:
    """Every run directory, newest first."""
    root = runs_root(runs_dir)
    if not root.exists():
        return []
    dirs = [entry for entry in root.iterdir() if entry.is_dir()]
    dirs.sort(key=lambda path: path.name, reverse=True)
    runs = []
    for run_dir in dirs:
        if not (run_dir / "run.json").exists():
            continue
        runs.append(summarize_run(run_dir))
    return runs
