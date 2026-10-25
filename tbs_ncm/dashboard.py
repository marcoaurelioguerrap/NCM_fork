"""Live dashboard for the TBS/NCM pipeline.

    python -m tbs_ncm.dashboard            # http://127.0.0.1:8765
    python -m tbs_ncm.dashboard --open

Standard library only: it reads the run directories written by
:mod:`tbs_ncm.monitor`, the match saves under ``artifacts/tbs/matches`` and the
arena/self-play summaries under ``artifacts/tbs/vs_ncm``.  The game browser
replays saved order lists through the game engine; no torch and no CDN.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .common import ARTIFACT_DIR
from .gamefeed import games_payload, match_detail_payload
from .monitor import list_runs, read_json, read_metrics, read_samples, runs_root, summarize_run

MAX_STEP_EVENTS = 900
MAX_EPOCH_EVENTS = 400


def _downsample(events: list[dict]) -> list[dict]:
    """Keep every non-step event and a uniform sample of step events."""
    steps = [event for event in events if event.get("kind") != "step"]
    dense = [event for event in events if event.get("kind") == "step"]
    if len(dense) > MAX_STEP_EVENTS:
        stride = len(dense) / MAX_STEP_EVENTS
        dense = [dense[int(i * stride)] for i in range(MAX_STEP_EVENTS - 1)] + [dense[-1]]
    if len(steps) > MAX_EPOCH_EVENTS:
        stride = len(steps) / MAX_EPOCH_EVENTS
        steps = [steps[int(i * stride)] for i in range(MAX_EPOCH_EVENTS - 1)] + [steps[-1]]
    merged = sorted(steps + dense, key=lambda event: event.get("t", 0))
    return merged


def _digits(value: object) -> str:
    return "".join(char for char in str(value or "") if char.isdigit())


def _argv_value(argv: list, name: str) -> str | None:
    """Return ``--name value`` or ``--name=value`` from a recorded argv."""
    for index, arg in enumerate(argv):
        if arg == name and index + 1 < len(argv):
            return str(argv[index + 1])
        if isinstance(arg, str) and arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def _argv_flag(argv: list, name: str) -> bool:
    return any(
        isinstance(arg, str) and (arg == name or arg.startswith(name + "="))
        for arg in argv
    )


def _short_dir(value: object) -> str:
    """``C:\\...\\artifacts\\tbs`` → ``artifacts/tbs`` for compact display."""
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    return "/".join(path.parts[-2:]) if len(path.parts) >= 2 else (path.name or text)


def _report_view(path: Path, data: dict) -> dict:
    return {
        "path": str(path),
        "run": path.parent.name,
        "started": data.get("started"),
        "elapsed_s": data.get("elapsed_s"),
        "learner": data.get("learner"),
        "frozen_opponent": data.get("frozen_opponent"),
        "out_dir": data.get("out_dir"),
        "config": data.get("config") or {},
        "iterations": data.get("iterations") or [],
        "error": data.get("error"),
    }


def _report_for_run(meta: dict) -> dict | None:
    """Find the ``vs_ncm`` report that belongs to one training run."""
    argv = meta.get("argv") or []
    cwd = Path(meta.get("cwd") or ".")
    candidates: list[Path] = []
    out_dir = _argv_value(argv, "--out-dir")
    if out_dir:
        path = Path(out_dir)
        if not path.is_absolute():
            path = cwd / path
        candidates.append(path / "report.json")

    started = _digits(meta.get("started"))
    for path in candidates:
        data = read_json(path, {})
        if isinstance(data, dict) and data:
            return _report_view(path, data)

    root = ARTIFACT_DIR / "vs_ncm"
    if root.is_dir():
        for path in sorted(root.rglob("report.json")):
            data = read_json(path, {})
            if not isinstance(data, dict):
                continue
            if started and started == _digits(data.get("started")):
                return _report_view(path, data)
    return None


def matchup_payload(meta: dict, report: dict | None = None) -> dict:
    """Who played whom in one run, derived from ``report.json`` then argv."""
    stage = str(meta.get("stage") or "")
    argv = list(meta.get("argv") or [])
    report = report or {}
    config = report.get("config") if isinstance(report.get("config"), dict) else {}

    def option(key: str, flag: str | None = None) -> object:
        value = config.get(key)
        if value is None or value == "":
            value = _argv_value(argv, flag or "--" + key.replace("_", "-"))
        return value

    def flag(key: str, name: str | None = None) -> bool:
        value = config.get(key)
        if isinstance(value, bool):
            return value
        return _argv_flag(argv, name or "--" + key.replace("_", "-"))

    info: dict = {
        "kind": stage,
        "has_opponent": stage in ("vs_ncm", "arena", "selfplay"),
        "summary": "",
        "description": "",
        "learner": None,
        "opponent": None,
        "settings": {},
    }

    if info["has_opponent"]:
        learner_path = str(
            report.get("learner")
            or option("checkpoint_dir")
            or option("learner_checkpoint_dir", "--learner-checkpoint-dir")
            or ARTIFACT_DIR
        )
        configured_opponent = str(option("opponent_checkpoint_dir") or "").strip()
        if stage == "vs_ncm":
            opponent_path = str(
                report.get("frozen_opponent") or configured_opponent or learner_path
            )
        else:
            opponent_path = configured_opponent or learner_path

        update_opponent = flag("update_opponent")
        teacher = flag("teacher")
        learner_side = str(option("learner_side") or "B")
        alternate = not flag("no_alternate_sides")

        if stage == "vs_ncm":
            if update_opponent:
                opponent_label = "previous iteration's learner"
                info["summary"] = "vs previous iteration's learner"
                info["description"] = (
                    "Learner NCM vs the previous iteration's learner "
                    "(fictitious self-play; the opponent moves every iteration)."
                )
            elif configured_opponent:
                opponent_label = "fixed NCM checkpoint"
                info["summary"] = "vs fixed NCM checkpoint"
                info["description"] = "Learner NCM vs a fixed NCM checkpoint."
            else:
                opponent_label = "frozen copy of the initial checkpoint"
                info["summary"] = "vs frozen initial checkpoint"
                info["description"] = (
                    "Learner NCM vs a frozen copy of its initial checkpoint; "
                    "baseline and post-training use the same opponent."
                )
        elif stage == "selfplay":
            if configured_opponent:
                opponent_label = "fixed NCM checkpoint"
                info["summary"] = "self-play vs NCM checkpoint"
                info["description"] = "NCM self-play: learner vs a fixed NCM checkpoint."
            else:
                opponent_label = "same checkpoint (mirror)"
                info["summary"] = "self-play (same checkpoint)"
                info["description"] = (
                    "NCM self-play: learner vs a copy of the same checkpoint."
                )
        else:  # arena
            same = Path(configured_opponent or learner_path).resolve() == Path(
                learner_path
            ).resolve()
            if configured_opponent and not same:
                opponent_label = "fixed NCM checkpoint"
                info["summary"] = "vs NCM checkpoint"
                info["description"] = "NCM arena: learner vs a fixed NCM checkpoint."
            else:
                opponent_label = "same checkpoint (mirror)"
                info["summary"] = "NCM mirror match"
                info["description"] = (
                    "NCM arena: mirror match between copies of the same checkpoint."
                )

        info["learner"] = {
            "label": "Learner NCM",
            "short": _short_dir(learner_path),
            "path": learner_path,
            "side": learner_side,
        }
        info["opponent"] = {
            "label": opponent_label,
            "short": _short_dir(opponent_path),
            "path": opponent_path,
        }
        settings = {
            "learner_side": learner_side,
            "alternate_sides": alternate,
            "games": option("games"),
            "teacher": teacher,
        }
        if stage == "vs_ncm":
            settings["update_opponent"] = update_opponent
            settings["eval_games"] = option("eval_games")
        info["settings"] = {
            key: value for key, value in settings.items() if value not in (None, "")
        }
        return info

    if stage == "generator":
        info["summary"] = "no opponent (offline fine-tune)"
        info["description"] = "Offline fine-tune on a fixed dataset — no live opponent."
        info["settings"] = {
            key: value
            for key, value in {
                "dataset": _short_dir(option("train")),
                "validation": _short_dir(option("validation")),
                "epochs": option("epochs"),
            }.items()
            if value not in (None, "")
        }
    elif stage == "tokenizer":
        info["summary"] = "no opponent (tokenizer training)"
        info["description"] = "Tokenizer training on a text corpus — no opponent."
    elif stage == "data":
        info["summary"] = "no opponent (greedy data generation)"
        info["description"] = "Greedy-policy data generation — no live opponent."
    elif stage == "eval":
        info["summary"] = "no opponent (order evaluation)"
        info["description"] = "Order evaluation on a fixed dataset — no opponent."
    else:
        info["summary"] = "no opponent"
        info["description"] = f"{stage.capitalize() or 'This'} run has no live opponent."
    return info


def run_payload(run_id: str, root: Path) -> dict | None:
    base = root.resolve()
    run_dir = (root / unquote(run_id)).resolve()
    if base not in run_dir.parents or not run_dir.is_dir():
        return None
    summary = summarize_run(run_dir)
    events = read_metrics(run_dir)
    meta = read_json(run_dir / "run.json", {}) or {}
    payload = {
        "id": summary["id"],
        "meta": meta,
        "status": read_json(run_dir / "status.json", {}) or {},
        "summary": summary,
        "events": _downsample(events),
        "n_events": len(events),
        "samples": read_samples(run_dir, limit=60),
        "files": sorted(entry.name for entry in run_dir.iterdir()),
    }
    if summary.get("stage") == "vs_ncm":
        report = _report_for_run(meta)
        if report:
            payload["report"] = report
    payload["matchup"] = matchup_payload(meta, payload.get("report"))
    return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "tbs-ncm-dashboard/1.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # keep the console clean
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        route = urlparse(self.path).path
        root = runs_root(self.server.runs_dir)  # type: ignore[attr-defined]

        if route in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "/api/runs":
            runs = list_runs(root)
            for run in runs:
                meta = read_json(root / run["id"] / "run.json", {}) or {}
                run["matchup"] = matchup_payload(meta)
            self._json({"runs": runs, "runs_dir": str(root)})
            return
        if route.startswith("/api/run/"):
            payload = run_payload(route[len("/api/run/"):], root)
            self._json(payload if payload else {"error": "unknown run"})
            return
        if route == "/api/games" or route == "/api/matches":
            self._json(games_payload())
            return
        if route.startswith("/api/match/"):
            payload = match_detail_payload(unquote(route[len("/api/match/"):]))
            self._json(payload)
            return
        if route == "/favicon.ico":
            self._send(b"", "image/x-icon", 204)
            return
        guessed = mimetypes.guess_type(route)[0] or "text/plain"
        self._send(f"not found: {route}".encode("utf-8"), guessed, 404)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NCM training</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --line: #262d36; --text: #e6edf3;
    --dim: #8b949e; --accent: #58a6ff; --ok: #3fb950; --warn: #d29922; --bad: #f85149;
    --c1: #58a6ff; --c2: #f0883e; --c3: #3fb950; --c4: #bc8cff; --c5: #ff7b72;
    --a: #58a6ff; --b: #f0883e;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  body.modal-open { overflow: hidden; }
  header { display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
           padding: 12px 18px; border-bottom: 1px solid var(--line);
           position: sticky; top: 0; background: rgba(13,17,23,.96); z-index: 5; }
  h1 { font-size: 14px; font-weight: 600; margin: 0; letter-spacing: .4px; }
  h2 { font-size: 11px; margin: 0 0 8px; color: var(--dim); font-weight: 600;
       text-transform: uppercase; letter-spacing: .8px; }
  .tabs { display: flex; gap: 4px; }
  .tab { background: transparent; color: var(--dim); border: 1px solid transparent;
         border-radius: 6px; padding: 5px 10px; font: inherit; cursor: pointer; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--text); background: var(--panel); border-color: var(--line); }
  select, input { background: var(--panel); color: var(--text); border: 1px solid var(--line);
           border-radius: 6px; padding: 5px 8px; font: inherit; }
  select { max-width: 44vw; }
  .badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 9px;
           border: 1px solid var(--line); border-radius: 999px; font-size: 11px;
           text-transform: uppercase; letter-spacing: .6px; color: var(--dim); }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--dim); }
  .live .dot { background: var(--ok); box-shadow: 0 0 0 0 rgba(63,185,80,.7);
               animation: pulse 1.6s infinite; }
  .stalled .dot { background: var(--warn); }
  .crashed .dot { background: var(--bad); }
  @keyframes pulse { 70% { box-shadow: 0 0 0 7px rgba(63,185,80,0); }
                     100% { box-shadow: 0 0 0 0 rgba(63,185,80,0); } }
  .grow { flex: 1; }
  main { padding: 16px 18px 60px; max-width: 1500px; margin: 0 auto; }
  .toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
  .filters { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .filters label { color: var(--dim); font-size: 11px; display: flex; align-items: center; gap: 6px; }
  .matchup-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .match-side { flex: 1 1 240px; min-width: 0; background: #0b0f14;
                border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  .match-side .k { color: var(--dim); font-size: 10px; text-transform: uppercase;
                   letter-spacing: .6px; }
  .match-side .v { font-size: 13px; margin-top: 2px; }
  .match-side .path { color: var(--dim); font-size: 10.5px; margin-top: 3px;
                      word-break: break-all; }
  .match-vs { color: var(--dim); font-size: 10.5px; text-transform: uppercase;
              letter-spacing: .8px; }
  .match-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }
  .match-note { color: var(--dim); margin-top: 9px; }
  #kpis, #game-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(132px,1fr));
          gap: 10px; margin-bottom: 16px; }
  .kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
         padding: 9px 11px; }
  .kpi .k { color: var(--dim); font-size: 10.5px; text-transform: uppercase;
            letter-spacing: .7px; }
  .kpi .v { font-size: 17px; margin-top: 3px; font-variant-numeric: tabular-nums; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px,1fr));
          gap: 14px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
          padding: 12px 14px; }
  .chart { width: 100%; height: 168px; display: block; }
  .chart text { fill: var(--dim); font-size: 10px; font-family: inherit; }
  .empty { color: var(--dim); font-size: 12px; padding: 28px 0; text-align: center; }
  .legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px;
            font-size: 11px; color: var(--dim); }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .swatch { width: 9px; height: 3px; border-radius: 2px; display: inline-block; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; color: var(--dim); font-weight: 500; font-size: 10.5px;
       text-transform: uppercase; letter-spacing: .6px; padding: 4px 8px 6px 0; }
  td { padding: 5px 8px 5px 0; border-top: 1px solid var(--line); vertical-align: top; }
  tr.game-row { cursor: pointer; }
  tr.game-row:hover td { background: rgba(88,166,255,.06); }
  tr.game-row.disabled { cursor: default; }
  tr.game-row.disabled:hover td { background: transparent; }
  td.matchup { white-space: nowrap; padding-right: 18px; }
  td.when { white-space: nowrap; font-variant-numeric: tabular-nums; }
  td.result-cell { min-width: 112px; white-space: nowrap; }
  td.mono, .mono { font-family: inherit; white-space: pre; }
  .bad { color: var(--bad); background: rgba(248,81,73,.15); border-radius: 2px; }
  .ok { color: var(--ok); }
  .warn { color: var(--warn); }
  .dim { color: var(--dim); }
  .bar { height: 7px; background: #0b0f14; border-radius: 4px; overflow: hidden;
         border: 1px solid var(--line); }
  .bar > i { display: block; height: 100%; background: var(--accent); }
  .cbfill { background: var(--c4); }
  .wide { grid-column: 1 / -1; }
  .msg { color: var(--warn); }
  footer { color: var(--dim); font-size: 11px; margin-top: 18px; }
  a { color: var(--accent); text-decoration: none; }
  .tag { display: inline-block; padding: 1px 6px; border: 1px solid var(--line);
         border-radius: 4px; color: var(--dim); font-size: 10px; text-transform: uppercase; }
  .tag.arena { color: var(--accent); }
  .tag.play { color: var(--c4); }
  .tag.selfplay { color: var(--warn); }
  .quality { white-space: nowrap; }
  .quality b { color: var(--ok); }
  .bad-text { color: var(--bad); }
  .qbar { display: inline-flex; height: 5px; width: 74px; margin-left: 5px;
          border-radius: 3px; overflow: hidden; background: #0b0f14;
          border: 1px solid var(--line); vertical-align: middle; }
  .qbar i { display: block; height: 100%; }
  .qbar .gen { background: var(--ok); }
  .qbar .rep { background: var(--warn); }
  .qbar .fb { background: var(--bad); }
  .modal { position: fixed; inset: 0; background: rgba(3,7,12,.82); z-index: 40;
           padding: 24px; overflow: auto; }
  .modal-card { max-width: 1450px; margin: 0 auto; background: var(--bg);
                border: 1px solid var(--line); border-radius: 10px;
                box-shadow: 0 20px 80px rgba(0,0,0,.6); }
  .modal-head { display: flex; align-items: center; gap: 12px; padding: 12px 16px;
                 border-bottom: 1px solid var(--line); position: sticky; top: 0;
                 background: var(--bg); z-index: 2; }
  .modal-head h2 { margin: 0; color: var(--text); font-size: 13px;
                   text-transform: none; letter-spacing: 0; }
  .modal-head .grow { flex: 1; }
  .close { background: transparent; color: var(--dim); border: 1px solid var(--line);
           border-radius: 6px; font: inherit; padding: 3px 9px; cursor: pointer; }
  .close:hover { color: var(--text); }
  .modal-body { padding: 16px; }
  .detail-summary { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
  .chip { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
          padding: 7px 10px; min-width: 90px; }
  .chip .k { color: var(--dim); font-size: 10px; text-transform: uppercase; letter-spacing: .6px; }
  .chip .v { font-size: 15px; margin-top: 2px; }
  .detail-grid { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(280px, .75fr);
                 gap: 14px; align-items: start; }
  @media (max-width: 980px) { .detail-grid { grid-template-columns: 1fr; } }
  .board svg { width: 100%; height: auto; display: block; }
  .board-legend { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 8px;
                  font-size: 11px; color: var(--dim); }
  .board-legend span { display: inline-flex; align-items: center; gap: 5px; }
  .board-legend i { width: 10px; height: 10px; border-radius: 50%; display: inline-block;
                    border: 1px solid rgba(255,255,255,.25); }
  .side-a { color: var(--a); }
  .side-b { color: var(--b); }
  .replay-controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
                     margin-top: 10px; }
  .replay-controls button { background: var(--panel); color: var(--text);
                            border: 1px solid var(--line); border-radius: 6px;
                            padding: 4px 9px; font: inherit; cursor: pointer; }
  .replay-controls button:hover { border-color: var(--accent); }
  .replay-controls input[type=range] { flex: 1; min-width: 140px; accent-color: var(--accent); }
  .frame-caption { margin-top: 8px; min-height: 34px; color: var(--text); }
  .frame-caption .sub { color: var(--dim); font-size: 11px; }
  .markers { display: flex; flex-direction: column; gap: 4px; max-height: 280px;
             overflow: auto; }
  .marker { text-align: left; background: transparent; color: var(--text);
            border: 1px solid var(--line); border-radius: 5px; padding: 4px 7px;
            font: inherit; cursor: pointer; }
  .marker:hover { border-color: var(--accent); }
  .marker .dim { font-size: 10px; }
  .quality-table td { padding: 3px 8px 3px 0; }
  .detail-charts { margin-top: 14px; }
  .detail-charts .card { min-height: 226px; }
  .detail-quality { margin-top: 14px; }
  .stack { display: flex; flex-direction: column; gap: 10px; }
  .pill { display: inline-block; padding: 1px 7px; border-radius: 999px;
          background: rgba(88,166,255,.12); color: var(--accent); }
  .pill.b { background: rgba(240,136,62,.14); color: var(--b); }
  .error { color: var(--bad); background: rgba(248,81,73,.08);
           border: 1px solid rgba(248,81,73,.3); border-radius: 8px;
           padding: 12px; margin-bottom: 14px; }
</style>
</head>
<body>
<header>
  <h1>NCM training</h1>
  <nav class="tabs">
    <button id="tab-training" class="tab active" onclick="setTab('training')">Training</button>
    <button id="tab-games" class="tab" onclick="setTab('games')">Games</button>
  </nav>
  <span id="where" class="dim"></span>
  <span class="grow"></span>
  <span id="clock" class="dim"></span>
</header>
<main>
  <section id="training-view">
    <div class="toolbar">
      <select id="runs"></select>
      <span id="badge" class="badge"><i class="dot"></i><span>no run</span></span>
    </div>
    <div id="matchup" class="card wide" style="margin-bottom:14px" hidden></div>
    <div id="kpis"></div>
    <div id="panels" class="grid"></div>
    <div id="extra" class="grid" style="margin-top:14px"></div>
    <div id="report" class="grid" style="margin-top:14px"></div>
    <div id="notes" class="card wide" style="margin-top:14px"></div>
  </section>
  <section id="games-view" hidden>
    <div id="game-kpis"></div>
    <div class="card wide" style="margin-bottom:14px">
      <div class="filters">
        <label>source
          <select id="filter-source">
            <option value="all">all</option>
            <option value="arena">arena</option>
            <option value="play">play</option>
            <option value="selfplay">self-play</option>
          </select>
        </label>
        <label>result
          <select id="filter-result">
            <option value="all">all</option>
            <option value="win">NCM/learner win</option>
            <option value="loss">NCM/learner loss</option>
            <option value="unfinished">unfinished</option>
          </select>
        </label>
        <label>seed <input id="filter-seed" size="8" placeholder="any"></label>
        <label>search <input id="filter-text" size="22" placeholder="file, label, scenario"></label>
        <span class="dim">blue A / orange B · NCM/learner = evaluated model · click a saved game to replay</span>
      </div>
    </div>
    <div id="games-table"></div>
  </section>
  <footer id="foot"></footer>
</main>
<div id="detail" class="modal" hidden>
  <div class="modal-card">
    <div class="modal-head">
      <h2 id="detail-title">game</h2>
      <span class="grow"></span>
      <button class="close" onclick="closeDetail()">close</button>
    </div>
    <div id="detail-body" class="modal-body"></div>
  </div>
</div>
<script>
const STAGE_PANELS = {
  generator: [
    {title: 'Loss', series: [
      {key: 'loss', label: 'train', kinds: ['step']},
      {key: 'loss_ema', label: 'train EMA', kinds: ['step']},
      {key: 'val_loss', label: 'val', kinds: ['epoch']}]},
    {title: 'Generation quality', series: [
      {key: 'exact', label: 'exact', kinds: ['epoch']},
      {key: 'parseable', label: 'parseable', kinds: ['epoch']},
      {key: 'legal_rate', label: 'legal on first try', kinds: ['epoch']}]},
    {title: 'Learning rate', series: [{key: 'lr', label: 'lr', kinds: ['step']}]},
    {title: 'Throughput', series: [{key: 'steps_per_s', label: 'steps/s', kinds: ['step']}]},
    {title: 'GPU utilization', series: [{key: 'gpu_util', label: 'gpu %', kinds: ['step']}]},
    {title: 'VRAM', series: [{key: 'gpu_mem_mb', label: 'vram MiB', kinds: ['step']}]}
  ],
  tokenizer: [
    {title: 'Loss', series: [
      {key: 'loss', label: 'total', kinds: ['step']},
      {key: 'recon', label: 'recon', kinds: ['step']},
      {key: 'vq', label: 'vq', kinds: ['step']}]},
    {title: 'Reconstruction', series: [
      {key: 'val_exact', label: 'exact', kinds: ['epoch']},
      {key: 'val_char', label: 'char', kinds: ['epoch']}]},
    {title: 'Learning rate', series: [{key: 'lr', label: 'lr', kinds: ['step']}]},
    {title: 'GPU utilization', series: [{key: 'gpu_util', label: 'gpu %', kinds: ['step']}]},
    {title: 'VRAM', series: [{key: 'gpu_mem_mb', label: 'vram MiB', kinds: ['step']}]}
  ],
  eval: [
    {title: 'Scores', series: [
      {key: 'exact', label: 'exact'}, {key: 'parseable', label: 'parseable'},
      {key: 'legal_rate', label: 'legal on first try'}]}
  ],
  data: [
    {title: 'Examples', series: [
      {key: 'examples', label: 'examples'}, {key: 'games', label: 'games'}]},
    {title: 'Speed', series: [{key: 'examples_per_s', label: 'examples/s'}]}
  ],
  arena: [
    {title: 'Result per game (learner vs opponent)', series: [
      {key: 'learner_won', label: 'learner win vs opponent (0/1)', kinds: ['step']},
      {key: 'finished', label: 'finished (0/1)', kinds: ['step']},
      {key: 'wins', label: 'wins so far', kinds: ['step']}]},
    {title: 'Score delta', series: [{key: 'score_delta', label: 'learner − opponent', kinds: ['step']}]},
    {title: 'Turns / orders', series: [
      {key: 'turns', label: 'turns', kinds: ['step']},
      {key: 'orders', label: 'orders', kinds: ['step']}]},
    {title: 'NCM order quality', series: [
      {key: 'generated', label: 'generated', kinds: ['step']},
      {key: 'repaired', label: 'repaired', kinds: ['step']},
      {key: 'fallbacks', label: 'fallback', kinds: ['step']}]}
  ],
  vs_ncm: [
    {title: 'Result per game (learner vs opponent)', series: [
      {key: 'learner_won', label: 'learner win vs opponent (0/1)', kinds: ['step']},
      {key: 'finished', label: 'finished (0/1)', kinds: ['step']},
      {key: 'wins', label: 'wins so far', kinds: ['step']}]},
    {title: 'Score delta', series: [{key: 'score_delta', label: 'learner − opponent', kinds: ['step']}]},
    {title: 'Turns / orders', series: [
      {key: 'turns', label: 'turns', kinds: ['step']},
      {key: 'orders', label: 'orders', kinds: ['step']}]},
    {title: 'NCM order quality', series: [
      {key: 'generated', label: 'generated', kinds: ['step']},
      {key: 'repaired', label: 'repaired', kinds: ['step']},
      {key: 'fallbacks', label: 'fallback', kinds: ['step']}]},
    {title: 'Rows generated', series: [{key: 'rows', label: 'rows', kinds: ['step']}]}
  ],
  selfplay: [
    {title: 'Rows / games', series: [
      {key: 'rows', label: 'rows', kinds: ['step']},
      {key: 'game_rows', label: 'this game', kinds: ['step']}]},
    {title: 'NCM order quality', series: [
      {key: 'generated', label: 'generated', kinds: ['step']},
      {key: 'repaired', label: 'repaired', kinds: ['step']},
      {key: 'fallbacks', label: 'fallback', kinds: ['step']}]},
    {title: 'Win (learner vs opponent)', series: [
      {key: 'learner_won', label: 'learner win vs opponent (0/1)', kinds: ['step']},
      {key: 'finished', label: 'finished (0/1)', kinds: ['step']}]}
  ]
};

const STAGE_KPIS = {
  generator: [
    ['progress', 'progress', 'pct'], ['epoch', 'epoch', 'int'], ['step', 'step', 'int'],
    ['loss', 'loss', 'num'], ['val_loss', 'val loss', 'num'], ['exact', 'exact', 'pct'],
    ['parseable', 'parseable', 'pct'], ['legal_rate', 'legal', 'pct'],
    ['best_epoch', 'best epoch', 'int'], ['lr', 'lr', 'sci'],
    ['steps_per_s', 'steps/s', 'num'], ['eta_s', 'eta', 'dur'], ['gpu_mem_mb', 'vram', 'mib']
  ],
  tokenizer: [
    ['progress', 'progress', 'pct'], ['epoch', 'epoch', 'int'], ['step', 'step', 'int'],
    ['loss', 'loss', 'num'], ['recon', 'recon', 'num'], ['vq', 'vq', 'num'],
    ['val_exact', 'val exact', 'pct'], ['val_char', 'val char', 'pct'],
    ['best_epoch', 'best epoch', 'int'], ['lr', 'lr', 'sci'],
    ['eta_s', 'eta', 'dur'], ['gpu_mem_mb', 'vram', 'mib']
  ],
  data: [
    ['progress', 'progress', 'pct'], ['games', 'games', 'int'],
    ['examples', 'examples', 'int'], ['examples_per_s', 'examples/s', 'num'],
    ['eta_s', 'eta', 'dur']
  ],
  eval: [
    ['exact', 'exact', 'pct'], ['parseable', 'parseable', 'pct'],
    ['legal_rate', 'legal', 'pct'], ['ms_per_order', 'ms/order', 'num'], ['n', 'states', 'int']
  ],
  arena: [
    ['progress', 'progress', 'pct'], ['step', 'game', 'int'], ['wins', 'wins', 'int'],
    ['losses', 'losses', 'int'], ['unfinished', 'unfinished', 'int'],
    ['win_rate', 'win rate vs opp', 'pct'], ['finished_rate', 'finished', 'pct'],
    ['avg_score_delta', 'avg score', 'num'], ['avg_turns', 'avg turns', 'num'],
    ['generated', 'generated', 'int'], ['repaired', 'repaired', 'int'],
    ['fallbacks', 'fallbacks', 'int']
  ],
  vs_ncm: [
    ['progress', 'progress', 'pct'], ['step', 'step', 'int'], ['iteration', 'iteration', 'int'],
    ['wins', 'wins', 'int'], ['finished', 'finished games', 'int'],
    ['win_rate', 'win rate vs opp', 'pct'], ['avg_score_delta', 'avg score', 'num'],
    ['avg_turns', 'avg turns', 'num'], ['turns', 'last turns', 'int'],
    ['orders', 'last orders', 'int'], ['generated', 'generated', 'int'],
    ['repaired', 'repaired', 'int'], ['fallbacks', 'fallbacks', 'int']
  ],
  selfplay: [
    ['progress', 'progress', 'pct'], ['step', 'step', 'int'], ['iteration', 'iteration', 'int'],
    ['rows', 'rows', 'int'], ['game_rows', 'this game', 'int'],
    ['generated', 'generated', 'int'], ['repaired', 'repaired', 'int'],
    ['fallbacks', 'fallbacks', 'int'], ['learner_won', 'learner win vs opp (0/1)', 'num']
  ]
};

const COLORS = ['--c1', '--c2', '--c3', '--c4', '--c5'];
const TERRAIN = {
  '.': '#27352b', 'r': '#3a3a3a', 'F': '#1d4728', '#': '#4a4036', 'M': '#5a5048',
  'C': '#454a58', 'f': '#554a3a', 'p': '#334b58', 'A': '#4a4555', 'H': '#5b3838',
  '~': '#1b3550'
};
const SIDE_COLOR = {A: 'var(--a)', B: 'var(--b)'};
const state = {
  tab: 'training', runs: [], id: null, payload: null,
  games: [], reports: [], detail: null, frame: 0, playing: false,
  timer: null, speed: 350, runSignature: '', gamesSignature: null,
  filters: {source: 'all', result: 'all', seed: '', text: ''}
};

const $ = (id) => document.getElementById(id);
const esc = (text) => String(text ?? '')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const attr = (text) => esc(text).replace(/"/g, '&quot;');

function fmt(value, how) {
  if (value === null || value === undefined || !Number.isFinite(+value)) return '—';
  const v = +value;
  if (how === 'pct') return (v * 100).toFixed(1) + '%';
  if (how === 'pct100') return v.toFixed(0) + '%';
  if (how === 'int') return v.toFixed(0);
  if (how === 'sci') return v === 0 ? '0' : v.toExponential(1);
  if (how === 'dur') return dur(v);
  if (how === 'mib') return v.toFixed(0) + ' MiB';
  if (how === 'num') {
    const a = Math.abs(v);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 10) return v.toFixed(2);
    if (a >= 1) return v.toFixed(3);
    return v.toFixed(4);
  }
  return String(value);
}

function dur(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const s = Math.round(seconds);
  if (s < 90) return s + 's';
  const m = Math.round(s / 60);
  if (m < 90) return m + 'm';
  return (m / 60).toFixed(1) + 'h';
}

function merged() {
  const out = {};
  if (!state.payload) return out;
  for (const event of state.payload.events) {
    if (event.best) out.best_epoch = event.epoch;
    for (const [k, v] of Object.entries(event)) {
      if (['t', 'kind', 'stage', 'text'].includes(k)) continue;
      if (v !== null && v !== undefined) out[k] = v;
    }
  }
  const status = state.payload.status || {};
  out.state = status.state;
  out.heartbeat = status.heartbeat;
  if (out.progress === undefined && out.step && state.payload.meta.total_steps) {
    out.progress = out.step / state.payload.meta.total_steps;
  }
  return out;
}

function seriesPoints(key, kinds) {
  const points = [];
  if (!state.payload) return points;
  const stage = state.payload.summary ? state.payload.summary.stage : '';
  // Arena rounds restart their local step counter for baseline/post.  Use the
  // event order for game-oriented stages so every game gets its own x.
  const perGame = ['arena', 'vs_ncm', 'selfplay'].includes(stage);
  let index = 0;
  for (const event of state.payload.events) {
    index += 1;
    if (kinds && !kinds.includes(event.kind)) continue;
    let value = event[key];
    if (typeof value === 'boolean') value = value ? 1 : 0;
    if (value === null || value === undefined || typeof value !== 'number') continue;
    const x = perGame ? index
      : event.kind === 'epoch' && Number.isFinite(+event.epoch) ? +event.epoch
      : Number.isFinite(+event.step) ? +event.step
      : Number.isFinite(+event.games) ? +event.games
      : Number.isFinite(+event.epoch) ? +event.epoch : index;
    points.push([x, value]);
  }
  return points;
}

function drawChart(host, visible, opts = {}) {
  if (!host) return;
  const W = Math.max(320, host.clientWidth || 460), H = 168;
  const padL = 54, padR = 14, padT = 10, padB = 22;
  const all = visible.flatMap((s) => s.points.map((p) => p[1]));
  if (!visible.length || !all.length) {
    host.innerHTML = '<div class="empty">no data yet</div>';
    return;
  }
  const xs = visible.flatMap((s) => s.points.map((p) => p[0]));
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...all), ymax = Math.max(...all);
  if (xmax === xmin) xmax = xmin + 1;
  if (ymax === ymin) { const d = Math.abs(ymin || 1) * 0.1; ymin -= d; ymax += d; }
  const ypad = (ymax - ymin) * 0.08;
  ymin -= ypad; ymax += ypad;
  const X = (x) => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
  const Y = (y) => padT + (1 - (y - ymin) / (ymax - ymin)) * (H - padT - padB);

  let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  for (let i = 0; i <= 3; i++) {
    const value = ymin + (ymax - ymin) * (i / 3);
    const y = Y(value);
    svg += `<line x1="${padL}" x2="${W - padR}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="var(--line)" stroke-width="1"/>`;
    svg += `<text x="${padL - 8}" y="${(y + 3).toFixed(1)}" text-anchor="end">${esc(fmt(value, 'num'))}</text>`;
  }
  for (const frac of [0, 0.5, 1]) {
    const value = xmin + (xmax - xmin) * frac;
    const x = X(value);
    const label = opts.xLabel ? opts.xLabel(value) : fmt(Math.round(value), 'int');
    svg += `<text x="${x.toFixed(1)}" y="${H - 6}" text-anchor="${frac === 0 ? 'start' : frac === 1 ? 'end' : 'middle'}">${esc(label)}</text>`;
  }
  visible.forEach((s, i) => {
    const color = s.color ? `var(${s.color})` : `var(${COLORS[i % COLORS.length]})`;
    const path = s.points.map((p, j) => `${j ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ');
    svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="1.6" stroke-linejoin="round"/>`;
    const last = s.points[s.points.length - 1];
    svg += `<circle cx="${X(last[0]).toFixed(1)}" cy="${Y(last[1]).toFixed(1)}" r="2.4" fill="${color}"/>`;
  });
  svg += '</svg>';
  const legend = visible.map((s, i) => {
    const last = s.points[s.points.length - 1][1];
    const color = s.color ? `var(${s.color})` : `var(${COLORS[i % COLORS.length]})`;
    return `<span><i class="swatch" style="background:${color}"></i>${esc(s.label)} ${esc(fmt(last, 'num'))}</span>`;
  }).join('');
  host.innerHTML = svg + `<div class="legend">${legend}</div>`;
}

function lineChart(host, series) {
  if (!host) return;
  const visible = series.map((s) => ({...s, points: seriesPoints(s.key, s.kinds)}))
    .filter((s) => s.points.length);
  drawChart(host, visible);
}

const MATCH_SETTING_LABELS = {
  update_opponent: 'opponent', teacher: 'teacher', games: 'self-play games',
  eval_games: 'eval games', epochs: 'epochs', learner_side: 'learner side',
  alternate_sides: 'alternate sides', dataset: 'dataset', validation: 'validation'
};

function matchupTags(matchup) {
  const s = matchup.settings || {};
  const tags = [];
  if ('update_opponent' in s) {
    tags.push(s.update_opponent
      ? '<span class="tag play">opponent updates</span>'
      : '<span class="tag">frozen opponent</span>');
  }
  if (matchup.has_opponent && s.learner_side) {
    tags.push(`<span class="tag">side ${esc(s.learner_side)}${s.alternate_sides ? ' · alternating' : ''}</span>`);
  }
  if (s.teacher) tags.push('<span class="tag selfplay">teacher rows</span>');
  if (s.games !== undefined && s.games !== null && s.games !== '') {
    tags.push(`<span class="tag">${esc(s.games)} games</span>`);
  }
  if (s.eval_games !== undefined && s.eval_games !== null && s.eval_games !== '') {
    tags.push(`<span class="tag arena">${esc(s.eval_games)} eval</span>`);
  }
  if (s.epochs !== undefined && s.epochs !== null && s.epochs !== '') {
    tags.push(`<span class="tag">${esc(s.epochs)} epochs</span>`);
  }
  const handled = ['update_opponent', 'teacher', 'games', 'eval_games', 'epochs',
                   'learner_side', 'alternate_sides'];
  for (const [key, value] of Object.entries(s)) {
    if (handled.includes(key) || value === null || value === undefined || value === '') continue;
    tags.push(`<span class="tag">${esc(MATCH_SETTING_LABELS[key] || key.replace(/_/g, ' '))}: ${esc(value)}</span>`);
  }
  return tags.join('');
}

function renderMatchup() {
  const host = $('matchup');
  if (!host) return;
  const matchup = state.payload ? state.payload.matchup : null;
  if (!matchup) {
    host.hidden = true;
    host.innerHTML = '';
    return;
  }
  host.hidden = false;
  const learner = matchup.learner || {};
  const opponent = matchup.opponent || {};
  const tags = matchupTags(matchup);
  let body;
  if (matchup.has_opponent && learner.path && opponent.path) {
    body = `<div class="matchup-row">
      <div class="match-side">
        <div class="k">learner</div>
        <div class="v side-a">${esc(learner.label || 'Learner NCM')}</div>
        <div class="path" title="${attr(learner.path)}">${esc(learner.short || learner.path)}</div>
      </div>
      <div class="match-vs">vs</div>
      <div class="match-side">
        <div class="k">opponent</div>
        <div class="v side-b">${esc(opponent.label || 'Opponent')}</div>
        <div class="path" title="${attr(opponent.path)}">${esc(opponent.short || opponent.path)}</div>
      </div>
      <div class="match-tags">${tags}</div>
    </div>
    <div class="match-note">${esc(matchup.description || '')}</div>`;
  } else {
    body = `<div>${esc(matchup.description || 'This run has no live opponent.')}</div>` +
      (tags ? `<div class="match-tags" style="margin-top:8px;margin-left:0">${tags}</div>` : '');
  }
  host.innerHTML = `<h2>${matchup.has_opponent ? 'Match-up' : 'Run type'}</h2>${body}`;
}

function renderKpis() {
  const data = merged();
  const stage = state.payload ? state.payload.summary.stage : null;
  let spec = STAGE_KPIS[stage];
  if (!spec) {
    spec = Object.keys(data)
      .filter((k) => typeof data[k] === 'number' && !['t', 'step', 'epoch', 'heartbeat'].includes(k))
      .slice(0, 12)
      .map((k) => [k, k.replace(/_/g, ' '), 'num']);
  }
  $('kpis').innerHTML = spec.map(([key, label, how]) =>
    `<div class="kpi"><div class="k">${esc(label)}</div><div class="v">${fmt(data[key], how)}</div></div>`
  ).join('');
}

function renderPanels() {
  const stage = state.payload ? state.payload.summary.stage : null;
  const panels = STAGE_PANELS[stage] || [];
  const cards = panels.map((panel, i) => {
    const hasData = panel.series.some((series) => seriesPoints(series.key, series.kinds).length);
    if (!hasData) return '';
    return `<div class="card"><h2>${esc(panel.title)}</h2><div id="chart-${i}"></div></div>`;
  }).join('');
  $('panels').innerHTML = cards || '<div class="card wide"><div class="empty">no metric panels for this run type</div></div>';
  panels.forEach((panel, i) => {
    const host = $('chart-' + i);
    if (host) lineChart(host, panel.series);
  });
}

function diffRow(target, generated) {
  const a = String(target ?? ''), b = String(generated ?? '');
  const n = Math.max(a.length, b.length);
  let out = '';
  for (let i = 0; i < n; i++) {
    const want = a[i] ?? '', got = b[i] ?? '';
    if (want === got) out += `<span class="dim">${esc(got)}</span>`;
    else out += `<span class="bad">${esc(got === ' ' ? '·' : got) || '·'}</span>`;
  }
  return out;
}

function renderSamples() {
  const samples = (state.payload && state.payload.samples) || [];
  const rows = samples.slice().reverse().slice(0, 12);
  const stage = state.payload ? state.payload.summary.stage : null;
  if (!rows.length && stage !== 'tokenizer' && stage !== 'generator') {
    $('extra').innerHTML = '';
    return;
  }
  const codebook = renderCodebook();
  const content = rows.length
    ? `<table><thead><tr><th>epoch</th><th>target</th><th>generated</th></tr></thead><tbody>${
        rows.map((s) => `<tr><td class="dim">${esc(s.epoch ?? '')}</td>
          <td class="mono dim">${esc(s.target ?? '')}</td>
          <td class="mono">${diffRow(s.target, s.generated)}</td></tr>`).join('')}</tbody></table>`
    : '<div class="empty">no generated samples yet (they appear after the first validation pass)</div>';
  $('extra').innerHTML = `<div class="card ${codebook ? '' : 'wide'}"><h2>Samples (red = wrong character)</h2>${content}</div>${codebook}`;
}

function renderCodebook() {
  const stage = state.payload ? state.payload.summary.stage : null;
  if (stage !== 'tokenizer') return '';
  const data = merged();
  const active = Array.isArray(data.active_codes) ? data.active_codes : null;
  if (!active) return '';
  const ppl = Array.isArray(data.perplexity) ? data.perplexity : [];
  const total = 4096;
  const rows = active.map((count, i) => {
    const pct = Math.min(100, (count / total) * 100);
    return `<tr><td class="dim">L${i}</td>
      <td style="width:60%"><div class="bar"><i class="cbfill" style="width:${pct.toFixed(2)}%"></i></div></td>
      <td>${esc(count)}/4096</td><td class="dim">${ppl[i] !== undefined ? 'ppl ' + esc(+ppl[i]).toFixed(0) : ''}</td></tr>`;
  }).join('');
  return `<div class="card"><h2>Codebook usage per level</h2><table>${rows}</table></div>`;
}

function renderNotes() {
  const events = (state.payload && state.payload.events) || [];
  const notes = events.filter((event) => event.kind === 'note' || event.kind === 'final').slice(-6);
  if (!notes.length) { $('notes').innerHTML = '<h2>Notes</h2><div class="empty">nothing logged yet</div>'; return; }
  $('notes').innerHTML = '<h2>Notes</h2><table>' + notes.map((event) =>
    `<tr><td class="dim">${esc(new Date((event.t || 0) * 1000).toLocaleTimeString())}</td><td>${
      esc(event.text || event.kind)}</td><td class="dim">${
      Object.entries(event).filter(([k, v]) => v && typeof v === 'number' && !['t', 'step', 'epoch'].includes(k))
        .slice(0, 4).map(([k, v]) => k + ' ' + fmt(v, 'num')).join('  ')}</td></tr>`).join('') + '</table>';
}

function reportRecord(summary) {
  if (!summary || !Object.keys(summary).length) return '<span class="dim">—</span>';
  const wins = Number(summary.wins) || 0;
  const losses = Number(summary.losses) || 0;
  const unfinished = Number(summary.unfinished) || 0;
  return `<span class="mono">${wins}W / ${losses}L / ${unfinished}U</span>
    <div class="dim">win ${esc(fmt(summary.win_rate, 'pct'))} · score ${esc(fmt(summary.avg_score_delta, 'num'))}</div>`;
}

function renderReport() {
  const report = state.payload && state.payload.report;
  if (!report || !Array.isArray(report.iterations) || !report.iterations.length) {
    $('report').innerHTML = '';
    return;
  }
  const matchup = (state.payload && state.payload.matchup) || {};
  const opponent = matchup.opponent || {};
  const updates = !!(matchup.settings && matchup.settings.update_opponent);
  const opponentLine = opponent.label
    ? `Opponent: <b>${esc(opponent.label)}</b>` +
      (opponent.short ? ` <span class="dim">(${esc(opponent.short)})</span>` : '') +
      (updates
        ? ' — <span class="warn">changes every iteration</span>'
        : ' — <span class="dim">same opponent in every iteration</span>')
    : '';
  const rows = report.iterations.map((iteration) => {
    const baseline = iteration.baseline || {};
    const post = iteration.post || {};
    const deltaScore = Number.isFinite(+iteration.delta_score) ? (+iteration.delta_score).toFixed(2) : '&ndash;';
    const deltaWin = Number.isFinite(+iteration.delta_win_rate) ? (iteration.delta_win_rate * 100).toFixed(1) + '%' : '&ndash;';
    return `<tr>
      <td>${esc(iteration.iteration)}</td>
      <td>${reportRecord(baseline)}</td>
      <td>${reportRecord(post)}</td>
      <td class="${Number(iteration.delta_score) < 0 ? 'bad-text' : 'ok'}">${esc(deltaScore)}</td>
      <td>${esc(deltaWin)}</td>
      <td class="dim">${esc(iteration.train_rows ?? '')}</td>
    </tr>`;
  }).join('');
  $('report').innerHTML = `<div class="card wide"><h2>Iterations — ${esc(report.run || '')}</h2>
    ${opponentLine ? `<div class="dim" style="margin-bottom:8px">${opponentLine}</div>` : ''}
    <table><thead><tr><th>iter</th><th>before (learner vs opponent)</th><th>after (learner vs opponent)</th>
    <th>Δ score</th><th>Δ win rate</th><th>train rows</th></tr></thead><tbody>${rows}</tbody></table>
    <div class="dim" style="margin-top:8px">${esc(report.path || '')}</div></div>`;
}

function renderBadge() {
  const badge = $('badge');
  const summary = state.payload ? state.payload.summary : null;
  if (!summary) {
    badge.className = 'badge';
    badge.innerHTML = '<i class="dot"></i><span>no run</span>';
    return;
  }
  const age = summary.heartbeat_age_s;
  const status = summary.state || 'unknown';
  let cls = 'badge', text = status;
  if (status === 'running') {
    if (age !== null && age > 30) { cls += ' stalled'; text = `stalled ${dur(age)}`; }
    else { cls += ' live'; text = 'live'; }
  } else if (status === 'crashed') cls += ' crashed';
  badge.className = cls;
  badge.innerHTML = `<i class="dot"></i><span>${esc(text)}</span>`;
}

async function getJSON(url) {
  const response = await fetch(url, {cache: 'no-store'});
  if (!response.ok) throw new Error(url + ' ' + response.status);
  return response.json();
}

function renderRunSelect() {
  const select = $('runs');
  const signature = state.runs.map((run) => run.id + ':' + run.state).join('|');
  if (signature !== state.runSignature) {
    select.innerHTML = state.runs.map((run) => {
      const last = run.last || {};
      const detail = [
        run.stage,
        run.matchup && run.matchup.summary ? run.matchup.summary : '',
        run.state,
        Number.isFinite(+last.exact) ? 'exact ' + (last.exact * 100).toFixed(1) + '%' : '',
        Number.isFinite(+last.win_rate) ? 'win ' + (last.win_rate * 100).toFixed(0) + '%' : '',
        Number.isFinite(+last.avg_score_delta) ? 'score ' + fmt(last.avg_score_delta, 'num') : '',
        Number.isFinite(+last.loss) ? 'loss ' + fmt(last.loss, 'num') : ''
      ].filter(Boolean).join(' · ');
      return `<option value="${attr(run.id)}">${esc(run.id)} — ${esc(detail)}</option>`;
    }).join('');
    state.runSignature = signature;
  }
  if (state.id && state.runs.some((run) => run.id === state.id)) select.value = state.id;
}

async function pollTraining() {
  try {
    const runs = await getJSON('/api/runs');
    state.runs = runs.runs || [];
    if (!state.id || !state.runs.some((run) => run.id === state.id)) {
      const stored = localStorage.getItem('tbs-ncm-run');
      state.id = state.runs.some((run) => run.id === stored) ? stored : (state.runs[0] || {}).id || null;
    }
    renderRunSelect();
    if (state.id) {
      localStorage.setItem('tbs-ncm-run', state.id);
      state.payload = await getJSON('/api/run/' + encodeURIComponent(state.id));
      if (state.payload.error) {
        $('clock').innerHTML = `<span class="msg">run ${esc(state.id)}: ${esc(state.payload.error)}</span>`;
        return;
      }
      $('where').textContent = state.id;
      renderBadge();
      renderMatchup();
      renderKpis();
      renderPanels();
      renderSamples();
      renderReport();
      renderNotes();
    } else {
      state.payload = null;
      $('where').textContent = runs.runs_dir || '';
      renderBadge();
      renderMatchup();
      $('kpis').innerHTML = '';
      $('panels').innerHTML = `<div class="card wide"><div class="empty">
        no runs yet — start one with <code>python -m tbs_ncm.train_generator</code>
        (or <code>train_tokenizer</code>, <code>generate_data</code>).<br>
        logs land in ${esc(runs.runs_dir || 'artifacts/tbs/runs')}</div></div>`;
      $('extra').innerHTML = '';
      $('report').innerHTML = '';
      $('notes').innerHTML = '';
    }
    $('clock').textContent = new Date().toLocaleTimeString() + ' · refreshing 2s';
  } catch (error) {
    $('clock').innerHTML = `<span class="msg">disconnected: ${esc(error.message)}</span>`;
  }
}

function qualityStats(row) {
  const q = row.quality || {};
  const learner = q.learner || row.learner || {};
  const gen = Number(learner.generated) || 0;
  const rep = Number(learner.repaired) || 0;
  const fb = Number(learner.fallbacks) || 0;
  const ill = Number(learner.illegal) || 0;
  const total = gen + rep + fb;
  return {gen, rep, fb, ill, total, pct: total ? gen / total : null};
}

function qualityCell(row) {
  const q = qualityStats(row);
  if (!q.total) return '<span class="dim">—</span>';
  const pct = (q.pct * 100).toFixed(0);
  const g = (q.gen / q.total * 100).toFixed(0);
  const r = (q.rep / q.total * 100).toFixed(0);
  const f = (q.fb / q.total * 100).toFixed(0);
  return `<span class="quality"><b>${pct}%</b> <span class="dim">gen</span>
    <span class="qbar"><i class="gen" style="width:${g}%"></i><i class="rep" style="width:${r}%"></i><i class="fb" style="width:${f}%"></i></span>
    <span class="dim">${q.gen}/${q.rep}/${q.fb}</span></span>`;
}

function sideName(row, side) {
  return (row.side_labels && row.side_labels[side]) || side;
}

function whenText(row) {
  if (!row) return '';
  if (row.finished_at) return String(row.finished_at);
  const stamp = +row.mtime;
  if (!Number.isFinite(stamp) || stamp <= 0) return '';
  const date = new Date(stamp * 1000);
  const pad = (value) => String(value).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function whenCell(row) {
  const text = whenText(row);
  if (!text) return '<span class="dim">—</span>';
  const parts = text.split(' ');
  if (parts.length < 2) return esc(text);
  return `<div>${esc(parts[0])}</div><div class="dim">${esc(parts.slice(1).join(' '))}</div>`;
}

function matchupText(row) {
  return `A · ${sideName(row, 'A')} vs B · ${sideName(row, 'B')}`;
}

function resultCell(row) {
  const label = row.result_label || (
    row.learner_won ? 'NCM/learner win'
      : row.finished ? 'NCM/learner loss' : 'unfinished'
  );
  const winner = row.winner_label ? `<div class="dim">winner ${esc(row.winner_label)}</div>` : '';
  if (label === 'unfinished') return '<span class="dim">unfinished</span>';
  let cls = 'dim';
  if (row.learner_won) cls = 'ok';
  else if (/loss/i.test(label)) cls = 'bad-text';
  return `<div class="${cls}">${esc(label)}</div>${winner}`;
}

function sideCell(row) {
  const learner = row.learner_side || row.ncm_side;
  const labels = row.side_labels || {A: 'A', B: 'B'};
  const parts = ['A', 'B'].map((side) => {
    const cls = side === 'A' ? 'side-a' : 'side-b';
    const mark = side === learner ? ' style="font-weight:700"' : '';
    return `<span class="${cls}"${mark}>${side}</span> · ${esc(labels[side] || side)}`;
  });
  return parts.join('<span class="dim"> vs </span>');
}

function updateResultFilterLabels() {
  const source = state.filters.source;
  const who = source === 'play' ? 'NCM' : source === 'arena' || source === 'selfplay'
    ? 'Learner' : 'NCM/learner';
  const win = document.querySelector('#filter-result option[value="win"]');
  const loss = document.querySelector('#filter-result option[value="loss"]');
  if (win) win.textContent = `${who} win`;
  if (loss) loss.textContent = `${who} loss`;
}

function filteredGames() {
  const f = state.filters;
  return state.games.filter((row) => {
    if (f.source !== 'all' && row.source !== f.source) return false;
    if (f.result === 'win' && row.learner_won !== true) return false;
    if (f.result === 'loss' && (!row.finished || row.learner_won !== false)) return false;
    if (f.result === 'unfinished' && row.finished) return false;
    if (f.seed && !String(row.seed ?? '').includes(f.seed)) return false;
    if (f.text) {
      const hay = `${row.id || ''} ${row.file || ''} ${row.label || ''} ${row.scenario || ''} ${whenText(row)}`.toLowerCase();
      if (!hay.includes(f.text.toLowerCase())) return false;
    }
    return true;
  });
}

function renderGameKpis(rows) {
  const total = rows.length;
  const finished = rows.filter((row) => row.finished).length;
  const wins = rows.filter((row) => row.learner_won).length;
  const withQuality = rows.map(qualityStats).filter((q) => q.total);
  const gen = withQuality.reduce((sum, q) => sum + q.gen, 0);
  const rep = withQuality.reduce((sum, q) => sum + q.rep, 0);
  const fb = withQuality.reduce((sum, q) => sum + q.fb, 0);
  const orders = gen + rep + fb;
  const scores = rows.filter((row) => Number.isFinite(+row.score_delta));
  const avgScore = scores.length ? scores.reduce((sum, row) => sum + (+row.score_delta), 0) / scores.length : null;
  const replayable = rows.filter((row) => row.has_replay).length;
  const source = state.filters.source;
  const who = source === 'play' ? 'NCM' : source === 'arena' || source === 'selfplay'
    ? 'Learner' : 'NCM/learner';
  const cards = [
    ['games', total, 'int'],
    ['finished', finished, 'int'],
    [`${who} wins`, wins, 'int'],
    [`${who} win rate`, finished ? wins / finished : null, 'pct'],
    ['avg score Δ', avgScore, 'num'],
    ['replayable', replayable, 'int'],
    ['generated', orders ? gen / orders : null, 'pct'],
    ['fallback', orders ? fb / orders : null, 'pct']
  ];
  $('game-kpis').innerHTML = cards.map(([label, value, how]) =>
    `<div class="kpi"><div class="k">${esc(label)}</div><div class="v">${fmt(value, how)}</div></div>`
  ).join('');
}

function renderGamesTable() {
  const rows = filteredGames();
  renderGameKpis(rows);
  if (!rows.length) {
    $('games-table').innerHTML = '<div class="card wide"><div class="empty">no games match these filters</div></div>';
    return;
  }
  const body = rows.map((row) => {
    const file = row.file;
    const disabled = file ? '' : ' disabled';
    const source = row.source || 'play';
    const quality = qualityCell(row);
    const score = Number.isFinite(+row.score_delta) ? fmt(row.score_delta, 'num') : '—';
    const label = row.label || row.id || '';
    const detail = [file || '', row.arena_phase || ''].filter(Boolean).join(' · ');
    return `<tr class="game-row${disabled}" ${file ? `data-file="${attr(file)}"` : ''} title="${attr(matchupText(row))}">
      <td><div>${esc(label)}</div><div class="dim">${esc(detail)}</div></td>
      <td class="when">${whenCell(row)}</td>
      <td>${esc(row.seed ?? '')}</td>
      <td class="matchup">${sideCell(row)}</td>
      <td class="result-cell">${resultCell(row)}</td>
      <td>${esc(row.turns ?? '')}</td>
      <td>${esc(score)}</td>
      <td>${quality}</td>
      <td><span class="tag ${attr(source)}">${esc(source)}</span></td>
      <td>${row.has_replay ? '<span class="ok">open</span>' : '<span class="dim">—</span>'}</td>
    </tr>`;
  }).join('');
  $('games-table').innerHTML = `<div class="card wide"><h2>Games (${rows.length})</h2>
    <table><thead><tr><th>game</th><th>played</th><th>seed</th><th>matchup</th><th>result</th><th>turns</th>
    <th>score Δ</th><th>NCM quality (gen/rep/fb)</th><th>source</th><th>replay</th></tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

async function pollGames() {
  try {
    const data = await getJSON('/api/games');
    state.games = data.games || [];
    state.reports = data.reports || [];
    const signature = state.games.map((row) => {
      const q = qualityStats(row);
      return `${row.id}:${row.source}:${row.status}:${row.finished}:${row.turns}:${row.score_delta}:${whenText(row)}:${q.gen}/${q.rep}/${q.fb}`;
    }).join('|');
    if (signature !== state.gamesSignature) {
      state.gamesSignature = signature;
      renderGamesTable();
    }
    $('clock').textContent = new Date().toLocaleTimeString() + ' · refreshing 2s';
  } catch (error) {
    $('clock').innerHTML = `<span class="msg">disconnected: ${esc(error.message)}</span>`;
  }
}

function setTab(tab) {
  state.tab = tab;
  $('tab-training').classList.toggle('active', tab === 'training');
  $('tab-games').classList.toggle('active', tab === 'games');
  $('training-view').hidden = tab !== 'training';
  $('games-view').hidden = tab !== 'games';
  if (tab === 'training') pollTraining();
  else pollGames();
}

function closeDetail() {
  stopReplay();
  state.detail = null;
  $('detail').hidden = true;
  $('detail-body').innerHTML = '';
  document.body.classList.remove('modal-open');
}

async function openGame(file) {
  if (!file) return;
  closeDetail();
  document.body.classList.add('modal-open');
  $('detail').hidden = false;
  $('detail-body').innerHTML = '<div class="empty">loading replay…</div>';
  $('detail-title').textContent = file;
  try {
    const payload = await getJSON('/api/match/' + encodeURIComponent(file));
    state.detail = payload;
    state.frame = 0;
    renderDetail();
    const requested = new URLSearchParams(location.search).get('frame');
    if (requested !== null && Number.isFinite(+requested)) setFrame(+requested);
  } catch (error) {
    $('detail-body').innerHTML = `<div class="error">cannot load ${esc(file)}: ${esc(error.message)}</div>`;
  }
}

function renderDetail() {
  const payload = state.detail;
  if (!payload || payload.error) {
    $('detail-body').innerHTML = `<div class="error">${esc(payload ? payload.error : 'no game')}</div>`;
    return;
  }
  const match = payload.match || {};
  const learner = match.learner_side || match.ncm_side;
  const learnerLabel = match.source === 'play' ? 'NCM side' : 'learner side';
  $('detail-title').textContent = `${match.file || match.id || ''} · ${matchupText(match)} · ${payload.scenario || ''} · seed ${payload.seed ?? ''}`;
  const replay = payload.replay_matches === false ? '<span class="bad">replay differs</span>'
    : payload.replay_matches ? '<span class="ok">replay verified</span>' : '<span class="dim">replay unchecked</span>';
  $('detail-body').innerHTML = `
    <div class="detail-summary">
      <div class="chip"><div class="k">result</div><div class="v">${resultCell(match)}</div></div>
      <div class="chip"><div class="k">played</div><div class="v">${whenText(match) ? esc(whenText(match)) : '—'}</div></div>
      ${Number.isFinite(+match.duration_s) ? `<div class="chip"><div class="k">duration</div><div class="v">${esc(dur(+match.duration_s))}</div></div>` : ''}
      <div class="chip"><div class="k">matchup</div><div class="v">${sideCell(match)}</div></div>
      ${match.opponent_label ? `<div class="chip"><div class="k">opponent</div><div class="v">${esc(match.opponent_label)}</div></div>` : ''}
      ${learner ? `<div class="chip"><div class="k">${learnerLabel}</div><div class="v"><span class="${learner === 'A' ? 'side-a' : 'side-b'}">${esc(learner)}</span> · ${esc(sideName(match, learner))}</div></div>` : ''}
      ${match.winner_label ? `<div class="chip"><div class="k">winner</div><div class="v ok">${esc(match.winner_label)}</div></div>` : ''}
      <div class="chip"><div class="k">turns</div><div class="v">${esc(match.turns ?? '')}</div></div>
      <div class="chip"><div class="k">orders</div><div class="v">${esc(payload.n_orders)}</div></div>
      ${match.arena_phase ? `<div class="chip"><div class="k">arena phase</div><div class="v">${esc(match.arena_phase)}</div></div>` : ''}
      <div class="chip"><div class="k">score Δ</div><div class="v">${Number.isFinite(+match.score_delta) ? esc(fmt(match.score_delta, 'num')) : '—'}</div></div>
      <div class="chip"><div class="k">replay</div><div class="v">${replay}</div></div>
    </div>
    <div class="detail-grid">
      <div class="card">
        <h2>Board replay</h2>
        <div id="board-legend" class="board-legend"></div>
        <div id="board" class="board"></div>
        <div class="replay-controls">
          <button onclick="setFrame(0)">⏮</button>
          <button onclick="stepFrame(-1)">◀</button>
          <button id="play-button" onclick="toggleReplay()">▶ play</button>
          <button onclick="stepFrame(1)">▶</button>
          <button onclick="setFrame(${payload.frames.length - 1})">⏭</button>
          <input id="frame-slider" type="range" min="0" max="${payload.frames.length - 1}" value="0" oninput="setFrame(+this.value)">
          <select id="replay-speed" onchange="state.speed=+this.value">
            <option value="700">slow</option>
            <option value="350" selected>normal</option>
            <option value="150">fast</option>
            <option value="60">very fast</option>
          </select>
        </div>
        <div id="frame-caption" class="frame-caption"></div>
      </div>
      <div class="card">
        <h2>Actions & events</h2>
        <div id="markers" class="markers"></div>
        <div class="dim" style="margin-top:8px">action descriptions: ${esc(payload.n_orders)}</div>
      </div>
    </div>
    <div id="detail-charts" class="grid detail-charts"></div>
    <div id="detail-quality" class="card wide detail-quality"></div>
  `;
  renderBoardLegend();
  renderBoard();
  renderDetailCharts();
  renderQuality();
  renderMarkers();
}

function renderBoardLegend() {
  const host = $('board-legend');
  const payload = state.detail;
  if (!host || !payload) return;
  const match = payload.match || {};
  const labels = match.side_labels || {A: 'A', B: 'B'};
  host.innerHTML = ['A', 'B'].map((side) => {
    const color = side === 'A' ? 'var(--a)' : 'var(--b)';
    return `<span><i style="background:${color}"></i><span class="${side === 'A' ? 'side-a' : 'side-b'}">${side}</span> · ${esc(labels[side] || side)}</span>`;
  }).join('');
}

function renderBoard() {
  const payload = state.detail;
  if (!payload || !payload.frames || !payload.frames.length) return;
  const index = Math.max(0, Math.min(state.frame, payload.frames.length - 1));
  state.frame = index;
  const frame = payload.frames[index];
  const tiles = payload.tiles || [];
  const size = 24;
  const pad = size * 1.1;
  const points = new Map();
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const tile of tiles) {
    const q = tile.pos[0], r = tile.pos[1];
    const x = size * Math.sqrt(3) * (q + r / 2);
    const y = size * 1.5 * r;
    points.set(`${q},${r}`, [x, y]);
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const width = (maxX - minX) + pad * 2, height = (maxY - minY) + pad * 2;
  const offsetX = -minX + pad, offsetY = -minY + pad;
  const corners = (cx, cy) => {
    const out = [];
    for (let i = 0; i < 6; i++) {
      const angle = Math.PI / 180 * (60 * i + 30);
      out.push(`${(cx + size * Math.cos(angle)).toFixed(1)},${(cy + size * Math.sin(angle)).toFixed(1)}`);
    }
    return out.join(' ');
  };
  const owners = String(frame.owners || '');
  let svg = `<svg viewBox="0 0 ${width.toFixed(0)} ${height.toFixed(0)}" preserveAspectRatio="xMidYMid meet">`;
  tiles.forEach((tile, tileIndex) => {
    const [q, r] = tile.pos;
    const [x, y] = points.get(`${q},${r}`);
    const cx = x + offsetX, cy = y + offsetY;
    const owner = owners[tileIndex] && owners[tileIndex] !== '.' ? owners[tileIndex] : null;
    const terrain = TERRAIN[tile.terrain] || '#333b45';
    const ownerColor = owner ? SIDE_COLOR[owner] || 'var(--dim)' : '#26303b';
    svg += `<polygon points="${corners(cx, cy)}" fill="${terrain}" stroke="${ownerColor}" stroke-width="${owner ? 2.4 : 1}" />`;
    if (tile.objective) {
      svg += `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${(size * 0.13).toFixed(1)}" fill="none" stroke="var(--warn)" stroke-width="2"/>`;
    }
    if (tile.capture_points) {
      svg += `<circle cx="${(cx + size * 0.55).toFixed(1)}" cy="${(cy - size * 0.55).toFixed(1)}" r="2" fill="${ownerColor}"/>`;
    }
  });
  const groups = new Map();
  for (const unit of frame.units || []) {
    const [id, owner, type, q, r, hp, acted, loaded, deployed] = unit;
    const key = `${q},${r}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({id, owner, type, q, r, hp, acted, loaded, deployed});
  }
  const action = frame.action || {};
  const sideLabels = (payload.match && payload.match.side_labels) || {A: 'A', B: 'B'};
  const highlighted = new Set([action.unit, action.target].filter(Boolean));
  for (const [key, units] of groups) {
    const [x, y] = points.get(key);
    const baseX = x + offsetX, baseY = y + offsetY;
    const stacked = units.length > 1;
    const air = units.filter((unit) => (payload.unit_meta[unit.type] || {}).domain === 'air');
    const surface = units.filter((unit) => (payload.unit_meta[unit.type] || {}).domain !== 'air');
    units.forEach((unit, index) => {
      let cx = baseX, cy = baseY;
      if (stacked && air.length === 1 && surface.length === 1) {
        cx = baseX;
        cy = baseY + (air[0] === unit ? -size * 0.22 : size * 0.22);
      } else if (stacked) {
        const angle = 2 * Math.PI * index / units.length;
        cx = baseX + size * 0.28 * Math.cos(angle);
        cy = baseY + size * 0.28 * Math.sin(angle);
      }
      const meta = payload.unit_meta[unit.type] || {name: unit.type, glyph: '?', max_hp: 100};
      const radius = stacked ? size * 0.32 : size * 0.40;
      const color = SIDE_COLOR[unit.owner] || 'var(--dim)';
      const opacity = unit.acted ? 0.48 : 1;
      const ring = highlighted.has(unit.id) ? ` stroke="var(--warn)" stroke-width="2.5"` : ' stroke="#0d1117" stroke-width="1.5"';
      svg += `<g opacity="${opacity}"><title>${esc(unit.id)} ${esc(meta.name)} · ${esc(unit.owner)} · ${esc(sideLabels[unit.owner] || unit.owner)} · ${unit.hp} HP</title>`;
      svg += `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${radius.toFixed(1)}" fill="${color}"${ring}/>`;
      svg += `<text x="${cx.toFixed(1)}" y="${(cy + size * 0.16).toFixed(1)}" text-anchor="middle" font-size="${(size * 0.72).toFixed(1)}" fill="#f5f5f5">${esc(meta.glyph)}</text>`;
      const barW = radius * 2, barX = cx - radius, barY = cy + radius + 2.5;
      const frac = Math.max(0, Math.min(1, unit.hp / (meta.max_hp || 100)));
      svg += `<rect x="${barX.toFixed(1)}" y="${barY.toFixed(1)}" width="${barW.toFixed(1)}" height="3" rx="1.5" fill="#0b0f14"/>`;
      svg += `<rect x="${barX.toFixed(1)}" y="${barY.toFixed(1)}" width="${(barW * frac).toFixed(1)}" height="3" rx="1.5" fill="${unit.hp > 50 ? 'var(--ok)' : unit.hp > 25 ? 'var(--warn)' : 'var(--bad)'}"/>`;
      svg += '</g>';
    });
  }
  svg += '</svg>';
  const host = $('board');
  if (host) host.innerHTML = svg;
  const slider = $('frame-slider');
  if (slider) slider.value = index;
  const caption = $('frame-caption');
  if (caption) {
    const describe = frame.describe || (frame.action && frame.action.action) || 'start';
    const playerName = sideLabels[frame.player] || '';
    caption.innerHTML = `<b>T${esc(frame.turn)} ${esc(frame.player || '')}${playerName ? ' · ' + esc(playerName) : ''}</b> — ${esc(describe)}
      <div class="sub">frame ${index + 1}/${payload.frames.length} · funds ${esc(sideLabels.A || 'A')} ${esc(frame.funds?.A ?? '')} / ${esc(sideLabels.B || 'B')} ${esc(frame.funds?.B ?? '')}</div>`;
  }
  const button = $('play-button');
  if (button) button.textContent = state.playing ? '⏸ pause' : '▶ play';
}

function renderMarkers() {
  const payload = state.detail;
  const host = $('markers');
  if (!host) return;
  const markers = (payload.markers || []).slice(-30).reverse();
  if (!markers.length) {
    host.innerHTML = '<div class="empty">no capture, kill or objective events recorded</div>';
    return;
  }
  const sideLabels = (payload.match && payload.match.side_labels) || {};
  host.innerHTML = markers.map((marker) => {
    const player = marker.player || '';
    const playerName = sideLabels[player] ? ' · ' + sideLabels[player] : '';
    return `<button class="marker" onclick="jumpTo(${marker.seq})">
      <div>${esc(marker.text)}</div>
      <div class="dim">T${esc(marker.turn)} · ${esc(player)}${esc(playerName)} · ${esc(marker.type)}</div>
    </button>`;
  }).join('');
}

function timelineSeries(key) {
  const timeline = (state.detail && state.detail.timeline) || [];
  return {
    A: timeline.map((entry, index) => [index, entry.A ? entry.A[key] : null])
      .filter((point) => point[1] !== null && point[1] !== undefined),
    B: timeline.map((entry, index) => [index, entry.B ? entry.B[key] : null])
      .filter((point) => point[1] !== null && point[1] !== undefined)
  };
}

function renderDetailCharts() {
  const payload = state.detail;
  const host = $('detail-charts');
  if (!host || !payload) return;
  const timeline = payload.timeline || [];
  const xLabel = (value) => {
    const entry = timeline[Math.round(value)];
    return entry ? `T${entry.turn}${entry.player || ''}` : '';
  };
  const chartDefs = [
    ['Board score', 'score'],
    ['Units', 'units'],
    ['Total HP', 'hp'],
    ['Sites owned', 'sites'],
    ['Funds', 'funds']
  ];
  const labels = (payload.match && payload.match.side_labels) || {A: 'A', B: 'B'};
  host.innerHTML = chartDefs.map(([title], index) =>
    `<div class="card"><h2>${esc(title)}</h2><div id="dchart-${index}"></div></div>`).join('')
    + '<div class="card"><h2>NCM order quality per turn</h2><div id="dchart-quality"></div></div>';
  chartDefs.forEach(([title, key], index) => {
    const series = timelineSeries(key);
    drawChart($('dchart-' + index), [
      {label: `A · ${labels.A || 'A'}`, points: series.A, color: '--a'},
      {label: `B · ${labels.B || 'B'}`, points: series.B, color: '--b'}
    ], {xLabel});
  });
  const history = (payload.quality && payload.quality.history) || [];
  const side = payload.match && (payload.match.learner_side || payload.match.ncm_side) || 'B';
  const sideLabel = sideName(payload.match || {}, side);
  const qualityPoints = (key) => history.map((entry, index) => {
    const stats = entry.agents ? entry.agents[side] : entry.ncm;
    const value = stats ? stats[key] : null;
    return [index, value];
  }).filter((point) => Number.isFinite(+point[1]));
  if (qualityPoints('generated').length) {
    drawChart($('dchart-quality'), [
      {label: `generated (${side} · ${sideLabel})`, points: qualityPoints('generated'), color: '--c3'},
      {label: `repaired (${side} · ${sideLabel})`, points: qualityPoints('repaired'), color: '--c2'},
      {label: `fallback (${side} · ${sideLabel})`, points: qualityPoints('fallbacks'), color: '--c5'}
    ], {xLabel: (value) => {
      const entry = history[Math.round(value)];
      return entry ? `T${entry.turn}${entry.player || ''}` : '';
    }});
  } else {
    $('dchart-quality').innerHTML = '<div class="empty">per-turn provenance is recorded for games played after this dashboard update; aggregate totals are below</div>';
  }
}

function renderQuality() {
  const payload = state.detail;
  const host = $('detail-quality');
  if (!host || !payload) return;
  const quality = payload.quality || {};
  const match = payload.match || {};
  const side = match.learner_side || match.ncm_side || 'B';
  const sideLabel = sideName(match, side);
  const owner = match.source === 'play' ? 'NCM' : 'Learner NCM';
  const opponentLabel = match.opponent_label || 'opponent';
  const learner = quality.learner || {};
  const opponent = quality.opponent || {};
  const gen = Number(learner.generated) || 0;
  const rep = Number(learner.repaired) || 0;
  const fb = Number(learner.fallbacks) || 0;
  const ill = Number(learner.illegal) || 0;
  const total = gen + rep + fb;
  const fmtCount = (value) => Number.isFinite(+value) ? String(value) : '—';
  let reasons = quality.fallback_reasons || {};
  if (reasons && typeof reasons === 'object' && reasons[side] && typeof reasons[side] === 'object') {
    reasons = reasons[side];
  }
  const reasonRows = Object.entries(reasons || {}).map(([reason, count]) =>
    `<tr><td>${esc(reason)}</td><td>${esc(count)}</td></tr>`).join('');
  const heading = `${owner} quality (side ${side}${sideLabel !== owner ? ' · ' + sideLabel : ''})`;
  if (!total) {
    host.innerHTML = `<h2>${esc(heading)}</h2>
      <div class="empty">no provenance saved for this game — it was played before the meta sidecar was added</div>`;
    return;
  }
  const cell = (value, cls) => Number.isFinite(+value)
    ? `<td class="${cls}">${esc(value)}</td>` : '<td class="dim">—</td>';
  host.innerHTML = `<h2>${esc(heading)}</h2>
    <table class="quality-table"><thead><tr><th>generated</th><th>repaired</th><th>fallback</th>
    <th>illegal samples</th><th>generated share</th></tr></thead><tbody><tr>
      ${cell(learner.generated, 'ok')}
      ${cell(learner.repaired, 'warn')}
      ${cell(learner.fallbacks, 'bad-text')}
      ${cell(learner.illegal, '')}
      <td>${total ? (gen / total * 100).toFixed(1) + '%' : '—'}</td>
    </tr></tbody></table>
    ${opponent && Object.keys(opponent).length ? `<div class="dim" style="margin-top:8px">${esc(opponentLabel)}: gen ${fmtCount(opponent.generated)} · rep ${fmtCount(opponent.repaired)} · fb ${fmtCount(opponent.fallbacks)}</div>` : ''}
    ${reasonRows ? `<h2 style="margin-top:14px">Fallback reasons</h2><table class="quality-table"><tbody>${reasonRows}</tbody></table>` : ''}`;
}

function setFrame(index) {
  if (!state.detail || !state.detail.frames) return;
  const max = state.detail.frames.length - 1;
  state.frame = Math.max(0, Math.min(max, Number(index) || 0));
  renderBoard();
}

function stepFrame(delta) {
  setFrame(state.frame + delta);
}

function toggleReplay() {
  if (state.playing) { stopReplay(); return; }
  if (!state.detail || !state.detail.frames) return;
  if (state.frame >= state.detail.frames.length - 1) state.frame = 0;
  state.playing = true;
  state.timer = setInterval(() => {
    if (!state.detail) { stopReplay(); return; }
    if (state.frame >= state.detail.frames.length - 1) { stopReplay(); return; }
    setFrame(state.frame + 1);
  }, state.speed);
  renderBoard();
}

function stopReplay() {
  state.playing = false;
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  const button = $('play-button');
  if (button) button.textContent = '▶ play';
}

function jumpTo(seq) {
  if (!state.detail) return;
  const index = state.detail.frames.findIndex((frame) => frame.seq === seq);
  if (index >= 0) setFrame(index);
}

$('runs').addEventListener('change', (event) => {
  state.id = event.target.value;
  localStorage.setItem('tbs-ncm-run', state.id);
  state.payload = null;
  pollTraining();
});

$('filter-source').addEventListener('change', (event) => {
  state.filters.source = event.target.value;
  updateResultFilterLabels();
  renderGamesTable();
});
$('filter-result').addEventListener('change', (event) => {
  state.filters.result = event.target.value;
  renderGamesTable();
});
$('filter-seed').addEventListener('input', (event) => {
  state.filters.seed = event.target.value.trim();
  renderGamesTable();
});
$('filter-text').addEventListener('input', (event) => {
  state.filters.text = event.target.value.trim();
  renderGamesTable();
});

$('games-table').addEventListener('click', (event) => {
  const row = event.target.closest('tr[data-file]');
  if (!row) return;
  openGame(row.dataset.file);
});

document.addEventListener('keydown', (event) => {
  if (!state.detail) return;
  if (event.key === 'Escape') { closeDetail(); return; }
  if (event.key === 'ArrowLeft') { event.preventDefault(); stepFrame(-1); }
  if (event.key === 'ArrowRight') { event.preventDefault(); stepFrame(1); }
  if (event.key === ' ') { event.preventDefault(); toggleReplay(); }
});

window.addEventListener('resize', () => {
  if (!state.payload) return;
  const stage = state.payload.summary.stage;
  (STAGE_PANELS[stage] || []).forEach((panel, i) => {
    const host = document.getElementById('chart-' + i);
    if (host) lineChart(host, panel.series);
  });
  if (state.detail && !state.detail.error) renderDetailCharts();
});

const startup = new URLSearchParams(location.search);
if (startup.get('run')) state.id = startup.get('run');
updateResultFilterLabels();
pollTraining();
pollGames();
if (startup.get('tab') === 'games') setTab('games');
if (startup.get('game')) setTimeout(() => openGame(startup.get('game')), 350);
setInterval(() => {
  if (state.tab === 'training') pollTraining();
  else pollGames();
}, 2000);
</script>
</body>
</html>
"""


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="live dashboard for tbs_ncm runs, games and replays")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args(argv)

    root = runs_root(args.runs_dir)
    root.mkdir(parents=True, exist_ok=True)

    try:
        server = DashboardServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"cannot bind {args.host}:{args.port} ({exc}); try --port {args.port + 1}")
        return 1
    server.runs_dir = str(root)  # type: ignore[attr-defined]

    url = f"http://{args.host}:{args.port}/"
    print(f"tbs_ncm dashboard: {url}")
    print(f"reading runs from {root}")
    print("ctrl-c to stop")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
