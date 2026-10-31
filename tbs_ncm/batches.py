"""Evaluation-batch index for the dashboard.

A *batch* is one evaluation or league run: a persona league, a mixed league, a
``vs_ncm`` iteration phase, or a self-play summary.  The dashboard groups saved
games by these runs instead of treating every game as an isolated result.

Most runs write a ``report.json`` under ``artifacts/tbs/personas/<run>/``.  The
older persona leagues include one row per game, so a game can be matched by
``(seed, map, pair)``.  Mixed leagues store aggregate counters only; those games
are matched by the run's time window and the personas/heuristics/map recorded
in their meta sidecar.  ``vs_ncm`` arena games and self-play rows carry their
origin path directly.

The module is deliberately stdlib-only and torch-free: the dashboard imports it
on every poll.  Parsed reports are cached by file modification time.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

try:  # imported lazily enough that tests can pass an explicit root
    from .common import ARTIFACT_DIR
except Exception:  # pragma: no cover - only relevant to partial test stubs
    ARTIFACT_DIR = Path("artifacts/tbs")

_BATCH_CACHE: dict[str, tuple[tuple[tuple[str, int], ...], dict]] = {}
_DATE_TOKEN = re.compile(r"20\d{6}(?:[_-]\d{4})?")
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_MIXED_TOLERANCE_S = 10.0


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _seed_key(value: object) -> int | str | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _pair_key(pair: object) -> frozenset[str]:
    if not isinstance(pair, list):
        return frozenset()
    return frozenset(str(item) for item in pair if str(item))


def _parse_time(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return time.mktime(time.strptime(text, _TIME_FORMAT))
    except ValueError:
        return None


def _row_time(row: dict) -> float | None:
    for key in ("started_at", "finished_at"):
        parsed = _parse_time(row.get(key))
        if parsed is not None:
            return parsed
    try:
        stamp = float(row.get("mtime") or 0)
    except (TypeError, ValueError):
        stamp = 0.0
    return stamp if stamp > 0 else None


def short_label(name: object) -> str:
    """Turn a run directory into a compact chart label."""
    text = str(name or "").strip()
    if not text:
        return "batch"
    if text.startswith("personas/"):
        text = text.split("/", 1)[1]
    if text.startswith("mixed_league"):
        prefix = "mixed "
        text = text[len("mixed_league") :].strip("_- ")
    elif text.startswith("top_round_robin"):
        prefix = "RR "
        text = text[len("top_round_robin") :].strip("_- ")
    elif text.startswith("round_robin"):
        prefix = "RR "
        text = text[len("round_robin") :].strip("_- ")
    elif text.startswith("league"):
        prefix = ""
        text = text[len("league") :].strip("_- ")
    else:
        prefix = ""
    text = _DATE_TOKEN.sub(" ", text)
    text = re.sub(r"[_\-]+", " ", text).strip()
    text = re.sub(r"\s{2,}", " ", text)
    if not text:
        # A run named only by date (e.g. mixed_league_20261025) keeps a compact
        # fallback instead of repeating the whole directory name.
        text = str(name).replace("_", " ")
        text = text.replace("mixed league", "mixed")
        text = re.sub(r"(\d{8})\s+\d{4}$", r"\1", text).strip()
        return text[:28] + "…" if len(text) > 29 else text
    label = prefix + text
    return label[:28] + "…" if len(label) > 29 else label


def _persona_batches(root: Path) -> list[dict]:
    batches: list[dict] = []
    base = root / "personas"
    if not base.is_dir():
        return batches
    for path in sorted(base.glob("*/report.json")):
        data = _json(path, {})
        if not isinstance(data, dict):
            continue
        games = data.get("games")
        game_list = games if isinstance(games, list) else []
        if game_list:
            expected = len(game_list)
        else:
            try:
                expected = int(data.get("matches") or games or 0)
            except (TypeError, ValueError):
                expected = 0
        if expected <= 0:
            continue

        name = path.parent.name
        mixed = "ncm" in data or "ncm_personas" in data
        if mixed:
            kind = "mixed_league"
            mode = "mixed"
        elif name.startswith("top_round_robin") or name.startswith("round_robin"):
            kind = "round_robin"
            mode = "league"
        else:
            kind = "league"
            mode = "league"

        keys: set[tuple[int | str, str, frozenset[str]]] = set()
        game_maps: list[str] = []
        for game in game_list:
            seed = _seed_key(game.get("seed"))
            map_name = str(game.get("map") or "").strip()
            pair = _pair_key(game.get("pair"))
            if map_name and map_name not in game_maps:
                game_maps.append(map_name)
            if seed is not None and map_name and pair:
                keys.add((seed, map_name, pair))

        maps = _string_list(data.get("maps")) or game_maps
        heuristics = [
            str(item.get("name"))
            for item in (data.get("heuristics") or [])
            if isinstance(item, dict) and item.get("name")
        ]
        seconds = data.get("seconds") or 0
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            seconds = 0.0
        at = _mtime(path)
        batches.append(
            {
                "id": f"personas/{name}",
                "label": name,
                "short": short_label(name),
                "kind": kind,
                "mode": mode,
                "at": at,
                "start": at - seconds if seconds > 0 else None,
                "end": at,
                "generator": data.get("generator"),
                "decode": data.get("decode"),
                "maps": maps,
                "max_turns": data.get("max_turns"),
                "expected_games": expected,
                "game_keys": keys,
                "ncm_personas": _string_list(
                    data.get("ncm_personas") or data.get("personas")
                ),
                "heuristics": heuristics,
                "report_path": str(path),
            }
        )
    return batches


def _fill_intervals(batches: list[dict]) -> None:
    """Infer starts for reports that do not record their wall-clock duration."""
    ordered = sorted(
        (batch for batch in batches if batch.get("mode") != "vs_ncm"),
        key=lambda batch: batch.get("end") or 0,
    )
    previous_end: dict[str, float] = {}
    for batch in ordered:
        start = batch.get("start")
        end = float(batch.get("end") or 0)
        mode = str(batch.get("mode") or "league")
        if start is None:
            # A resumed or report-only run has seconds=0.  It starts where the
            # previous run of the same kind ended; the first one gets a
            # generous one-hour window.
            batch["start"] = previous_end.get(mode, end - 3600)
        if end:
            previous_end[mode] = end


def _vs_ncm_batches(root: Path) -> tuple[list[dict], dict[str, str], dict[str, str]]:
    batches: list[dict] = []
    by_arena_path: dict[str, str] = {}
    by_summary_path: dict[str, str] = {}
    base = root / "vs_ncm"
    if not base.is_dir():
        return batches, by_arena_path, by_summary_path

    for path in sorted(base.rglob("arena_*.json")):
        data = _json(path, {})
        if not isinstance(data, dict):
            continue
        relative = path.relative_to(base).parts
        if len(relative) < 2:
            continue
        run = relative[0]
        iteration = relative[1]
        phase = "post" if "post" in path.stem else "baseline"
        batch_id = f"vs_ncm/{run}/{iteration}/{phase}"
        matches = data.get("matches") or []
        maps: list[str] = []
        for game in matches:
            scenario = str(game.get("scenario") or "").strip()
            if scenario and scenario not in maps:
                maps.append(scenario)
        at = _mtime(path)
        batches.append(
            {
                "id": batch_id,
                "label": f"{run}/{iteration} {phase}",
                "short": f"{iteration} {phase}",
                "kind": "vs_ncm",
                "mode": "vs_ncm",
                "at": at,
                "start": at,
                "end": at,
                "generator": None,
                "decode": None,
                "maps": maps,
                "max_turns": None,
                "expected_games": len(matches),
                "game_keys": set(),
                "arena_path": str(path),
            }
        )
        by_arena_path[str(path)] = batch_id
        data_label = str(data.get("label") or "")
        if data_label:
            by_arena_path[data_label] = batch_id

    for path in sorted(base.rglob("selfplay.summary.json")):
        data = _json(path, {})
        if not isinstance(data, dict):
            continue
        relative = path.relative_to(base).parts
        if len(relative) < 2:
            continue
        run = relative[0]
        iteration = relative[1]
        batch_id = f"vs_ncm/{run}/{iteration}/selfplay"
        at = _mtime(path)
        games = data.get("games") or []
        batches.append(
            {
                "id": batch_id,
                "label": f"{run}/{iteration} self-play",
                "short": f"self-play {iteration}",
                "kind": "selfplay",
                "mode": "selfplay",
                "at": at,
                "start": at,
                "end": at,
                "generator": None,
                "decode": None,
                "maps": ["border_skirmish"],
                "max_turns": None,
                "expected_games": len(games),
                "game_keys": set(),
            }
        )
        by_summary_path[str(path)] = batch_id
    return batches, by_arena_path, by_summary_path


def _scan_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    personas = root / "personas"
    if personas.is_dir():
        paths.extend(sorted(personas.glob("*/report.json")))
    vs = root / "vs_ncm"
    if vs.is_dir():
        paths.extend(sorted(vs.rglob("arena_*.json")))
        paths.extend(sorted(vs.rglob("selfplay.summary.json")))
    return paths


def build_index(root: Path | str | None = None) -> dict:
    """Build (and cache) the evaluation-batch index for one artifact root."""
    root_path = Path(root) if root is not None else Path(ARTIFACT_DIR)
    cache_key = str(root_path.resolve())
    paths = _scan_paths(root_path)
    signature = tuple((str(path), int(_mtime(path) * 1000)) for path in paths)
    cached = _BATCH_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1]

    persona = _persona_batches(root_path)
    _fill_intervals(persona)
    vs_batches, by_arena_path, by_summary_path = _vs_ncm_batches(root_path)
    batches = persona + vs_batches
    batches.sort(key=lambda batch: (batch.get("at") or 0, batch.get("id") or ""))

    by_id = {batch["id"]: batch for batch in batches}
    by_label: dict[str, str] = {}
    by_key: dict[tuple, list[str]] = {}
    for batch in batches:
        by_label.setdefault(str(batch.get("label") or ""), batch["id"])
        for key in batch.get("game_keys") or ():
            by_key.setdefault(key, []).append(batch["id"])

    index = {
        "batches": batches,
        "by_id": by_id,
        "by_label": by_label,
        "by_key": by_key,
        "by_arena_path": by_arena_path,
        "by_summary_path": by_summary_path,
    }
    _BATCH_CACHE[cache_key] = (signature, index)
    return index


def _candidate_for_time(
    candidates: Iterable[dict], when: float | None
) -> dict | None:
    rows = list(candidates)
    if not rows:
        return None
    if when is None:
        return rows[-1]
    for batch in rows:
        if (batch.get("start") or 0) <= when <= (batch.get("end") or 0):
            return batch
    return min(rows, key=lambda batch: abs((batch.get("end") or 0) - when))


def _match_row(row: dict, index: dict) -> str | None:
    by_id = index["by_id"]

    raw = str(row.get("batch") or "").strip()
    if raw:
        if raw in by_id:
            return raw
        labelled = index["by_label"].get(raw)
        if labelled:
            return labelled

    arena_path = str(row.get("arena_path") or "").strip()
    if arena_path:
        match = index["by_arena_path"].get(arena_path)
        if match:
            return match
        # Older rows may carry a label instead of the full path.
        for key, batch_id in index["by_arena_path"].items():
            if Path(key).name and Path(key).name in arena_path:
                return batch_id
            if key and key in arena_path:
                return batch_id

    summary_path = str(row.get("batch_path") or "").strip()
    if summary_path:
        match = index["by_summary_path"].get(summary_path)
        if match:
            return match

    seed = _seed_key(row.get("seed"))
    map_name = str(row.get("scenario") or "").strip()
    learner = str(row.get("learner_persona") or "").strip()
    opponent = str(row.get("opponent_persona") or "").strip()
    if seed is not None and map_name and learner and opponent:
        key = (seed, map_name, frozenset({learner, opponent}))
        candidates = [
            by_id[batch_id] for batch_id in index["by_key"].get(key, []) if batch_id in by_id
        ]
        if candidates:
            batch = _candidate_for_time(candidates, _row_time(row))
            if batch:
                return str(batch["id"])

    when = _row_time(row)
    if when is None:
        return None
    file_name = str(row.get("file") or row.get("id") or "")
    label_name = str(row.get("label") or "")
    if not (file_name.startswith("mixed-") or label_name.startswith("mixed-")):
        return None
    mixed = []
    for batch in index["batches"]:
        if batch.get("mode") != "mixed":
            continue
        start = float(batch.get("start") or 0) - _MIXED_TOLERANCE_S
        # ``end`` is the report write time, so a game started before it.  A
        # small forward allowance only absorbs clock rounding; a generous one
        # would steal the first game of the next back-to-back run.
        end = float(batch.get("end") or 0) + 1.0
        if not (start <= when <= end):
            continue
        personas = set(batch.get("ncm_personas") or [])
        if learner and personas and learner not in personas:
            continue
        heuristics = set(batch.get("heuristics") or [])
        if opponent and heuristics and opponent not in heuristics:
            continue
        maps = set(batch.get("maps") or [])
        if map_name and maps and map_name not in maps:
            continue
        mixed.append(batch)
    batch = _candidate_for_time(mixed, when)
    return str(batch["id"]) if batch else None


def attach_batches(rows: list[dict], root: Path | str | None = None) -> list[dict]:
    """Attach ``batch*`` fields to game rows and return public batch metadata.

    Rows are mutated in place.  ``games_payload`` calls this once per poll and
    forwards the returned list to the Overview tab.
    """
    index = build_index(root)
    by_id = index["by_id"]
    counts: dict[str, int] = {}

    for row in rows:
        batch_id = _match_row(row, index)
        if batch_id and batch_id in by_id:
            batch = by_id[batch_id]
            counts[batch_id] = counts.get(batch_id, 0) + 1
            row["batch"] = batch_id
            row["batch_label"] = batch.get("label") or batch_id
            row["batch_short"] = batch.get("short") or row["batch_label"]
            row["batch_kind"] = batch.get("kind") or ""
            row["batch_at"] = batch.get("at")
            row["batch_generator"] = batch.get("generator")
            row["batch_decode"] = batch.get("decode")
            row["batch_maps"] = batch.get("maps") or []
            row["batch_max_turns"] = batch.get("max_turns")
        else:
            row["batch"] = None
            row["batch_label"] = None
            row["batch_short"] = None
            row["batch_kind"] = None
            row["batch_at"] = None
            row["batch_generator"] = None
            row["batch_decode"] = None
            row["batch_maps"] = []
            row["batch_max_turns"] = None

    public: list[dict] = []
    for batch in index["batches"]:
        if not batch.get("expected_games"):
            continue
        public.append(
            {
                "id": batch["id"],
                "label": batch.get("label") or batch["id"],
                "short": batch.get("short") or batch.get("label") or batch["id"],
                "kind": batch.get("kind") or "",
                "at": batch.get("at"),
                "generator": batch.get("generator"),
                "decode": batch.get("decode"),
                "maps": batch.get("maps") or [],
                "max_turns": batch.get("max_turns"),
                "expected_games": batch.get("expected_games") or 0,
                "assigned": counts.get(batch["id"], 0),
            }
        )
    return public
