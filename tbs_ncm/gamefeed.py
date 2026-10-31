"""Read-side game feed for the TBS/NCM dashboard.

The training dashboard only needs :mod:`tbs_ncm.monitor` run directories.  The
game browser needs something else: a single index of every saved match plus the
arena and self-play summaries around it, and a way to turn a saved order list
back into a board the browser can draw.

This module is deliberately torch-free.  It imports the game engine (pure
standard library) and replays accepted orders, which is cheap: a 600-order game
takes about ten milliseconds.  Nothing here is used by training code.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from .batches import attach_batches
from .common import ARTIFACT_DIR, GAME_REPO

from tbs.engine import rules
from tbs.engine.data import available_scenarios, load_scenario, load_specs
from tbs.engine.serialize import load_game, state_hash
from tbs.engine.state import new_game

MATCH_DIR = ARTIFACT_DIR / "matches"
VS_NCM_DIR = ARTIFACT_DIR / "vs_ncm"
RUNS_DIR = ARTIFACT_DIR / "runs"

_MAX_DETAIL_CACHE = 6
_DETAIL_CACHE: "OrderedDict[str, tuple[tuple[float, ...], dict]]" = OrderedDict()
_MATCH_CACHE: dict[str, tuple[tuple[float, ...], dict]] = {}
_ARENA_CACHE: dict[str, tuple[float, dict]] = {}
_SCENARIO_INDEX: dict[str, Path] | None = None
_SPECS = None


# ---------------------------------------------------------------------------
# small helpers


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _stamp(*paths: Path) -> tuple[float, ...]:
    values = []
    for path in paths:
        try:
            values.append(path.stat().st_mtime)
        except OSError:
            values.append(0.0)
    return tuple(values) or (0.0,)


def _time_text(stamp: float) -> str:
    """Local ``YYYY-MM-DD HH:MM`` for a file/epoch timestamp."""
    if not stamp:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp))


def _display_time(value: object) -> str:
    """Normalize metadata timestamps for the narrow Games table.

    League writers may store ISO-8601 (with microseconds and a timezone);
    the dashboard reads a compact local ``YYYY-MM-DD HH:MM``.  Parsing here
    keeps older saved games correct without rewriting their sidecars.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    if "T" in text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone()
            return parsed.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
    return text


def scenario_index() -> dict[str, Path]:
    global _SCENARIO_INDEX
    if _SCENARIO_INDEX is not None:
        return _SCENARIO_INDEX
    index: dict[str, Path] = {}
    for path in available_scenarios():
        index[Path(path).stem] = Path(path)
    maps = GAME_REPO / "data" / "maps"
    if maps.is_dir():
        for path in maps.glob("*.json"):
            index.setdefault(path.stem, path)
    _SCENARIO_INDEX = index
    return index


def specs():
    global _SPECS
    if _SPECS is None:
        _SPECS = load_specs()
    return _SPECS


def resolve_scenario(name: str | None, meta: dict | None = None) -> Path | None:
    """Find the scenario JSON for a saved game.

    New games write ``scenario_path`` into their meta sidecar.  Older saves only
    carry the scenario name, which is enough for every bundled map.
    """
    if meta:
        raw = meta.get("scenario_path")
        if isinstance(raw, str) and raw:
            path = Path(raw)
            if path.is_file():
                return path
    if name:
        path = scenario_index().get(str(name))
        if path is not None:
            return path
        fallback = GAME_REPO / "data" / "maps" / f"{name}.json"
        if fallback.is_file():
            return fallback
    return None


def board_stats(state, player: str) -> dict:
    """The same cap-safe score ``arena._board_stats`` reports."""
    units = state.units_of(player)
    sites = state.owned_tiles(player)
    hp = sum(unit.hp for unit in units)
    board = {
        "units": len(units),
        "hp": hp,
        "sites": len(sites),
        "funds": state.players[player].funds,
        "hq_hp": sum(unit.hp for unit in units if unit.type_id == "hq"),
    }
    board["score"] = round(2 * board["sites"] + board["units"] + hp / 100, 2)
    return board


def save_board_stats(data: dict, player: str) -> dict:
    """Board score straight from a save dictionary (no engine round-trip)."""
    units = [unit for unit in data.get("units", []) if unit.get("owner") == player]
    tiles = data.get("map", {}).get("tiles", [])
    hp = sum(int(unit.get("hp", 0)) for unit in units)
    sites = sum(1 for tile in tiles if tile.get("owner") == player)
    funds = 0
    for entry in data.get("players", []):
        if entry.get("id") == player:
            funds = int(entry.get("funds", 0))
            break
    board = {
        "units": len(units),
        "hp": hp,
        "sites": sites,
        "funds": funds,
        "hq_hp": sum(int(unit.get("hp", 0)) for unit in units if unit.get("type") == "hq"),
    }
    board["score"] = round(2 * board["sites"] + board["units"] + hp / 100, 2)
    return board


def _meta_path(save: Path) -> Path:
    return save.with_name(save.stem + ".meta.json")


def _orders_path(save: Path) -> Path:
    return save.with_name(save.stem + ".orders.json")


def _meta_for(save: Path) -> dict:
    data = _json(_meta_path(save), {})
    return data if isinstance(data, dict) else {}


def _infer_play_side(name: str, meta: dict) -> str | None:
    value = meta.get("ncm_side")
    if value in ("A", "B", "both"):
        return value
    lowered = name.lower()
    if lowered.startswith("ncm_vs_") or lowered.startswith("ncm_match"):
        return "B"
    return None


def _infer_opponent(name: str, meta: dict) -> str | None:
    value = meta.get("opponent")
    if isinstance(value, str) and value:
        return value
    lowered = name.lower()
    if "vs_greedy" in lowered:
        return "greedy"
    if "vs_llm" in lowered:
        return "llm"
    if "vs-ncm" in lowered or "ncm-vs-ncm" in lowered:
        return "ncm"
    return None


def _pretty_opponent(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    aliases = {
        "llm": "LLM",
        "greedy": "Greedy",
        "ncm": "NCM",
        "human": "Human",
    }
    return aliases.get(text.lower(), text[:1].upper() + text[1:])


def _decorate_display(row: dict) -> dict:
    """Add human-readable side, result and winner labels.

    The raw arena/play data only has sides ``A``/``B`` and a ``learner_side``.
    Both players can be AI, so a bare "AI win" is ambiguous.  This function
    turns those fields into source-aware labels used by every dashboard view.
    """
    source = str(row.get("source") or "play").lower()
    learner = row.get("learner_side")
    ncm_side = row.get("ncm_side")
    opponent = _pretty_opponent(row.get("opponent"))
    labels = {"A": "A", "B": "B"}

    world = row.get("world_cup") if isinstance(row.get("world_cup"), dict) else {}
    team_labels = row.get("team_labels") if isinstance(row.get("team_labels"), dict) else {}
    if world and team_labels:
        labels = {
            side: str(team_labels.get(side) or side) for side in ("A", "B")
        }
        row["side_labels"] = labels
        row["opponent_label"] = labels["B"]
        winner_side = world.get("winner_side") or row.get("winner")
        winner_name = world.get("winner_name")
        if winner_side in labels:
            row["winner_label"] = labels[winner_side]
        elif winner_name:
            row["winner_label"] = str(winner_name)
        else:
            row["winner_label"] = None
        # The World Cup has its own turn cap, so a saved game can still say
        # "ongoing" when the league has already declared a draw or a board-score
        # winner.  The tournament metadata is authoritative for its games.
        if world.get("decided_by") in ("draw", "board", "seed", "victory"):
            row["finished"] = True
        if row.get("finished"):
            row["result_label"] = (
                "draw" if not (winner_side or winner_name) else "decided"
            )
        else:
            row["result_label"] = "unfinished"
        return row

    if source == "all_ai_league" and team_labels:
        # Neutral league games have no NCM "learner" side.  Use the labels the
        # league wrote instead of the bare A/B fallback and derive the result
        # from the league decision rather than learner_won.
        labels = {
            side: str(team_labels.get(side) or side) for side in ("A", "B")
        }
        league = row.get("league") if isinstance(row.get("league"), dict) else {}
        row["side_labels"] = labels
        row["opponent_label"] = labels["B"]
        winner_side = row.get("winner") or league.get("winner")
        decided_by = league.get("decided_by")
        if winner_side in labels:
            row["winner_label"] = labels[winner_side]
            row["result_label"] = (
                f"{labels[winner_side]} win"
                + (" (board)" if decided_by == "board" else "")
            )
        else:
            row["winner_label"] = None
            row["result_label"] = "draw" if row.get("finished") else "unfinished"
        return row

    if source == "play":
        if ncm_side in ("A", "B"):
            other = "B" if ncm_side == "A" else "A"
            labels[ncm_side] = "NCM"
            labels[other] = opponent or "Opponent"
            row["learner_side"] = ncm_side
            row["opponent_side"] = other
            row["opponent_label"] = labels[other]
        elif ncm_side == "both":
            labels = {"A": "NCM", "B": "NCM"}
            row["opponent_label"] = "NCM"
        elif opponent.lower() == "ncm":
            # Older ``vs-ncm`` saves do not say which side was the learner.
            labels = {"A": "NCM", "B": "NCM"}
            row["opponent_label"] = "NCM"
    elif source in ("arena", "selfplay"):
        if learner in ("A", "B"):
            other = "B" if learner == "A" else "A"
            learner_persona = row.get("learner_persona")
            opponent_persona = row.get("opponent_persona")
            if learner_persona and opponent_persona:
                labels[learner] = f"{learner_persona} · NCM"
                suffix = (
                    "Heuristic"
                    if row.get("opponent_kind") == "heuristic"
                    else "NCM"
                )
                labels[other] = f"{opponent_persona} · {suffix}"
                row["learner_label"] = labels[learner]
            else:
                labels[learner] = "Learner NCM"
                labels[other] = "Opponent NCM"
            row["opponent_side"] = other
            row["opponent_label"] = labels[other]
    else:
        row["opponent_label"] = opponent

    if source == "play" and ncm_side == "both":
        row["opponent_label"] = "NCM"
    if not row.get("opponent_label"):
        row["opponent_label"] = opponent

    winner = row.get("winner")
    if winner in ("A", "B"):
        row["winner_label"] = f"{winner} · {labels[winner]}"
    elif winner:
        row["winner_label"] = str(winner)
    else:
        row["winner_label"] = None

    finished = bool(row.get("finished"))
    status = str(row.get("status") or "").lower()
    learner_won = row.get("learner_won")
    if finished and status == "draw":
        row["result_label"] = "draw"
    elif finished and learner_won in (True, 1):
        row["result_label"] = "NCM win" if source == "play" else "Learner win"
    elif finished and learner_won in (False, 0) and row.get("learner_side") in ("A", "B"):
        row["result_label"] = "NCM loss" if source == "play" else "Learner loss"
    elif finished:
        row["result_label"] = "finished"
    else:
        row["result_label"] = "unfinished"

    row["side_labels"] = labels
    return row


# ---------------------------------------------------------------------------
# arena and self-play indexes


def _arena_rows() -> dict[str, dict]:
    """Every per-game row from ``vs_ncm/**/arena_*.json``, keyed by save basename."""
    rows: dict[str, dict] = {}
    if not VS_NCM_DIR.is_dir():
        return rows
    for path in sorted(VS_NCM_DIR.rglob("arena_*.json")):
        try:
            stamp = path.stat().st_mtime
        except OSError:
            continue
        cached = _ARENA_CACHE.get(str(path))
        if cached and cached[0] == stamp:
            data = cached[1]
        else:
            data = _json(path, {})
            if not isinstance(data, dict):
                continue
            _ARENA_CACHE[str(path)] = (stamp, data)
        phase = "post" if "post" in path.stem else "baseline"
        iteration = ""
        for part in path.parts:
            if part.startswith("iter"):
                iteration = part
                break
        for game in data.get("matches", []) or []:
            save = game.get("save")
            key = Path(save).name if isinstance(save, str) and save else ""
            if not key:
                key = f"{data.get('label', 'arena')}-seed{game.get('seed', '')}"
            rows[key] = {
                "arena_label": data.get("label"),
                "arena_phase": phase,
                "arena_iteration": iteration,
                "arena_path": str(path),
                "arena_source": path.parent.parent.name if path.parent.parent else "",
                "mtime": stamp,
                "finished_at": _time_text(stamp),
                "learner_side": game.get("learner_side"),
                "opponent_side": game.get("opponent_side"),
                "learner": game.get("learner"),
                "opponent_stats": game.get("opponent"),
                "score_delta": game.get("score_delta"),
                "finished": game.get("finished"),
                "learner_won": game.get("learner_won"),
                "turns": game.get("turns"),
                "orders": game.get("orders"),
                "seconds": game.get("seconds"),
                "replay_matches": game.get("replay_matches"),
                "board": game.get("board"),
            }
    return rows


def _reports() -> list[dict]:
    reports: list[dict] = []
    if not VS_NCM_DIR.is_dir():
        return reports
    for path in sorted(VS_NCM_DIR.rglob("report.json")):
        data = _json(path, {})
        if not isinstance(data, dict):
            continue
        entry = {
            "path": str(path),
            "run": path.parent.name,
            "started": data.get("started"),
            "iterations": data.get("iterations", []),
            "elapsed_s": data.get("elapsed_s"),
            "error": data.get("error"),
        }
        reports.append(entry)
    return reports


def _selfplay_rows() -> list[dict]:
    rows: list[dict] = []
    if not VS_NCM_DIR.is_dir():
        return rows
    for path in sorted(VS_NCM_DIR.rglob("selfplay.summary.json")):
        data = _json(path, {})
        if not isinstance(data, dict):
            continue
        try:
            stamp = path.stat().st_mtime
        except OSError:
            stamp = 0.0
        run = path.parent.parent.name if path.parent.parent else path.parent.name
        label = f"{run}/{path.parent.name}"
        for index, game in enumerate(data.get("games", []) or []):
            learner = game.get("learner") or {}
            opponent = game.get("opponent") or {}
            rows.append(
                {
                    "id": f"selfplay:{path}:{index}",
                    "file": None,
                    "source": "selfplay",
                    "label": label,
                    "scenario": "border_skirmish",
                    "seed": game.get("seed"),
                    "status": game.get("status"),
                    "winner": game.get("winner") or None,
                    "learner_side": game.get("learner_side"),
                    "opponent_side": None,
                    "learner_won": bool(game.get("learner_won")),
                    "finished": bool(game.get("finished", game.get("status") != "ongoing")),
                    "turns": game.get("turns"),
                    "orders": game.get("examples"),
                    "examples": game.get("examples"),
                    "kinds": game.get("kinds"),
                    "quality": {"learner": learner, "opponent": opponent},
                    "learner": learner,
                    "opponent_stats": opponent,
                    "score_delta": None,
                    "has_replay": False,
                    "finished_at": _time_text(stamp),
                    "mtime": stamp,
                    "duration_s": game.get("seconds"),
                    "meta_path": None,
                    "batch_path": str(path),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# match index


def _summarize_match(path: Path) -> dict | None:
    stamp = _stamp(path, _meta_path(path))
    if not stamp[0]:
        return None
    cached = _MATCH_CACHE.get(path.name)
    if cached and cached[0] == stamp:
        return cached[1]

    data = _json(path)
    if not isinstance(data, dict):
        return None
    units = data.get("units")
    meta = _meta_for(path)
    orders = _orders_path(path)
    row = {
        "id": path.name,
        "file": path.name,
        "source": "play",
        "scenario": data.get("scenario"),
        "seed": data.get("seed"),
        "status": data.get("status"),
        "winner": data.get("winner") or None,
        "turns": data.get("turn_number"),
        "orders": data.get("action_seq"),
        "started_at": None,
        "finished_at": _time_text(stamp[0]),
        "duration_s": None,
        "mtime": stamp[0],
        "unit_counts": (
            {owner: sum(1 for unit in units if unit.get("owner") == owner) for owner in ("A", "B")}
            if isinstance(units, list)
            else None
        ),
        "has_replay": orders.is_file(),
        "meta_path": str(_meta_path(path)) if _meta_path(path).is_file() else None,
        "ncm_side": _infer_play_side(path.name, meta),
        "opponent": _infer_opponent(path.name, meta),
        "opponent_kind": meta.get("opponent_kind"),
        "quality": None,
        "learner_side": None,
        "opponent_side": None,
        "learner_won": None,
        "finished": data.get("status") != "ongoing",
        "score_delta": None,
        "label": path.name,
    }
    if meta:
        opponent_quality = meta.get("opponent")
        row["quality"] = {
            "learner": meta.get("learner"),
            "opponent": opponent_quality if isinstance(opponent_quality, dict) else None,
        }
        row["source"] = meta.get("source") or row["source"]
        row["label"] = meta.get("label") or row["label"]
        row["learner_side"] = meta.get("learner_side") or row["learner_side"]
        row["opponent_side"] = meta.get("opponent_side") or row["opponent_side"]
        row["learner_persona"] = meta.get("learner_persona")
        row["opponent_persona"] = meta.get("opponent_persona")
        row["batch"] = meta.get("batch")
        row["started_at"] = _display_time(meta.get("started_at")) or row["started_at"]
        row["finished_at"] = _display_time(meta.get("finished_at")) or row["finished_at"]
        if isinstance(meta.get("league"), dict):
            row["league"] = meta["league"]
            row["finished"] = bool(row["finished"] or meta["league"].get("finished"))
            if meta["league"].get("winner") in ("A", "B"):
                row["winner"] = meta["league"]["winner"]
        if meta.get("seconds") is not None:
            row["duration_s"] = meta.get("seconds")
        # World Cup games carry neutral team labels and a winner that may have
        # been resolved on board score even when the engine winner is empty.
        if isinstance(meta.get("world_cup"), dict):
            row["world_cup"] = meta["world_cup"]
            winner_side = meta["world_cup"].get("winner_side")
            if winner_side in ("A", "B"):
                # Keep the dashboard's normal winner/learner_won fields
                # consistent with the league's board-score decision.
                row["winner"] = winner_side
        if isinstance(meta.get("team_labels"), dict):
            row["team_labels"] = meta["team_labels"]
    side = row.get("learner_side") or row.get("ncm_side")
    if side in ("A", "B"):
        row["learner_side"] = side
        row["learner_won"] = bool(row["winner"] and row["winner"] == side)
        other = "B" if side == "A" else "A"
        row["score_delta"] = round(
            save_board_stats(data, side)["score"]
            - save_board_stats(data, other)["score"],
            2,
        )
    _MATCH_CACHE[path.name] = (stamp, row)
    return row


def _match_rows() -> list[dict]:
    if not MATCH_DIR.is_dir():
        return []
    rows = []
    for path in sorted(MATCH_DIR.glob("*.json")):
        if path.name.endswith(".orders.json") or path.name.endswith(".meta.json"):
            continue
        row = _summarize_match(path)
        if row is not None:
            rows.append(row)
    return rows


def games_payload() -> dict:
    """Merge saved matches, arena games and self-play games into one list."""
    arena = _arena_rows()
    matches = _match_rows()
    index = {row["id"]: row for row in matches}

    # Attach arena metadata to saved games.
    for key, info in arena.items():
        row = index.get(key)
        if row is None:
            continue
        row = dict(row)
        row.update(
            {
                "source": "arena",
                "learner_side": info.get("learner_side"),
                "opponent_side": info.get("opponent_side"),
                "learner_won": info.get("learner_won"),
                "finished": info.get("finished"),
                "score_delta": info.get("score_delta"),
                "quality": {
                    "learner": info.get("learner"),
                    "opponent": info.get("opponent_stats"),
                },
                "label": info.get("arena_label") or row["label"],
                "arena_phase": info.get("arena_phase"),
                "arena_iteration": info.get("arena_iteration"),
                "arena_path": info.get("arena_path"),
                "board": info.get("board"),
            }
        )
        if info.get("turns") is not None:
            row["turns"] = info["turns"]
        if row.get("duration_s") is None:
            row["duration_s"] = info.get("seconds")
        index[key] = row

    # Arena games whose save file disappeared still belong in the list.
    for key, info in arena.items():
        if key in index:
            continue
        index[key] = {
            "id": key,
            "file": key if (MATCH_DIR / key).is_file() else None,
            "source": "arena",
            "scenario": "border_skirmish",
            "seed": None,
            "status": "finished" if info.get("finished") else "ongoing",
            "winner": None,
            "turns": info.get("turns"),
            "orders": info.get("orders"),
            "finished_at": info.get("finished_at"),
            "mtime": info.get("mtime") or 0,
            "duration_s": info.get("seconds"),
            "unit_counts": None,
            "has_replay": (MATCH_DIR / key).is_file(),
            "meta_path": None,
            "ncm_side": info.get("learner_side"),
            "opponent": "ncm",
            "quality": {"learner": info.get("learner"), "opponent": info.get("opponent_stats")},
            "learner_side": info.get("learner_side"),
            "opponent_side": info.get("opponent_side"),
            "learner_won": info.get("learner_won"),
            "finished": info.get("finished"),
            "score_delta": info.get("score_delta"),
            "label": info.get("arena_label") or key,
            "arena_phase": info.get("arena_phase"),
            "arena_iteration": info.get("arena_iteration"),
            "arena_path": info.get("arena_path"),
            "board": info.get("board"),
        }

    rows = [_decorate_display(dict(row)) for row in index.values()]
    rows.extend(_decorate_display(dict(row)) for row in _selfplay_rows())

    # Compute a score delta for saved play matches when the NCM side is known.
    for row in rows:
        if row.get("score_delta") is None and row.get("source") == "play":
            side = row.get("ncm_side")
            other = "B" if side == "A" else "A" if side == "B" else None
            if side and other and row.get("file"):
                save = _json(MATCH_DIR / row["file"])
                if isinstance(save, dict):
                    row["score_delta"] = round(
                        save_board_stats(save, side)["score"]
                        - save_board_stats(save, other)["score"],
                        2,
                    )

    rows.sort(
        key=lambda row: (
            row.get("mtime") or 0,
            str(row.get("finished_at") or ""),
            str(row.get("id") or ""),
        ),
        reverse=True,
    )
    batches = attach_batches(rows)
    reports = _reports()
    return {"games": rows, "reports": reports, "batches": batches, "count": len(rows)}


# ---------------------------------------------------------------------------
# match detail / replay


def _tile_list(state) -> list[dict]:
    tiles = sorted(state.map.tiles.values(), key=lambda tile: (tile.pos[1], tile.pos[0]))
    return [
        {
            "pos": [tile.pos[0], tile.pos[1]],
            "terrain": tile.terrain,
            "objective": bool(tile.objective),
            "capture_points": tile.capture_points,
        }
        for tile in tiles
    ]


def _unit_list(state) -> list[list]:
    units = []
    for unit in sorted(state.units.values(), key=lambda item: item.id):
        if unit.pos is None:
            continue
        units.append(
            [
                unit.id,
                unit.owner,
                unit.type_id,
                int(unit.pos[0]),
                int(unit.pos[1]),
                int(unit.hp),
                1 if unit.acted else 0,
                1 if unit.loaded else 0,
                1 if unit.deployed_this_turn else 0,
            ]
        )
    return units


def _owner_string(state, tiles: list[dict]) -> str:
    out = []
    for tile in tiles:
        pos = tuple(tile["pos"])
        tile_state = state.map.tiles.get(pos)
        out.append((tile_state.owner if tile_state is not None and tile_state.owner else "."))
    return "".join(out)


def _unit_meta() -> dict[str, dict]:
    result = {}
    for type_id, spec in specs().units.items():
        result[type_id] = {
            "name": spec.name,
            "glyph": spec.glyph,
            "max_hp": spec.max_hp,
            "domain": spec.domain,
        }
    return result


def _frame(state, seq: int, tiles: list[dict], order: dict, log_entry: dict | None) -> dict:
    entry = log_entry or {}
    return {
        "seq": seq,
        "turn": entry.get("turn", state.turn_number),
        "player": entry.get("player") or state.current_player,
        "action": entry.get("action") or order,
        "describe": entry.get("describe") or "",
        "events": entry.get("events") or [],
        "units": _unit_list(state),
        "owners": _owner_string(state, tiles),
        "funds": {player: state.players[player].funds for player in state.players},
        "status": state.status,
        "winner": state.winner,
    }


def _timeline_entry(state, seq: int, player: str, turn: int | None = None) -> dict:
    return {
        "seq": seq,
        "turn": state.turn_number if turn is None else turn,
        "player": player,
        "A": board_stats(state, "A"),
        "B": board_stats(state, "B"),
        "status": state.status,
        "winner": state.winner,
    }


def _markers(log: list[dict]) -> list[dict]:
    markers = []
    interesting = {"destroyed", "captured", "objective_captured", "eliminated"}
    for entry in log:
        for event in entry.get("events", []) or []:
            kind = event.get("type")
            if kind not in interesting:
                continue
            text = entry.get("describe") or kind
            if kind == "destroyed":
                unit = event.get("unit")
                if unit:
                    text = f"{unit} destroyed"
            elif kind == "captured":
                site = event.get("site")
                owner = event.get("new_owner") or event.get("owner")
                text = f"site {site} captured by {owner}"
            elif kind == "objective_captured":
                text = f"objective {event.get('site')} taken — {event.get('winner')} wins"
            elif kind == "eliminated":
                text = f"player {event.get('player')} eliminated"
            markers.append(
                {
                    "seq": entry.get("seq", 0),
                    "turn": entry.get("turn"),
                    "player": entry.get("player"),
                    "type": kind,
                    "text": text,
                }
            )
    return markers


def _quality_payload(meta: dict, row: dict | None) -> dict:
    history = meta.get("history") if isinstance(meta, dict) else None
    if not isinstance(history, list):
        history = []
    learner = (meta.get("learner") if isinstance(meta, dict) else None) or {}
    opponent = (meta.get("opponent") if isinstance(meta, dict) else None) or {}
    if row and row.get("quality"):
        learner = learner or row["quality"].get("learner") or {}
        opponent = opponent or row["quality"].get("opponent") or {}
    return {
        "history": history,
        "learner": learner,
        "opponent": opponent,
        "fallback_reasons": (meta.get("fallback_reasons") if isinstance(meta, dict) else None),
    }


def match_detail_payload(file_name: str) -> dict:
    """Replay one saved game into a compact board/timeline payload.

    ``file_name`` is a basename under ``artifacts/tbs/matches``.  The caller
    validates it before reaching this function; the checks here protect the
    endpoint as well.
    """
    name = Path(str(file_name)).name
    save_path = (MATCH_DIR / name).resolve()
    if MATCH_DIR.resolve() not in save_path.parents or not save_path.is_file():
        return {"error": "unknown match"}
    if name.endswith(".orders.json") or name.endswith(".meta.json"):
        return {"error": "not a match save"}

    orders_path = _orders_path(save_path)
    meta = _meta_for(save_path)
    key = str(save_path)
    stamp = _stamp(save_path, orders_path, _meta_path(save_path))
    cached = _DETAIL_CACHE.get(key)
    if cached and cached[0] == stamp:
        _DETAIL_CACHE.move_to_end(key)
        return cached[1]

    save = _json(save_path)
    if not isinstance(save, dict):
        return {"error": "cannot read match save"}
    name_scenario = save.get("scenario")
    scenario_path = resolve_scenario(name_scenario, meta)
    if scenario_path is None:
        return {
            "error": f"scenario {name_scenario!r} not found; cannot replay this game",
            "match": _summarize_match(save_path),
        }
    if not orders_path.is_file():
        return {
            "error": "no .orders.json sidecar; board replay is unavailable",
            "match": _summarize_match(save_path),
        }
    orders = _json(orders_path, [])
    if not isinstance(orders, list) or not orders:
        return {"error": "order sidecar is empty", "match": _summarize_match(save_path)}
    log = save.get("log") if isinstance(save.get("log"), list) else []

    scenario = load_scenario(scenario_path)
    scenario["seed"] = save.get("seed", scenario.get("seed"))
    try:
        state = new_game(scenario, specs())
        rules.start_game(state)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the server
        return {"error": f"cannot start scenario: {exc}", "match": _summarize_match(save_path)}

    tiles = _tile_list(state)
    frames = [_frame(state, -1, tiles, {}, None)]
    timeline = [_timeline_entry(state, -1, state.current_player)]
    last_timeline_seq = -1
    for index, order in enumerate(orders):
        try:
            rules.perform(state, order)
        except Exception as exc:  # noqa: BLE001 - a corrupt sidecar should not kill the page
            return {
                "error": f"replay failed at order {index}: {exc}",
                "match": _summarize_match(save_path),
                "frames": frames,
            }
        entry = log[index] if index < len(log) and isinstance(log[index], dict) else None
        frames.append(_frame(state, index, tiles, order if isinstance(order, dict) else {}, entry))
        if isinstance(order, dict) and order.get("action") == "end_turn":
            timeline.append(
                _timeline_entry(
                    state,
                    index,
                    (entry or {}).get("player") or state.current_player,
                    (entry or {}).get("turn"),
                )
            )
            last_timeline_seq = index
    if last_timeline_seq != len(frames) - 2:
        timeline.append(_timeline_entry(state, len(frames) - 2, state.current_player))

    try:
        rebuilt_ok = state_hash(state) == state_hash(load_game(save_path, specs()))
    except Exception:
        rebuilt_ok = None

    row = _summarize_match(save_path)
    arena_info = _arena_rows().get(name) or {}
    if row is not None and arena_info:
        row = dict(row)
        row.update(
            {
                "source": "arena",
                "learner_side": arena_info.get("learner_side"),
                "opponent_side": arena_info.get("opponent_side"),
                "learner_won": arena_info.get("learner_won"),
                "finished": arena_info.get("finished"),
                "score_delta": arena_info.get("score_delta"),
                "label": arena_info.get("arena_label") or row.get("label"),
                "arena_phase": arena_info.get("arena_phase"),
                "arena_iteration": arena_info.get("arena_iteration"),
                "arena_path": arena_info.get("arena_path"),
                "quality": {
                    "learner": arena_info.get("learner"),
                    "opponent": arena_info.get("opponent_stats"),
                },
            }
        )
        if row.get("score_delta") is None and row.get("learner_side") in ("A", "B"):
            side = row["learner_side"]
            other = "B" if side == "A" else "A"
            row["score_delta"] = round(
                save_board_stats(save, side)["score"] - save_board_stats(save, other)["score"],
                2,
            )
    if row is not None:
        row = _decorate_display(dict(row))
        attach_batches([row])
    quality = _quality_payload(meta, row)
    if not quality["learner"] and arena_info:
        quality["learner"] = arena_info.get("learner") or {}
        quality["opponent"] = arena_info.get("opponent_stats") or {}
    payload = {
        "match": row,
        "scenario": name_scenario,
        "scenario_path": str(scenario_path),
        "seed": save.get("seed"),
        "tiles": tiles,
        "unit_meta": _unit_meta(),
        "frames": frames,
        "timeline": timeline,
        "markers": _markers(log),
        "quality": quality,
        "replay_matches": rebuilt_ok,
        "n_orders": len(orders),
    }
    _DETAIL_CACHE[key] = (stamp, payload)
    _DETAIL_CACHE.move_to_end(key)
    while len(_DETAIL_CACHE) > _MAX_DETAIL_CACHE:
        _DETAIL_CACHE.popitem(last=False)
    return payload
