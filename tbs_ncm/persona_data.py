"""Generate persona-labelled (state, order) rows from scripted teacher games.

Each game has two scripted personas, one per side.  Both sides' accepted orders
are written, each into its own persona dataset, tagged with the map, opponent and
seed.  Because the two teachers play each other, every persona sees on-policy
states against a variety of doctrines instead of only greedy lines.

    python -m tbs_ncm.persona_data --games 24 --val-games 3 --test-games 3
    python -m tbs_ncm.persona_data --scenarios all --append

The output layout is what ``train_generator`` already consumes::

    tbs_ncm/data/personas/<persona>/{train,validation,test}.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

from .common import DATA_DIR
from .generate_data import MAX_STEPS_PER_TURN, dedupe, legal_lines, units_can_act
from .monitor import RunLogger
from .personas import PERSONAS, PersonaTeacher, get_persona, persona_names

import tbs_ncm.common  # noqa: F401  (adds the game repo to sys.path)

from tbs.agents.codec import encode_order, encode_state
from tbs.agents.protocol import AgentPlan
from tbs.agents.turn import apply_plan, build_request
from tbs.engine import rules
from tbs.engine.data import available_scenarios, load_scenario, load_specs
from tbs.engine.state import new_game

SPECS = load_specs()
DEFAULT_MAX_TURNS = 40
DEFAULT_DATA_DIR = DATA_DIR / "personas"

__all__ = ["main", "play_persona_game", "resolve_scenarios"]


def resolve_scenarios(value: str) -> list[Path]:
    """Turn ``all``, a map name, or a comma-separated list into scenario paths."""
    available = [Path(path) for path in available_scenarios()]
    if value.strip().lower() in ("", "all", "*"):
        return available
    wanted = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        path = Path(token)
        if path.exists():
            wanted.append(path)
            continue
        matches = [
            candidate
            for candidate in available
            if candidate.stem == token or candidate.name == token
        ]
        if not matches:
            raise SystemExit(
                f"unknown scenario {token!r}; available: "
                + ", ".join(candidate.stem for candidate in available)
            )
        wanted.extend(matches)
    return wanted


def play_persona_game(
    seed: int,
    persona_a: str,
    persona_b: str,
    *,
    scenario_path: str | Path,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_steps: int = MAX_STEPS_PER_TURN,
    record_legal: bool = False,
    teacher_seed: int | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Play one persona-vs-persona game and return rows for both sides."""
    scenario = load_scenario(str(scenario_path))
    scenario["seed"] = seed
    state = new_game(scenario, SPECS)
    rules.start_game(state)
    teachers = {
        "A": PersonaTeacher(persona_a, seed=(teacher_seed or seed) * 2 + 1),
        "B": PersonaTeacher(persona_b, seed=(teacher_seed or seed) * 2 + 2),
    }
    persona_of = {"A": persona_a, "B": persona_b}
    rows: dict[str, list[dict]] = {name: [] for name in {persona_a, persona_b}}
    limit = min(int(max_turns), int(scenario.get("max_turns") or max_turns))
    rejected = 0
    steps = 0

    while state.status == "ongoing" and state.turn_number <= limit:
        player = state.current_player
        teacher = teachers[player]
        for _step in range(max_steps):
            if state.status != "ongoing" or state.current_player != player:
                break
            if not units_can_act(state, player):
                break
            request = build_request(state, player, history=0)
            plan = teacher.plan(request)
            if not plan.orders:
                break
            order = plan.orders[0]
            prefix = encode_state(request.view)
            target = encode_order(order)
            row = {
                "prefix": prefix,
                "target": target,
                "persona": persona_of[player],
                "seed": seed,
                "teacher_seed": teacher_seed if teacher_seed is not None else seed,
                "turn": request.view["turn"],
                "player": player,
                "opponent_persona": persona_of["B" if player == "A" else "A"],
                "scenario": state.scenario_name,
                "kind": "persona",
            }
            if record_legal:
                row["legal"] = legal_lines(request.view)
            result = apply_plan(state, AgentPlan(orders=[order]))
            if not result.applied:
                rejected += 1
                break
            rows[persona_of[player]].append(row)
            steps += 1
        if state.status != "ongoing":
            break
        if state.current_player == player:
            rules.perform(state, {"action": "end_turn"})

    stats = {
        "seed": seed,
        "scenario": state.scenario_name,
        "personas": {"A": persona_a, "B": persona_b},
        "status": state.status,
        "winner": state.winner or "",
        "turns": state.turn_number,
        "steps": steps,
        "rejected": rejected,
        "rows": {name: len(items) for name, items in rows.items()},
    }
    for name, items in rows.items():
        for row in items:
            row["winner"] = state.winner or ""
            row["learner_won"] = bool(state.winner and state.winner == row["player"])
            row["status"] = state.status
    return rows, stats


def _job(args: tuple) -> tuple[dict[str, list[dict]], list[dict]]:
    personas, seeds, scenario_paths, max_turns, max_steps, record_legal = args
    rows: dict[str, list[dict]] = {name: [] for name in set(personas)}
    games: list[dict] = []
    for index, seed in enumerate(seeds):
        scenario_path = scenario_paths[index % len(scenario_paths)]
        game_rows, stats = play_persona_game(
            seed,
            personas[0],
            personas[1],
            scenario_path=scenario_path,
            max_turns=max_turns,
            max_steps=max_steps,
            record_legal=record_legal,
        )
        for name, items in game_rows.items():
            rows.setdefault(name, []).extend(items)
        games.append(stats)
    return rows, games


def _pairs(names: list[str]) -> list[tuple[str, str]]:
    return [(names[i], names[j]) for i in range(len(names)) for j in range(i, len(names))]


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_rows(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return len(rows)


def _cap_holds(rows: list[dict], cap: float, seed: int) -> list[dict]:
    """Keep at most ``cap`` fraction of hold orders.

    A teacher that is blocked or has nothing useful to do holds.  That is a
    legitimate action, but mirrors on defensive maps can be 90% holds, which
    would teach the NCM to pass instead of developing a style.  All non-hold
    rows are kept; holds are sampled down to the cap.
    """
    if cap <= 0 or cap >= 1 or not rows:
        return rows
    holds = [row for row in rows if str(row.get("target", "")).startswith("h ")]
    others = [row for row in rows if not str(row.get("target", "")).startswith("h ")]
    if not holds:
        return rows
    max_holds = int(len(others) * cap / (1.0 - cap))
    if len(holds) <= max_holds:
        return rows
    rng = random.Random(seed)
    kept = others + rng.sample(holds, max(0, max_holds))
    rng.shuffle(kept)
    return kept


def _summary(rows: list[dict], games: list[dict]) -> dict:
    legal_rows = [row for row in rows if isinstance(row.get("legal"), list)]
    return {
        "rows": len(rows),
        "games": len(games),
        "legal_rows": len(legal_rows),
        "legal_orders": sum(len(row["legal"]) for row in legal_rows),
        "actions": dict(Counter(str(row["target"]).split()[0] for row in rows)),
        "builds": dict(
            Counter(
                str(row["target"]).split()[2]
                for row in rows
                if str(row["target"]).startswith("b ") and len(str(row["target"]).split()) >= 3
            )
        ),
        "maps": dict(Counter(row.get("scenario", "") for row in rows)),
        "opponents": dict(Counter(row.get("opponent_persona", "") for row in rows)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbs_ncm.persona_data", description="generate persona-labelled NCM data"
    )
    parser.add_argument("--personas", default=",".join(persona_names()))
    parser.add_argument("--games", type=int, default=24, help="train games per cross-persona pair")
    parser.add_argument(
        "--self-games",
        type=int,
        default=0,
        help="train games for a persona mirror pair (default: --games)",
    )
    parser.add_argument("--val-games", type=int, default=3)
    parser.add_argument("--test-games", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--val-seed-base", type=int, default=900000)
    parser.add_argument("--test-seed-base", type=int, default=950000)
    parser.add_argument(
        "--scenarios",
        default="all",
        help="all, a comma-separated map list, or JSON paths",
    )
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_TURN)
    parser.add_argument(
        "--record-legal",
        action="store_true",
        help="also store every legal order per state (large; use for val/test only)",
    )
    parser.add_argument("--append", action="store_true", help="merge with existing split files")
    parser.add_argument(
        "--hold-cap",
        type=float,
        default=0.30,
        help="keep at most this fraction of hold orders per persona/split (0 disables)",
    )
    parser.add_argument(
        "--splits", default="train,validation,test", help="comma-separated subset"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--no-monitor", action="store_true")
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.personas.split(",") if name.strip()]
    for name in names:
        get_persona(name)
    wanted = {name.strip() for name in args.splits.split(",") if name.strip()}
    scenarios = resolve_scenarios(args.scenarios)
    if not scenarios:
        parser.error("no scenarios available")
    data_dir = Path(args.data_dir)
    pairs = _pairs(names)

    split_specs = {
        "train": (args.seed_base, args.games),
        "validation": (args.val_seed_base, args.val_games),
        "test": (args.test_seed_base, args.test_games),
    }
    jobs = []
    for name in ("train", "validation", "test"):
        if name not in wanted:
            continue
        base, count = split_specs[name]
        for pair_index, pair in enumerate(pairs):
            games = count
            if name == "train" and args.self_games and pair[0] == pair[1]:
                games = args.self_games
            seeds = [base + pair_index * 1000 + index for index in range(games)]
            paths = [
                scenarios[(pair_index + index) % len(scenarios)]
                for index in range(games)
            ]
            jobs.append(
                (name, pair, seeds, paths, args.max_turns, args.max_steps, args.record_legal)
            )

    total_games = sum(len(job[2]) for job in jobs)
    run = (
        None
        if args.no_monitor
        else RunLogger(
            "persona_data",
            runs_dir=args.runs_dir or None,
            argv=argv,
            total_steps=total_games,
            label=",".join(names),
        )
    )
    started = time.perf_counter()
    split_rows: dict[str, dict[str, list[dict]]] = {
        name: {persona: [] for persona in names} for name in wanted
    }
    split_games: dict[str, list[dict]] = {name: [] for name in wanted}
    try:
        with (run if run else nullcontext()):
            if args.workers > 1:
                results = []
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = {
                        pool.submit(_job, (job[1], job[2], [str(p) for p in job[3]], job[4], job[5], job[6])): job
                        for job in jobs
                    }
                    for future in as_completed(futures):
                        job = futures[future]
                        results.append((job, future.result()))
                for job, (rows, games) in results:
                    split = job[0]
                    for persona, items in rows.items():
                        split_rows[split].setdefault(persona, []).extend(items)
                    split_games[split].extend(games)
                    print(f"  {job[1][0]} vs {job[1][1]} ({split}): {sum(len(v) for v in rows.values())} rows")
            else:
                done_games = 0
                for name, pair, seeds, paths, max_turns, max_steps, record_legal in jobs:
                    rows: dict[str, list[dict]] = {}
                    games: list[dict] = []
                    for index, seed in enumerate(seeds):
                        game_rows, stats = play_persona_game(
                            seed,
                            pair[0],
                            pair[1],
                            scenario_path=paths[index],
                            max_turns=max_turns,
                            max_steps=max_steps,
                            record_legal=record_legal,
                        )
                        for persona, items in game_rows.items():
                            rows.setdefault(persona, []).extend(items)
                        games.append(stats)
                        done_games += 1
                        if run is not None:
                            run.log(
                                step=done_games,
                                split=name,
                                persona_a=pair[0],
                                persona_b=pair[1],
                                seed=seed,
                                scenario=stats["scenario"],
                                rows=sum(len(v) for v in game_rows.values()),
                                turns=stats["turns"],
                                winner=stats["winner"],
                            )
                    for persona, items in rows.items():
                        split_rows[name].setdefault(persona, []).extend(items)
                    split_games[name].extend(games)
                    print(
                        f"  {pair[0]} vs {pair[1]} ({name}): "
                        f"{sum(len(v) for v in rows.values())} rows, "
                        f"{len(games)} games, {time.perf_counter() - started:.1f}s"
                    )

            top_summary = {"personas": {}, "scenarios": [str(p) for p in scenarios]}
            for split in wanted:
                for persona in names:
                    rows = dedupe(split_rows[split].get(persona, []))
                    target = data_dir / persona / f"{split}.jsonl"
                    if args.append:
                        rows = dedupe(_read_rows(target) + rows)
                    raw_count = len(rows)
                    rows = _cap_holds(
                        rows, args.hold_cap, seed=sum(ord(char) for char in persona) + len(split)
                    )
                    count = _write_rows(target, rows)
                    games = [game for game in split_games[split] if persona in game["personas"].values()]
                    summary = _summary(rows, games)
                    summary["raw_rows"] = raw_count
                    summary["hold_cap"] = args.hold_cap
                    (data_dir / persona / f"{split}.summary.json").write_text(
                        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                    )
                    top_summary["personas"].setdefault(persona, {})[split] = summary
                    print(
                        f"  {persona}/{split}: {count} rows "
                        f"(from {raw_count}) -> {target}"
                    )
            (data_dir / "persona_data.summary.json").write_text(
                json.dumps(top_summary, indent=2) + "\n", encoding="utf-8"
            )
            if run is not None:
                run.finish(
                    state="done",
                    games=total_games,
                    seconds=round(time.perf_counter() - started, 1),
                    personas=len(names),
                    scenarios=len(scenarios),
                )
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    print(f"\ndone in {time.perf_counter() - started:.1f}s -> {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
