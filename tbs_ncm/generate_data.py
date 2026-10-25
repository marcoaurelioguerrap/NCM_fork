"""Generate stepwise (state text, order line) examples by playing the game.

The policy is the repository's greedy agent, plus occasional builds and a small
amount of move noise so the dataset is not a single deterministic line.  Every
example is an order the engine accepted, encoded with the same compact codec the
NCM player will use at inference time.

    python -m tbs_ncm.generate_data --train-games 300 --val-games 30 --test-games 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from .common import DATA_DIR
from .legal import legal_lines
from .monitor import RunLogger

from tbs.agents.codec import encode_order, encode_state
from tbs.agents.greedy import GreedyAgent
from tbs.agents.protocol import AgentPlan
from tbs.agents.turn import apply_plan, build_request
from tbs.engine import rules
from tbs.engine.data import available_scenarios, load_scenario, load_specs
from tbs.engine.state import new_game

GREEDY = GreedyAgent()
SPECS = load_specs()
_AVAILABLE = [Path(path) for path in available_scenarios()]
SCENARIO_PATH = next(
    (path for path in _AVAILABLE if path.stem == "border_skirmish"),
    _AVAILABLE[0] if _AVAILABLE else Path("data/maps/border_skirmish.json"),
)

BUILD_PROB = 0.30
MOVE_EPS = 0.12
MAX_TURNS = 40
MAX_STEPS_PER_TURN = 64
MAX_EXAMPLES_PER_GAME = 600


def units_can_act(state, player: str) -> bool:
    return any(
        not unit.acted and not unit.loaded and not unit.deployed_this_turn
        for unit in state.units_of(player)
    )


def choose_order(request, rng: random.Random, allow_build: bool) -> dict | None:
    view = request.view
    player = request.player
    if allow_build and rng.random() < BUILD_PROB:
        affordable = [option for option in view.get("build_options", []) if option["affordable"]]
        if affordable:
            preference = ("tank", "artillery", "aa_missile", "infantry")
            choice = max(
                affordable,
                key=lambda option: (
                    preference.index(option["unit"]) if option["unit"] in preference else -1,
                    option["cost"],
                ),
            )
            return {
                "action": "build",
                "site": list(choice["site"]),
                "unit": choice["unit"],
            }

    plan = GREEDY.plan(request)
    if plan.orders:
        order = dict(plan.orders[0])
        if order.get("action") == "move" and rng.random() < MOVE_EPS:
            options = view.get("unit_actions", {}).get(order["unit"], {})
            moves = options.get("move_to") or []
            if moves:
                destination = rng.choice(moves)[0]
                order = {"action": "move", "unit": order["unit"], "to": list(destination)}
        return order
    return None


def play_game(seed: int, record_legal: bool = False) -> tuple[list[dict], dict]:
    scenario = load_scenario(SCENARIO_PATH)
    scenario["seed"] = seed
    state = new_game(scenario, SPECS)
    rules.start_game(state)
    rng = random.Random(seed)
    rows: list[dict] = []

    while state.status == "ongoing" and state.turn_number <= MAX_TURNS:
        player = state.current_player
        built = False
        for _step in range(MAX_STEPS_PER_TURN):
            if state.status != "ongoing" or state.current_player != player:
                break
            if not units_can_act(state, player):
                break
            request = build_request(state, player, history=0)
            order = choose_order(request, rng, allow_build=not built)
            if order is None:
                break
            if order.get("action") == "build":
                built = True
            prefix = encode_state(request.view)
            target = encode_order(order)
            result = apply_plan(state, AgentPlan(orders=[order]))
            if not result.applied:
                break
            row = {
                "prefix": prefix,
                "target": target,
                "seed": seed,
                "turn": request.view["turn"],
                "player": player,
            }
            if record_legal:
                row["legal"] = legal_lines(request.view)
            rows.append(row)
        if state.status != "ongoing":
            break
        rules.perform(state, {"action": "end_turn"})

    if len(rows) > MAX_EXAMPLES_PER_GAME:
        stride = len(rows) / MAX_EXAMPLES_PER_GAME
        rows = [rows[int(index * stride)] for index in range(MAX_EXAMPLES_PER_GAME)]

    stats = {
        "seed": seed,
        "status": state.status,
        "winner": state.winner,
        "turns": state.turn_number,
        "examples": len(rows),
        "actions": dict(Counter(row["target"].split()[0] for row in rows)),
    }
    return rows, stats


def dedupe(rows: list[dict]) -> list[dict]:
    kept: dict[tuple[str, str], dict] = {}
    unique: list[dict] = []
    for row in rows:
        key = (row["prefix"], row["target"])
        first = kept.get(key)
        if first is None:
            kept[key] = row
            unique.append(row)
            continue
        if "legal" in row:
            merged = set(first.get("legal") or []) | set(row["legal"])
            first["legal"] = sorted(merged)
    return unique


def write_split(
    name: str,
    seeds: range,
    record_legal: bool = False,
    run: RunLogger | None = None,
    counter: dict | None = None,
    data_dir: Path | None = None,
) -> dict:
    rows: list[dict] = []
    games: list[dict] = []
    started = time.perf_counter()
    for seed in seeds:
        game_rows, game_stats = play_game(seed, record_legal)
        rows.extend(game_rows)
        games.append(game_stats)
        if counter is not None:
            counter["games"] = counter.get("games", 0) + 1
            counter["examples"] = counter.get("examples", 0) + len(game_rows)
        print(
            f"  seed {seed:>5}: {game_stats['examples']:>4} examples  "
            f"{game_stats['turns']:>3} turns  {game_stats['status']}"
            f"{' winner ' + str(game_stats['winner']) if game_stats['winner'] else ''}"
        )
        if run is not None:
            elapsed = max(1e-6, time.perf_counter() - started)
            run.log(
                step=(counter or {}).get("games", 0),
                split=name,
                games=(counter or {}).get("games", 0),
                examples=(counter or {}).get("examples", 0),
                game_examples=game_stats["examples"],
                turns=game_stats["turns"],
                winner=game_stats["winner"] or "",
                examples_per_s=round(len(rows) / elapsed, 2),
            )
    unique = dedupe(rows)
    target_dir = data_dir or DATA_DIR
    path = target_dir / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in unique:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "split": name,
        "seeds": [seeds.start, seeds.stop - 1],
        "examples": len(unique),
        "raw_examples": len(rows),
        "duplicates": len(rows) - len(unique),
        "record_legal": record_legal,
        "games": games,
        "actions": dict(Counter(row["target"].split()[0] for row in unique)),
    }
    (target_dir / f"{name}.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{name}: {len(unique)} unique examples ({len(rows)} raw) -> {path}\n"
        f"  actions: {summary['actions']}"
    )
    if run is not None:
        run.log(
            kind="note",
            step=(counter or {}).get("games", 0),
            text=f"{name}: {len(unique)} unique examples "
            f"({len(rows)} raw, {len(rows) - len(unique)} duplicates), "
            f"actions {summary['actions']}",
            split=name,
            unique_examples=len(unique),
            duplicates=len(rows) - len(unique),
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="generate TBS order data for NCM")
    parser.add_argument("--train-games", type=int, default=300)
    parser.add_argument("--val-games", type=int, default=30)
    parser.add_argument("--test-games", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument(
        "--splits",
        default="train,validation,test",
        help="comma-separated subset of train,validation,test",
    )
    parser.add_argument(
        "--record-legal",
        action="store_true",
        help="store every legal order line per state (needed for the generator's "
        "legal-on-first-try metric; recommended for validation and test only)",
    )
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument(
        "--data-dir",
        default="",
        help="where to write the split files (default tbs_ncm/data)",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None
    wanted = {name.strip() for name in args.splits.split(",") if name.strip()}
    splits = {
        "train": range(args.seed_base + 1, args.seed_base + 1 + args.train_games),
        "validation": range(
            args.seed_base + 10000, args.seed_base + 10000 + args.val_games
        ),
        "test": range(args.seed_base + 20000, args.seed_base + 20000 + args.test_games),
    }
    total_games = sum(len(splits[name]) for name in wanted if name in splits)
    run = (
        None
        if args.no_monitor
        else RunLogger(
            "data",
            runs_dir=args.runs_dir or None,
            argv=sys.argv,
            total_steps=total_games,
            label="+".join(sorted(wanted)),
        )
    )
    counter = {"games": 0, "examples": 0}
    try:
        with (run if run else nullcontext()):
            for name in ("train", "validation", "test"):
                if name not in wanted:
                    continue
                print(f"generating {name} ...")
                write_split(
                    name,
                    splits[name],
                    record_legal=args.record_legal,
                    run=run,
                    counter=counter,
                    data_dir=data_dir,
                )
            if run is not None:
                run.finish(
                    state="done",
                    games=counter["games"],
                    examples=counter["examples"],
                )
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
