"""Generate push-labelled NCM rows against randomized conventional AI opponents.

    python -m tbs_ncm.generate_heuristic_data --games-per-map 2 --max-turns 40

Each persona is the learner in one side of every game; the opponent is a seeded
:class:`~tbs_ncm.heuristic.HeuristicAgent` (randomized scripted policy).  Rows
are written per persona and contain only learner-side states, labelled by the
push-biased persona teacher.  This gives the NCM training data on every map with
longer games and a varied non-NCM opponent.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .common import DATA_DIR
from .generate_data import MAX_STEPS_PER_TURN, dedupe, units_can_act
from .personas import PersonaTeacher, persona_names, push_persona, random_persona

import tbs_ncm.common  # noqa: F401

from tbs.agents.codec import encode_order, encode_state
from tbs.agents.protocol import AgentPlan
from tbs.agents.turn import apply_plan, build_request
from tbs.engine import rules
from tbs.engine.data import available_scenarios, load_scenario, load_specs
from tbs.engine.state import new_game

SPECS = load_specs()


def _resolve_maps(value: str) -> list[Path]:
    available = [Path(path) for path in available_scenarios()]
    if value.strip().lower() in ("", "all", "*"):
        return available
    wanted = {token.strip() for token in value.split(",") if token.strip()}
    return [path for path in available if path.stem in wanted or path.name in wanted]


def play_game(
    *,
    persona: str,
    learner_side: str,
    opponent_seed: int,
    seed: int,
    scenario_path: Path,
    max_turns: int,
    max_steps: int,
    push: bool,
    record_legal: bool,
) -> tuple[list[dict], dict]:
    scenario = load_scenario(str(scenario_path))
    scenario["seed"] = seed
    state = new_game(scenario, SPECS)
    rules.start_game(state)
    other = "A" if learner_side == "B" else "B"
    learner = PersonaTeacher(
        push_persona(persona) if push else persona, seed=seed * 2 + 1, push=push
    )
    opponent_persona = random_persona(opponent_seed)
    opponent = PersonaTeacher(opponent_persona, seed=seed * 2 + 2)
    teachers = {learner_side: learner, other: opponent}
    rows: list[dict] = []
    limit = min(int(max_turns), int(scenario.get("max_turns") or max_turns))
    turns = 0
    rejected = 0

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
            if player == learner_side:
                target = encode_order(order)
                row = {
                    "prefix": encode_state(request.view),
                    "target": target,
                    "persona": persona,
                    "seed": seed,
                    "turn": request.view["turn"],
                    "player": player,
                    "opponent_seed": opponent_seed,
                    "scenario": state.scenario_name,
                    "kind": "persona",
                    "source": "heuristic",
                }
                if record_legal:
                    from .legal import legal_lines

                    row["legal"] = legal_lines(request.view)
                rows.append(row)
            result = apply_plan(state, AgentPlan(orders=[order]))
            if not result.applied:
                rejected += 1
                break
        if state.status != "ongoing":
            break
        if state.current_player == player:
            rules.perform(state, {"action": "end_turn"})
        turns += 1

    winner = state.winner or ""
    for row in rows:
        row["winner"] = winner
        row["status"] = state.status
    stats = {
        "persona": persona,
        "seed": seed,
        "scenario": state.scenario_name,
        "learner_side": learner_side,
        "opponent_seed": opponent_seed,
        "opponent_params": asdict(opponent_persona),
        "turns": state.turn_number,
        "status": state.status,
        "winner": winner,
        "examples": len(rows),
        "rejected": rejected,
        "actions": dict(Counter(str(row["target"]).split()[0] for row in rows)),
    }
    return rows, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="push data vs randomized heuristics")
    parser.add_argument("--personas", default=",".join(persona_names()))
    parser.add_argument("--maps", default="all", help="all or comma-separated map names")
    parser.add_argument("--games-per-map", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=800000)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_TURN)
    parser.add_argument("--out-dir", default=str(DATA_DIR / "personas_heuristic"))
    parser.add_argument("--push", action="store_true", help="use push-biased labels")
    parser.add_argument("--record-legal", action="store_true")
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.personas.split(",") if name.strip()]
    maps = _resolve_maps(args.maps)
    if not names:
        parser.error("no personas selected")
    if not maps:
        parser.error("no maps matched")

    out_dir = Path(args.out_dir)
    started = time.perf_counter()
    summaries = []
    for persona_index, name in enumerate(names):
        rows: list[dict] = []
        games: list[dict] = []
        for map_index, map_path in enumerate(maps):
            for game in range(args.games_per_map):
                seed = args.seed_base + persona_index * 1000 + map_index * 10 + game
                opponent_seed = seed + 500000
                learner_side = "B" if (game + map_index) % 2 == 0 else "A"
                game_rows, stats = play_game(
                    persona=name,
                    learner_side=learner_side,
                    opponent_seed=opponent_seed,
                    seed=seed,
                    scenario_path=map_path,
                    max_turns=args.max_turns,
                    max_steps=args.max_steps,
                    push=args.push,
                    record_legal=args.record_legal,
                )
                rows.extend(game_rows)
                games.append(stats)
                print(
                    f"  {name} vs heuristic-{opponent_seed} {map_path.stem} "
                    f"seed {seed}: {stats['examples']} rows, "
                    f"{stats['turns']} turns, {stats['status']} "
                    f"{stats['winner'] or ''}",
                    flush=True,
                )
        rows = dedupe(rows)
        target = out_dir / name / "train.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
        summary = {
            "persona": name,
            "push": args.push,
            "rows": len(rows),
            "games": games,
            "maps": [path.stem for path in maps],
            "actions": dict(Counter(str(row["target"]).split()[0] for row in rows)),
            "seconds": round(time.perf_counter() - started, 1),
        }
        (target.parent / "train.summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(summary)
        print(
            f"{name}: {len(rows)} rows, actions {summary['actions']} -> {target}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
