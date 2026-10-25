"""Generate (state, order) training rows from NCM-vs-NCM games.

This is the self-play counterpart to ``generate_data.py``.  ``generate_data``
labels states with the scripted greedy agent; here the learner and the opponent
are NCM checkpoints.  The learner can play the same checkpoint (a mirror match)
or a frozen older one (a league/self-play round).

For every learner step the applied order is stored as the target.  Orders that
were sampled but rejected, repaired, or replaced by the greedy fallback are
tagged in ``kind``; ``--include-fallback`` controls whether the greedy fallbacks
become rows, and ``--teacher`` adds a second row per state labelled by the
scripted greedy agent (DAgger).

    python -m tbs_ncm.selfplay --games 40 --seed-base 40000 \
        --learner-checkpoint-dir artifacts/tbs \
        --opponent-checkpoint-dir artifacts/tbs \
        --temperature 0.8 --out artifacts/tbs/selfplay/selfplay_train.jsonl

The output schema is the one ``train_generator.py`` already consumes:
``prefix`` is the compact state text, ``target`` is the canonical order line.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from .arena import load_agent, reset_agent
from .common import ARTIFACT_DIR
from .generate_data import (
    MAX_STEPS_PER_TURN,
    MAX_TURNS,
    SCENARIO_PATH,
    SPECS,
    dedupe,
    units_can_act,
)
from .monitor import RunLogger
from .personas import PersonaTeacher

from tbs.agents.codec import encode_order, encode_state
from tbs.agents.greedy import GreedyAgent
from tbs.agents.protocol import AgentPlan
from tbs.agents.turn import apply_plan, build_request
from tbs.engine import rules
from tbs.engine.data import load_scenario
from tbs.engine.state import new_game

__all__ = ["collect_game", "main", "write_rows"]


def _provenance(agent, before: tuple[int, int, int]) -> str:
    """Classify what ``NcmAgent.plan`` just did: generated, repaired or fallback."""
    generated, repaired, fallbacks = before
    if agent.generated > generated:
        return "generated"
    if agent.repaired > repaired:
        return "repaired"
    if agent.fallbacks > fallbacks:
        return "fallback"
    return "none"


def collect_game(
    learner,
    opponent,
    *,
    seed: int,
    learner_side: str = "B",
    teacher: object | None = None,
    teacher_mode: str = "all",
    include_fallback: bool = False,
    max_turns: int = MAX_TURNS,
    max_steps: int = MAX_STEPS_PER_TURN,
    scenario_path: Path | str | None = None,
) -> tuple[list[dict], dict]:
    """Play one NCM-vs-NCM game and return the learner rows plus game stats.

    ``teacher_mode="all"`` labels every learner state (classic DAgger);
    ``teacher_mode="fallback"`` emits a teacher row only where the NCM agent
    had to use its scripted recovery policy.  The latter targets exactly the
    distribution that still fails, which is the point of a short fine-tune.
    """
    if learner_side not in ("A", "B"):
        raise ValueError(f"learner_side must be 'A' or 'B', got {learner_side!r}")
    if teacher_mode not in ("all", "fallback"):
        raise ValueError(f"teacher_mode must be 'all' or 'fallback', got {teacher_mode!r}")
    other = "A" if learner_side == "B" else "B"
    agents = {learner_side: learner, other: opponent}
    reset_agent(learner)
    reset_agent(opponent)

    scenario = load_scenario(str(scenario_path or SCENARIO_PATH))
    scenario["seed"] = seed
    state = new_game(scenario, SPECS)
    rules.start_game(state)

    rows: list[dict] = []
    started = time.perf_counter()
    while state.status == "ongoing" and state.turn_number <= max_turns:
        player = state.current_player
        agent = agents[player]
        for _step in range(max_steps):
            if state.status != "ongoing" or state.current_player != player:
                break
            if not units_can_act(state, player):
                break
            request = build_request(state, player, history=0)
            before = (
                getattr(agent, "generated", 0),
                getattr(agent, "repaired", 0),
                getattr(agent, "fallbacks", 0),
            )
            lines_before = len(getattr(agent, "lines", []))
            plan = agent.plan(request)
            lines = getattr(agent, "lines", [])
            sample = lines[-1] if len(lines) > lines_before else ""
            if not plan.orders:
                break

            order = plan.orders[0]
            if agent is learner:
                kind = _provenance(agent, before)
                wants_row = kind in ("generated", "repaired") or (
                    include_fallback and kind == "fallback"
                )
                wants_teacher = teacher is not None and (
                    teacher_mode == "all" or kind == "fallback"
                )
                if wants_row or wants_teacher:
                    prefix = encode_state(request.view)
                    target = encode_order(order)
                    base = {
                        "prefix": prefix,
                        "sample": sample,
                        "seed": seed,
                        "turn": request.view["turn"],
                        "player": player,
                        "learner_side": learner_side,
                        "opponent": "ncm",
                        "scenario": state.scenario_name,
                    }
                    if wants_row:
                        row = dict(base)
                        row.update({"target": target, "kind": kind})
                        rows.append(row)
                    if wants_teacher:
                        teacher_plan = teacher.plan(request)
                        if teacher_plan.orders:
                            teacher_target = encode_order(teacher_plan.orders[0])
                            teacher_row = dict(base)
                            teacher_row.update(
                                {
                                    "target": teacher_target,
                                    "sample": "",
                                    "kind": "teacher"
                                    if teacher_mode == "all"
                                    else "teacher-fallback",
                                    "teacher_for": kind,
                                }
                            )
                            rows.append(teacher_row)

            result = apply_plan(state, AgentPlan(orders=[order]))
            if not result.applied:
                break

        if state.status != "ongoing":
            break
        if state.current_player == player:
            rules.perform(state, {"action": "end_turn"})

    winner = state.winner or ""
    for row in rows:
        row["winner"] = winner
        row["learner_win"] = bool(winner and winner == learner_side)
        row["status"] = state.status

    stats = {
        "seed": seed,
        "scenario": state.scenario_name,
        "learner_side": learner_side,
        "status": state.status,
        "winner": winner,
        "learner_won": bool(winner and winner == learner_side),
        "turns": state.turn_number,
        "examples": len(rows),
        "kinds": dict(Counter(row["kind"] for row in rows)),
        "learner": {
            "generated": learner.generated,
            "repaired": learner.repaired,
            "fallbacks": learner.fallbacks,
            "illegal": learner.illegal,
            "generation_errors": learner.generation_errors,
        },
        "opponent": {
            "generated": opponent.generated,
            "repaired": opponent.repaired,
            "fallbacks": opponent.fallbacks,
            "illegal": opponent.illegal,
            "generation_errors": opponent.generation_errors,
        },
        "seconds": round(time.perf_counter() - started, 2),
    }
    return rows, stats


def expand_teacher_rows(rows: list[dict]) -> list[dict]:
    """Turn the per-row ``teacher`` label into additional DAgger training rows."""
    expanded: list[dict] = []
    for row in rows:
        expanded.append(row)
        teacher = row.get("teacher")
        if isinstance(teacher, str) and teacher and teacher != row["target"]:
            extra = dict(row)
            extra["target"] = teacher
            extra["sample"] = ""
            extra["kind"] = "teacher"
            expanded.append(extra)
    return expanded


def write_rows(path: Path | str, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def _sides(index: int, learner_side: str, alternate: bool) -> str:
    if not alternate or index % 2 == 0:
        return learner_side
    return "A" if learner_side == "B" else "B"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbs_ncm.selfplay", description="generate NCM-vs-NCM training data"
    )
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=40000)
    parser.add_argument("--learner-checkpoint-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--opponent-checkpoint-dir", default="")
    parser.add_argument("--learner-side", choices=("A", "B"), default="B")
    parser.add_argument("--no-alternate-sides", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--opponent-temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--retry-temperature", type=float, default=0.5)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument(
        "--teacher",
        action="store_true",
        help="also store a greedy-labelled row for every learner state (DAgger)",
    )
    parser.add_argument(
        "--teacher-persona",
        default="",
        help="use this scripted persona as the DAgger teacher instead of greedy",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=("all", "fallback"),
        default="all",
        help="label every state, or only states where the NCM had to fall back",
    )
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_TURN)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out",
        default=str(ARTIFACT_DIR / "selfplay" / "selfplay_train.jsonl"),
    )
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    learner = load_agent(
        args.learner_checkpoint_dir,
        device=args.device,
        temperature=args.temperature,
        retries=args.retries,
        retry_temperature=args.retry_temperature,
        prefix_len=args.prefix_len,
    )
    opponent = load_agent(
        args.opponent_checkpoint_dir or args.learner_checkpoint_dir,
        device=args.device,
        temperature=args.opponent_temperature,
        retries=args.retries,
        retry_temperature=args.retry_temperature,
        prefix_len=args.prefix_len,
    )
    if args.teacher_persona:
        teacher = PersonaTeacher(args.teacher_persona, seed=args.seed_base)
    elif args.teacher:
        teacher = GreedyAgent()
    else:
        teacher = None

    seeds = list(range(args.seed_base, args.seed_base + args.games))
    run = (
        None
        if args.no_monitor
        else RunLogger(
            "selfplay",
            runs_dir=args.runs_dir or None,
            argv=argv,
            total_steps=len(seeds),
            label=args.label or f"games{args.games}",
        )
    )
    rows: list[dict] = []
    games: list[dict] = []
    started = time.perf_counter()
    try:
        with (run if run else nullcontext()):
            for index, seed in enumerate(seeds):
                side = _sides(index, args.learner_side, not args.no_alternate_sides)
                game_rows, stats = collect_game(
                    learner,
                    opponent,
                    seed=seed,
                    learner_side=side,
                    teacher=teacher,
                    teacher_mode=args.teacher_mode,
                    include_fallback=args.include_fallback,
                    max_turns=args.max_turns,
                    max_steps=args.max_steps,
                )
                rows.extend(game_rows)
                games.append(stats)
                elapsed = max(1e-6, time.perf_counter() - started)
                print(
                    f"  seed {seed:>6} side {side}: {stats['examples']:>4} rows  "
                    f"{stats['turns']:>3} turns  {stats['status']}"
                    f"{' winner ' + stats['winner'] if stats['winner'] else ''}  "
                    f"kinds {stats['kinds']}"
                )
                if run is not None:
                    run.log(
                        step=index + 1,
                        seed=seed,
                        learner_side=side,
                        examples=len(game_rows),
                        game_examples=stats["examples"],
                        turns=stats["turns"],
                        winner=stats["winner"],
                        learner_won=stats["learner_won"],
                        generated=stats["learner"]["generated"],
                        repaired=stats["learner"]["repaired"],
                        fallbacks=stats["learner"]["fallbacks"],
                        opponent_generated=stats["opponent"]["generated"],
                        rows=len(rows),
                        examples_per_s=round(len(rows) / elapsed, 2),
                    )

            if teacher is not None:
                print(
                    f"teacher rows: {sum(1 for row in rows if row['kind'].startswith('teacher'))} "
                    f"({args.teacher_mode})"
                )

            unique = dedupe(rows)
            out_path = write_rows(args.out, unique)
            wins = sum(game["learner_won"] for game in games)
            finished = sum(game["status"] != "ongoing" for game in games)
            summary = {
                "stage": "selfplay",
                "learner_checkpoint_dir": str(args.learner_checkpoint_dir),
                "opponent_checkpoint_dir": str(
                    args.opponent_checkpoint_dir or args.learner_checkpoint_dir
                ),
                "temperature": args.temperature,
                "opponent_temperature": args.opponent_temperature,
                "teacher": bool(teacher),
                "teacher_persona": args.teacher_persona or ("greedy" if teacher else ""),
                "teacher_mode": args.teacher_mode if teacher else "",
                "include_fallback": args.include_fallback,
                "games": games,
                "examples": len(unique),
                "raw_examples": len(rows),
                "duplicates": len(rows) - len(unique),
                "kinds": dict(Counter(row["kind"] for row in unique)),
                "wins": wins,
                "finished": finished,
                "win_rate": (wins / finished) if finished else 0.0,
                "seconds": round(time.perf_counter() - started, 2),
                "out": str(out_path),
            }
            summary_path = out_path.with_suffix(".summary.json")
            summary_path.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"\n{len(unique)} unique rows ({len(rows)} raw) -> {out_path}\n"
                f"  kinds: {summary['kinds']}\n"
                f"  learner record: {wins}W/{finished - wins}L "
                f"({summary['win_rate']:.1%} of finished)  "
                f"{summary['seconds']:.0f}s"
            )
            if run is not None:
                run.finish(
                    state="done",
                    examples=len(unique),
                    duplicates=len(rows) - len(unique),
                    wins=wins,
                    finished=finished,
                    win_rate=summary["win_rate"],
                    seconds=summary["seconds"],
                )
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
