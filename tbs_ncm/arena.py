"""Headless matches between two NCM checkpoints.

The released ``play.py`` plays one NCM checkpoint against greedy or an LLM.
This module plays NCM against NCM -- either the same checkpoint in a mirror
match or a learner against a frozen opponent -- and records every match in the
same JSON shape the dashboard already reads.

    python -m tbs_ncm.arena --games 8 --seed-base 50000 \
        --learner-checkpoint-dir artifacts/tbs \
        --opponent-checkpoint-dir artifacts/tbs

``train_vs_ncm`` uses :func:`run_arena` for its before/after win-rate check.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from .common import ARTIFACT_DIR
from .generate_data import MAX_STEPS_PER_TURN, MAX_TURNS, SCENARIO_PATH, SPECS
from .ncm_agent import NcmAgent, NcmSampler
from .personas import PersonaTeacher
from .monitor import RunLogger, git_revision

from tbs.agents.greedy import GreedyAgent
from tbs.agents.turn import run_turn_stepwise
from tbs.engine import rules
from tbs.engine.data import load_scenario
from tbs.engine.serialize import replay, save_game, state_hash
from tbs.engine.state import new_game

__all__ = [
    "load_agent",
    "play_match",
    "reset_agent",
    "run_arena",
    "write_orders",
]


def load_agent(
    checkpoint_dir: Path | str,
    *,
    device: str | None = None,
    temperature: float = 0.0,
    retries: int = 6,
    retry_temperature: float = 0.5,
    prefix_len: int | None = 2048,
    generator: str | Path | None = None,
    fallback: str = "greedy",
    persona: str | None = None,
    decode: str = "sample",
) -> NcmAgent:
    """Load one checkpoint dir containing ``tokenizer.pt`` and ``generator.pt``.

    ``fallback="persona"`` replaces the greedy recovery policy with the
    persona's scripted teacher.  Without that, 40-60% of a weak model's orders
    are greedy, which washes the style out of the match-level numbers.

    ``prefix_len`` defaults to 2048 for inference.  The checkpoints were trained
    at 832, but RoPE lets the same weights run at a longer prefix; short-state
    outputs are unchanged and long states no longer fall back on context.
    """
    sampler = NcmSampler(
        checkpoint_dir,
        device=device,
        prefix_len=prefix_len,
        generator=generator,
    )
    recovery = GreedyAgent()
    if fallback == "persona":
        if not persona:
            raise ValueError("fallback='persona' needs a persona name")
        recovery = PersonaTeacher(persona)
    return NcmAgent(
        sampler,
        temperature=temperature,
        retries=retries,
        retry_temperature=retry_temperature,
        fallback=recovery,
        decode=decode,
    )


def reset_agent(agent: NcmAgent) -> None:
    """Clear the per-game counters and sampled lines."""
    agent.steps = 0
    agent.samples = 0
    agent.generated = 0
    agent.repaired = 0
    agent.repaired_samples = 0
    agent.constrained = 0
    agent.tactical = 0
    agent.fallbacks = 0
    agent.illegal = 0
    agent.rejected = 0
    agent.recovered = 0
    agent.state_too_long = 0
    agent.generation_errors = 0
    agent.lines.clear()
    agent.constrained_lines.clear()
    agent.fallback_reasons.clear()
    agent.repairs.clear()
    agent.actions.clear()
    agent.builds.clear()
    agent.orders = 0
    agent.model_actions.clear()
    agent.model_builds.clear()
    agent.model_orders = 0
    agent.geometry_samples = 0
    agent.enemy_distance = 0.0
    agent.home_distance = 0.0
    agent._last_line = ""
    agent._last_order = None


def _agent_stats(agent: NcmAgent) -> dict:
    total = max(1, agent.steps or agent.orders)
    order_total = max(1, agent.orders)
    return {
        "steps": agent.steps,
        "samples": agent.samples,
        "generated": agent.generated,
        "repaired": agent.repaired,
        "repaired_samples": agent.repaired_samples,
        "constrained": agent.constrained,
        "tactical": agent.tactical,
        "fallbacks": agent.fallbacks,
        "illegal": agent.illegal,
        "rejected": agent.rejected,
        "recovered": agent.recovered,
        "state_too_long": agent.state_too_long,
        "generation_errors": agent.generation_errors,
        "orders": agent.orders,
        "actions": dict(agent.actions),
        "builds": dict(agent.builds),
        "repairs": dict(agent.repairs),
        "model_orders": agent.model_orders,
        "model_actions": dict(agent.model_actions),
        "model_builds": dict(agent.model_builds),
        "attack_rate": agent.actions.get("attack", 0) / order_total,
        "capture_rate": agent.actions.get("capture", 0) / order_total,
        "hold_rate": agent.actions.get("hold", 0) / order_total,
        "build_rate": agent.actions.get("build", 0) / order_total,
        "model_attack_rate": (
            agent.model_actions.get("attack", 0) / agent.model_orders
            if agent.model_orders
            else None
        ),
        "model_hold_rate": (
            agent.model_actions.get("hold", 0) / agent.model_orders
            if agent.model_orders
            else None
        ),
        "model_build_rate": (
            agent.model_actions.get("build", 0) / agent.model_orders
            if agent.model_orders
            else None
        ),
        "fallback_rate": agent.fallbacks / total,
        "fallback_rate_per_order": agent.fallbacks / order_total,
        "generated_share": agent.generated / total,
        "repaired_share": agent.repaired / total,
        "constrained_share": agent.constrained / total,
        "tactical_share": agent.tactical / total,
        "legal_share": (
            agent.generated + agent.repaired + agent.constrained
        ) / total,
        "model_share": agent.model_orders / total,
        "sample_legal_rate": agent.generated / max(1, agent.samples),
        "fallback_reasons": _reason_counts(agent),
        "mean_enemy_distance": (
            agent.enemy_distance / agent.geometry_samples
            if agent.geometry_samples
            else None
        ),
        "mean_home_distance": (
            agent.home_distance / agent.geometry_samples
            if agent.geometry_samples
            else None
        ),
    }



def _delta_stats(before: dict, after: dict) -> dict:
    deltas = {}
    for key, value in after.items():
        old = before.get(key, 0)
        if isinstance(value, (int, float)) and isinstance(old, (int, float)):
            deltas[key] = value - old
    return deltas


def _reason_key(reason: str) -> str:
    """Collapse detailed errors so a report has a handful of causes, not one per token count."""
    return str(reason).split(":", 1)[0].strip() or "unknown"


def _reason_counts(agent: NcmAgent) -> dict:
    return dict(Counter(_reason_key(reason) for reason in agent.fallback_reasons))



def _board_stats(state, player: str) -> dict:
    """A cap-safe board score: territory plus living units plus their hit points.

    Mirror NCM games often reach the turn cap without an HQ capture, so a raw
    win/loss is not a usable signal.  The score is a proxy for pressure and is
    reported next to the win rate; it is never treated as an actual win.
    """
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


def write_orders(path: Path, orders: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(orders, indent=2) + "\n", encoding="utf-8")


def play_match(
    learner: NcmAgent,
    opponent: NcmAgent,
    *,
    seed: int,
    learner_side: str = "B",
    max_turns: int = MAX_TURNS,
    max_steps: int = MAX_STEPS_PER_TURN,
    scenario_path: Path | str | None = None,
    out: Path | str | None = None,
    verify_replay: bool = True,
) -> dict:
    """Play one full game, learner on ``learner_side`` and opponent on the other.

    Both agents are :class:`~tbs_ncm.ncm_agent.NcmAgent` instances.  The game is
    driven by ``run_turn_stepwise`` so each side sees a fresh view after every
    order, exactly as it does in ``play.py``.  Every accepted order is collected
    for the replay check and for the dashboard's ``*.orders.json`` sidecar; the
    ``*.meta.json`` sidecar records per-turn generated/repaired/fallback counts.
    """
    if learner_side not in ("A", "B"):
        raise ValueError(f"learner_side must be 'A' or 'B', got {learner_side!r}")
    other = "A" if learner_side == "B" else "B"
    agents = {learner_side: learner, other: opponent}
    reset_agent(learner)
    reset_agent(opponent)

    scenario = load_scenario(str(scenario_path or SCENARIO_PATH))
    scenario["seed"] = seed
    state = new_game(scenario, SPECS)
    rules.start_game(state)

    orders: list[dict] = []
    history: list[dict] = []
    turns = 0
    started_at = time.time()
    started = time.perf_counter()
    while state.status == "ongoing" and state.turn_number <= max_turns:
        player = state.current_player
        turn = state.turn_number
        agent = agents[player]
        before = {side: _agent_stats(member) for side, member in agents.items()}
        result = run_turn_stepwise(state, agent, max_steps=max_steps)
        after = {side: _agent_stats(member) for side, member in agents.items()}
        orders.extend(result.orders)
        turns += 1
        history.append(
            {
                "turn": turn,
                "player": player,
                "orders": len(result.orders),
                "attempts": result.attempts,
                "board": {"A": _board_stats(state, "A"), "B": _board_stats(state, "B")},
                "agents": {
                    side: _delta_stats(before[side], after[side]) for side in agents
                },
            }
        )

    elapsed = time.perf_counter() - started
    finished_at = time.time()
    started_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at))
    finished_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(finished_at))
    learner_won = state.winner == learner_side
    won = state.winner
    board = {"A": _board_stats(state, "A"), "B": _board_stats(state, "B")}
    match = {
        "seed": seed,
        "started_at": started_text,
        "finished_at": finished_text,
        "scenario": state.scenario_name,
        "learner_side": learner_side,
        "opponent_side": other,
        "status": state.status,
        "winner": won or "",
        "learner_won": learner_won,
        "finished": state.status != "ongoing" and won in ("A", "B"),
        "turns": state.turn_number,
        "player_turns": turns,
        "orders": len(orders),
        "board": board,
        "learner_board": board[learner_side],
        "opponent_board": board[other],
        "score_delta": round(board[learner_side]["score"] - board[other]["score"], 2),
        "units": {
            "A": len(state.units_of("A")),
            "B": len(state.units_of("B")),
        },
        "learner": _agent_stats(learner),
        "opponent": _agent_stats(opponent),
        "seconds": round(elapsed, 2),
    }

    if verify_replay:
        rebuilt = replay(scenario, orders)
        match["replay_matches"] = state_hash(rebuilt) == state_hash(state)

    if out is not None:
        out_path = Path(out)
        save_game(state, out_path)
        write_orders(out_path.with_name(out_path.stem + ".orders.json"), orders)
        meta_path = out_path.with_name(out_path.stem + ".meta.json")
        meta = {
            "schema": 1,
            "source": "arena",
            "scenario": state.scenario_name,
            "scenario_path": str(scenario_path or SCENARIO_PATH),
            "seed": seed,
            "started_at": started_text,
            "finished_at": finished_text,
            "seconds": round(elapsed, 2),
            "learner_side": learner_side,
            "opponent_side": other,
            "learner": _agent_stats(learner),
            "opponent": _agent_stats(opponent),
            "fallback_reasons": {
                side: _reason_counts(member) for side, member in agents.items()
            },
            "history": history,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        match["save"] = str(out_path)
        match["orders_path"] = str(out_path.with_name(out_path.stem + ".orders.json"))
        match["meta_path"] = str(meta_path)

    return match


def run_arena(
    learner: NcmAgent,
    opponent: NcmAgent,
    *,
    seeds: list[int] | range,
    learner_side: str = "B",
    alternate_sides: bool = True,
    max_turns: int = MAX_TURNS,
    max_steps: int = MAX_STEPS_PER_TURN,
    scenario_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    label: str = "ncm-vs-ncm",
    verify_replay: bool = True,
    logger: RunLogger | None = None,
    log_every: int = 1,
) -> dict:
    """Play every seed and return win/loss/win-rate totals.

    With ``alternate_sides`` the learner plays B on even indexes and A on odd
    indexes, so a side-specific advantage does not masquerade as model strength.
    """
    seeds = list(seeds)
    if not seeds:
        raise ValueError("arena needs at least one seed")
    target_dir = Path(out_dir) if out_dir else ARTIFACT_DIR / "matches"
    target_dir.mkdir(parents=True, exist_ok=True)

    games: list[dict] = []
    started = time.perf_counter()
    for index, seed in enumerate(seeds):
        side = learner_side
        if alternate_sides and index % 2 == 1:
            side = "A" if learner_side == "B" else "B"
        out_path = target_dir / f"{label}-seed{seed}.json"
        match = play_match(
            learner,
            opponent,
            seed=seed,
            learner_side=side,
            max_turns=max_turns,
            max_steps=max_steps,
            scenario_path=scenario_path,
            out=out_path,
            verify_replay=verify_replay,
        )
        match["index"] = index
        games.append(match)
        if logger is not None and (index + 1) % max(1, log_every) == 0:
            wins = sum(game["learner_won"] for game in games)
            finished = sum(game["finished"] for game in games)
            logger.log(
                step=index + 1,
                seed=seed,
                learner_side=side,
                winner=match["winner"],
                learner_won=match["learner_won"],
                wins=wins,
                finished=finished,
                win_rate=(wins / finished) if finished else None,
                turns=match["turns"],
                orders=match["orders"],
                score_delta=match["score_delta"],
                generated=match["learner"]["generated"],
                repaired=match["learner"]["repaired"],
                fallbacks=match["learner"]["fallbacks"],
                seconds=match["seconds"],
            )
        print(
            f"  seed {seed:>6} side {side}: "
            f"{'WIN ' if match['learner_won'] else 'loss'}"
            f"{' (unfinished)' if not match['finished'] else ''}"
            f"  turns {match['turns']:>2}  orders {match['orders']:>3}  "
            f"score {match['score_delta']:+.1f}  "
            f"gen {match['learner']['generated']:>2} "
            f"rep {match['learner']['repaired']:>2} "
            f"fb {match['learner']['fallbacks']:>2}"
        )

    wins = sum(game["learner_won"] for game in games)
    finished = sum(game["finished"] for game in games)
    losses = sum(
        game["finished"] and not game["learner_won"] for game in games
    )
    count = max(1, len(games))
    reason_totals: Counter[str] = Counter()
    repair_totals: Counter[str] = Counter()
    for game in games:
        reason_totals.update(game["learner"].get("fallback_reasons", {}))
        repair_totals.update(game["learner"].get("repairs", {}))
    steps = sum(game["learner"].get("steps", game["learner"].get("orders", 0)) for game in games)
    generated = sum(game["learner"].get("generated", 0) for game in games)
    repaired = sum(game["learner"].get("repaired", 0) for game in games)
    fallbacks = sum(game["learner"].get("fallbacks", 0) for game in games)
    summary = {
        "label": label,
        "seeds": seeds,
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "unfinished": len(games) - finished,
        "win_rate": (wins / finished) if finished else 0.0,
        "finished_rate": finished / count,
        "avg_score_delta": round(
            sum(game["score_delta"] for game in games) / count, 2
        ),
        "avg_turns": round(sum(game["turns"] for game in games) / count, 1),
        "avg_orders": round(sum(game["orders"] for game in games) / count, 1),
        "steps": steps,
        "generated": generated,
        "repaired": repaired,
        "fallbacks": fallbacks,
        "fallback_rate": fallbacks / max(1, steps),
        "model_share": (generated + repaired) / max(1, steps),
        "fallback_reasons": dict(reason_totals),
        "repairs": dict(repair_totals),
        "learner_sides": sorted({game["learner_side"] for game in games}),
        "seconds": round(time.perf_counter() - started, 2),
        "git": git_revision(),
        "matches": games,
    }
    if logger is not None:
        logger.log(
            kind="arena",
            step=len(games),
            wins=wins,
            losses=losses,
            unfinished=len(games) - finished,
            win_rate=summary["win_rate"],
            finished_rate=summary["finished_rate"],
            avg_score_delta=summary["avg_score_delta"],
            avg_turns=summary["avg_turns"],
            fallback_rate=summary["fallback_rate"],
            model_share=summary["model_share"],
            seconds=summary["seconds"],
        )
    return summary



def _seeds(start: int, count: int) -> list[int]:
    return list(range(start, start + count))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbs_ncm.arena", description="play NCM against an NCM checkpoint"
    )
    parser.add_argument("--learner-checkpoint-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--opponent-checkpoint-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=50000)
    parser.add_argument("--learner-side", choices=("A", "B"), default="B")
    parser.add_argument("--no-alternate-sides", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--opponent-temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--retry-temperature", type=float, default=0.5)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--fallback", choices=("greedy", "persona"), default="greedy")
    parser.add_argument("--persona", default="", help="persona name for a persona fallback")
    parser.add_argument(
        "--decode",
        choices=("sample", "legal"),
        default="legal",
        help="sample: free generation + repair; legal: rank the view's legal orders",
    )
    parser.add_argument("--scenario", default="", help="scenario JSON path (default: first map)")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_TURN)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(ARTIFACT_DIR / "matches"))
    parser.add_argument("--label", default="ncm-vs-ncm")
    parser.add_argument("--no-replay-check", action="store_true")
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--no-monitor", action="store_true")
    args = parser.parse_args(argv)

    learner = load_agent(
        args.learner_checkpoint_dir,
        device=args.device,
        temperature=args.temperature,
        retries=args.retries,
        retry_temperature=args.retry_temperature,
        prefix_len=args.prefix_len,
        fallback=args.fallback,
        persona=args.persona or None,
        decode=args.decode,
    )
    opponent = load_agent(
        args.opponent_checkpoint_dir,
        device=args.device,
        temperature=args.opponent_temperature,
        retries=args.retries,
        retry_temperature=args.retry_temperature,
        prefix_len=args.prefix_len,
        fallback=args.fallback,
        persona=args.persona or None,
        decode=args.decode,
    )
    seeds = _seeds(args.seed_base, args.games)
    run = (
        None
        if args.no_monitor
        else RunLogger(
            "arena",
            runs_dir=args.runs_dir or None,
            argv=argv,
            total_steps=len(seeds),
            label=args.label,
        )
    )
    try:
        with (run if run else nullcontext()):
            summary = run_arena(
                learner,
                opponent,
                seeds=seeds,
                learner_side=args.learner_side,
                alternate_sides=not args.no_alternate_sides,
                max_turns=args.max_turns,
                max_steps=args.max_steps,
                scenario_path=args.scenario or None,
                out_dir=args.out_dir,
                label=args.label,
                verify_replay=not args.no_replay_check,
                logger=run,
            )
            print(
                f"\n{summary['games']} games: {summary['wins']} wins, "
                f"{summary['losses']} losses, {summary['unfinished']} unfinished, "
                f"win rate {summary['win_rate']:.1%} "
                f"(finished {summary['finished_rate']:.0%}), "
                f"avg score {summary['avg_score_delta']:+.1f}, "
                f"avg turns {summary['avg_turns']:.1f} "
                f"({summary['seconds']:.0f}s)"
            )
            print(
                f"learner: model share {summary['model_share']:.0%}, "
                f"fallback {summary['fallback_rate']:.0%}, "
                f"reasons {summary['fallback_reasons']}, "
                f"repairs {summary['repairs']}"
            )
            if run is not None:
                run.finish(
                    state="done",
                    **{
                        k: summary[k]
                        for k in (
                            "wins",
                            "losses",
                            "unfinished",
                            "win_rate",
                            "finished_rate",
                            "avg_score_delta",
                        )
                    },
                )
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
