"""Headless NCM-vs-opponent match with a replayable order log.

    python -m tbs_ncm.play --opponent greedy --ncm-side B --seed 7

    python -m tbs_ncm.play --opponent llm \
        --llm-model "unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from .common import ARTIFACT_DIR

from tbs.agents.council import CouncilAgent
from tbs.agents.generals import DEFAULT_GENERALS, load_generals, select_generals
from tbs.agents.greedy import GreedyAgent
from tbs.agents.llm import (
    DEFAULT_BASE_URL,
    JsonlLogger,
    LlmClient,
    LlmConfig,
    LlmTransportError,
    SingleLlmAgent,
)
from tbs.agents.turn import run_turn, run_turn_stepwise
from tbs.engine import rules
from tbs.engine.data import available_scenarios, load_scenario, load_specs
from tbs.engine.serialize import replay, save_game, state_hash
from tbs.engine.state import new_game

from .ncm_agent import NcmAgent, NcmSampler
from .personas import PersonaTeacher
from .generate_data import SCENARIO_PATH


def _orders_path(out: Path) -> Path:
    return out.with_name(out.stem + ".orders.json")


def _board_stats(state, player: str) -> dict:
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


def _ncm_stats(agent: NcmAgent) -> dict:
    steps = max(1, agent.steps or agent.orders)
    return {
        "steps": agent.steps,
        "samples": agent.samples,
        "generated": agent.generated,
        "repaired": agent.repaired,
        "constrained": agent.constrained,
        "tactical": agent.tactical,
        "fallbacks": agent.fallbacks,
        "illegal": agent.illegal,
        "rejected": agent.rejected,
        "recovered": agent.recovered,
        "state_too_long": agent.state_too_long,
        "generation_errors": agent.generation_errors,
        "repairs": dict(agent.repairs),
        "model_orders": agent.model_orders,
        "fallback_rate": agent.fallbacks / steps,
        "constrained_share": agent.constrained / steps,
        "tactical_share": agent.tactical / steps,
        "legal_share": (
            agent.generated + agent.repaired + agent.constrained
        ) / steps,
        "model_share": agent.model_orders / steps,
        "fallback_reasons": dict(Counter(agent.fallback_reasons)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbs_ncm.play", description=__doc__)
    parser.add_argument("--opponent", choices=("greedy", "llm", "council"), default="greedy")
    parser.add_argument("--ncm-side", choices=("A", "B"), default="B")
    parser.add_argument("--ncm-temperature", type=float, default=0.0)
    parser.add_argument("--ncm-retries", type=int, default=6)
    parser.add_argument("--ncm-retry-temperature", type=float, default=0.5)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--checkpoint-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--fallback", choices=("greedy", "persona"), default="greedy")
    parser.add_argument("--persona", default="", help="persona name for a persona fallback")
    parser.add_argument(
        "--decode",
        choices=("sample", "legal"),
        default="legal",
        help="sample: free generation + repair; legal: rank the view's legal orders",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--shadow", action="store_true", help="let the NCM agent play both sides")
    parser.add_argument("--scenario", help="scenario JSON path (default: bundled skirmish)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--out", default=str(ARTIFACT_DIR / "matches" / "ncm_match.json"))
    parser.add_argument("--orders", default=None)
    parser.add_argument("--llm-log", default=None)
    parser.add_argument("--llm-base-url", default=os.environ.get("TBS_LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--llm-model", default=os.environ.get("TBS_LLM_MODEL"))
    parser.add_argument("--llm-api-key", default=os.environ.get("TBS_LLM_API_KEY"))
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=1500)
    parser.add_argument("--llm-timeout", type=float, default=180.0)
    parser.add_argument(
        "--llm-attempts",
        type=int,
        default=2,
        help="whole-turn attempts after rejections for llm/council opponents",
    )
    parser.add_argument(
        "--council-members",
        help="comma-separated general ids for --opponent council (default: all)",
    )
    parser.add_argument(
        "--council-config",
        help="JSON roster override for --opponent council",
    )
    parser.add_argument(
        "--council-no-commander",
        action="store_true",
        help="skip the supreme commander's pre-brief (one fewer LLM call per turn)",
    )
    parser.add_argument(
        "--council-parallel",
        action="store_true",
        help="ask generals concurrently (the local server may still serialize them)",
    )
    parser.add_argument(
        "--label",
        help="dashboard label for this game (default: derived from opponent and seed)",
    )
    args = parser.parse_args(argv)

    scenario_path = Path(
        args.scenario if args.scenario else SCENARIO_PATH
    ).resolve()
    scenario = load_scenario(scenario_path)
    if args.seed is not None:
        scenario["seed"] = args.seed
    state = new_game(scenario, load_specs())
    rules.start_game(state)

    fallback = (
        PersonaTeacher(args.persona) if args.fallback == "persona" and args.persona else None
    )
    ncm = NcmAgent(
        NcmSampler(args.checkpoint_dir, device=args.device, prefix_len=args.prefix_len),
        temperature=args.ncm_temperature,
        retries=args.ncm_retries,
        retry_temperature=args.ncm_retry_temperature,
        decode=args.decode,
        fallback=fallback,
    )
    logger = JsonlLogger(args.llm_log) if args.llm_log else None

    council_meta = None
    if args.shadow:
        agents = {"A": ncm, "B": ncm}
        opponent_name = "ncm"
    else:
        other = args.ncm_side == "A" and "B" or "A"
        if args.opponent == "greedy":
            opponent = GreedyAgent()
            opponent_name = "greedy"
        else:
            if not args.llm_model:
                parser.error(
                    f'opponent "{args.opponent}" needs --llm-model or TBS_LLM_MODEL'
                )
            config = LlmConfig(
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key=args.llm_api_key,
                temperature=args.llm_temperature,
                max_tokens=args.llm_max_tokens,
                timeout=args.llm_timeout,
                enable_thinking=False,
            )
            if args.opponent == "llm":
                opponent = SingleLlmAgent(LlmClient(config), logger=logger)
                opponent_name = "llm"
            else:
                try:
                    generals = (
                        load_generals(args.council_config)
                        if args.council_config
                        else DEFAULT_GENERALS
                    )
                    if args.council_members:
                        generals = select_generals(
                            generals,
                            [
                                item.strip()
                                for item in args.council_members.split(",")
                                if item.strip()
                            ],
                        )
                except ValueError as exc:
                    parser.error(str(exc))
                opponent = CouncilAgent(
                    LlmClient(config),
                    generals=generals,
                    logger=logger,
                    use_commander=not args.council_no_commander,
                    parallel=args.council_parallel,
                )
                opponent_name = "council"
                council_meta = {
                    "members": [general.id for general in opponent.generals],
                    "use_commander": opponent.commander is not None,
                    "parallel": opponent.parallel,
                }
        agents = {args.ncm_side: ncm, other: opponent}

    print(
        f"scenario {state.scenario_name!r} seed {state.seed}  "
        f"NCM={args.ncm_side} ({args.checkpoint_dir})  opponent={opponent_name}"
    )

    orders: list[dict] = []
    history: list[dict] = []
    included_turns = 0
    generated_before = 0
    fallbacks_before = 0
    while state.status == "ongoing" and state.turn_number <= args.max_turns:
        player = state.current_player
        turn = state.turn_number
        agent = agents[player]
        generated_before = ncm.generated
        repaired_before = ncm.repaired
        fallbacks_before = ncm.fallbacks
        illegal_before = ncm.illegal
        generation_errors_before = ncm.generation_errors
        steps_before = ncm.steps
        rejected_before = ncm.rejected
        state_too_long_before = ncm.state_too_long
        try:
            if agent is ncm:
                result = run_turn_stepwise(
                    state, agent, max_steps=args.max_steps, logger=logger
                )
            else:
                result = run_turn(
                    state, agent, attempts=args.llm_attempts, logger=logger
                )
        except LlmTransportError as exc:
            print(f"\nLLM FAILURE on turn {state.turn_number}, player {player}: {exc}", file=sys.stderr)
            break
        generated = ncm.generated - generated_before
        repaired = ncm.repaired - repaired_before
        fallbacks = ncm.fallbacks - fallbacks_before
        illegal = ncm.illegal - illegal_before
        orders.extend(result.orders)
        included_turns += 1
        history.append(
            {
                "turn": turn,
                "player": player,
                "orders": len(result.orders),
                "attempts": result.attempts,
                "board": {"A": _board_stats(state, "A"), "B": _board_stats(state, "B")},
                "ncm": ({
                    "steps": ncm.steps - steps_before,
                    "generated": generated,
                    "repaired": repaired,
                    "fallbacks": fallbacks,
                    "illegal": illegal,
                    "rejected": ncm.rejected - rejected_before,
                    "state_too_long": ncm.state_too_long - state_too_long_before,
                    "generation_errors": ncm.generation_errors - generation_errors_before,
                } if agent is ncm else None),
                "council": (
                    dict(agent.last_stats)
                    if isinstance(agent, CouncilAgent)
                    else None
                ),
            }
        )
        extra = (
            f"  ncm gen={generated} repair={repaired} illegal={illegal} fallback={fallbacks}"
            if agent is ncm
            else ""
        )
        if isinstance(agent, CouncilAgent):
            stats = agent.last_stats
            extra = (
                f"  council={len(stats.get('generals', []))}"
                f" calls={stats.get('calls', 0)}"
                f" drop={len(stats.get('dropped', []))}"
            )
        rejected = f"  rejected={len(result.errors)}" if result.errors else ""
        print(
            f"turn {state.turn_number:>3} {player}  orders={len(result.orders):>2}  "
            f"calls={result.attempts}{rejected}{extra}  "
            f"units A={len(state.units_of('A'))} B={len(state.units_of('B'))}  "
            f"funds A={state.players['A'].funds} B={state.players['B'].funds}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    orders_path = Path(args.orders) if args.orders else _orders_path(out_path)
    orders_path.parent.mkdir(parents=True, exist_ok=True)
    save_game(state, out_path)
    orders_path.write_text(json.dumps(orders, indent=2) + "\n", encoding="utf-8")
    meta_path = out_path.with_name(out_path.stem + ".meta.json")
    ncm_stats = _ncm_stats(ncm)
    pretty_opponent = {
        "llm": "LLM",
        "greedy": "Greedy",
        "council": "Council",
        "ncm": "NCM",
    }.get(opponent_name, opponent_name)
    meta = {
        "schema": 1,
        "source": "play",
        "label": args.label or f"NCM vs {pretty_opponent} seed {state.seed}",
        "scenario": state.scenario_name,
        "scenario_path": str(scenario_path),
        "seed": state.seed,
        "ncm_side": "both" if args.shadow else args.ncm_side,
        "opponent": opponent_name,
        "council": council_meta,
        "checkpoint": str(Path(args.checkpoint_dir).resolve()),
        "learner": ncm_stats,
        "ncm": ncm_stats,
        "fallback_reasons": dict(Counter(ncm.fallback_reasons)),
        "history": history,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"\nstatus {state.status}" + (f" winner {state.winner}" if state.winner else ""))
    print(
        f"NCM generated {ncm.generated} legal orders, repaired {ncm.repaired}, "
        f"rejected {ncm.illegal} samples, fell back {ncm.fallbacks} times"
    )
    print(f"save   {out_path}")
    print(f"orders {orders_path} ({len(orders)} orders, {included_turns} player-turns)")
    print(f"meta   {meta_path}")
    rebuilt = replay(scenario, orders)
    matched = state_hash(rebuilt) == state_hash(state)
    print(f"replay {'MATCHES' if matched else 'DIFFERS'} state hash {state_hash(state)}")
    if state.status == "ongoing":
        print(f"stopped at the {args.max_turns}-turn cap while still ongoing")
        return 1
    return 0 if matched else 3


if __name__ == "__main__":
    raise SystemExit(main())
