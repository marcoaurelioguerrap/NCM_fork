"""League of NCM personas against randomized conventional AI opponents.

    python -m tbs_ncm.mixed_league --heuristics 5 --max-turns 40

Each NCM persona plays each seeded conventional AI (a randomized scripted
policy from :mod:`tbs_ncm.heuristic`).  Games are saved in the same form as the
persona league, so the dashboard can replay them; the report compares NCM
quality and heuristic parameters.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .arena import load_agent, play_match
from .common import ARTIFACT_DIR
from .generate_data import MAX_STEPS_PER_TURN
from .heuristic import HeuristicAgent
from .persona_league import _load_saved_match
from .personas import persona_names

import tbs_ncm.common  # noqa: F401

from tbs.engine.data import available_scenarios

PERSONA_DIR = ARTIFACT_DIR / "personas"
ACTION_KEYS = ("move", "attack", "build", "capture", "hold", "load", "unload")


def _resolve_maps(value: str) -> list[Path]:
    available = [Path(path) for path in available_scenarios()]
    if value.strip().lower() in ("", "all", "*"):
        return available
    wanted = {token.strip() for token in value.split(",") if token.strip()}
    return [path for path in available if path.stem in wanted or path.name in wanted]


def _aggregate(
    matches: list[dict], name: str, side_key: str, field: str
) -> dict:
    games = wins = losses = unfinished = 0
    score = 0.0
    actions: Counter[str] = Counter()
    stats_totals: Counter[str] = Counter()
    for match in matches:
        if match.get(field) != name:
            continue
        stats = match[side_key]
        games += 1
        winner = match.get("winner") or ""
        learner_side = match.get("learner_side")
        if side_key == "learner":
            if winner == learner_side:
                wins += 1
            elif winner:
                losses += 1
            else:
                unfinished += 1
        else:
            if winner and winner != learner_side:
                wins += 1
            elif winner:
                losses += 1
            else:
                unfinished += 1
        delta = float(match.get("score_delta") or 0.0)
        score += delta if side_key == "learner" else -delta
        actions.update(stats.get("actions", {}))
        for key in (
            "steps",
            "orders",
            "generated",
            "constrained",
            "tactical",
            "repaired",
            "fallbacks",
            "model_orders",
            "illegal",
        ):
            stats_totals[key] += int(stats.get(key, 0) or 0)
    steps = max(1, stats_totals["steps"] or stats_totals["orders"])
    return {
        "name": name,
        "games": games,
        "wins": wins,
        "losses": losses,
        "unfinished": unfinished,
        "win_rate": wins / games if games else 0.0,
        "avg_score_delta": score / games if games else 0.0,
        "actions": {key: actions.get(key, 0) / steps for key in ACTION_KEYS},
        "generated_share": stats_totals["generated"] / steps,
        "constrained_share": stats_totals["constrained"] / steps,
        "tactical_share": stats_totals["tactical"] / steps,
        "repaired_share": stats_totals["repaired"] / steps,
        "fallback_rate": stats_totals["fallbacks"] / steps,
        "legal_share": (
            stats_totals["generated"]
            + stats_totals["repaired"]
            + stats_totals["constrained"]
        )
        / steps,
        "orders": stats_totals["orders"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NCM vs randomized conventional AI league")
    parser.add_argument("--ncm-personas", default=",".join(persona_names()))
    parser.add_argument("--heuristics", type=int, default=5)
    parser.add_argument("--heuristic-seed-base", type=int, default=900000)
    parser.add_argument("--persona-dir", default=str(PERSONA_DIR))
    parser.add_argument("--maps", default="border_skirmish")
    parser.add_argument("--games-per-map", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=770000)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_TURN)
    parser.add_argument("--generator", default="mixed_generator.pt")
    parser.add_argument(
        "--decode",
        choices=("sample", "legal"),
        default="legal",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(ARTIFACT_DIR / "personas" / "mixed_league"))
    parser.add_argument("--match-dir", default=str(ARTIFACT_DIR / "matches"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.ncm_personas.split(",") if name.strip()]
    maps = _resolve_maps(args.maps)
    if not names:
        parser.error("no NCM personas selected")
    if not maps:
        parser.error("no maps matched")
    if args.heuristics < 1:
        parser.error("--heuristics must be at least 1")
    if args.report_only:
        args.resume = True

    persona_dir = Path(args.persona_dir)
    out_dir = Path(args.out_dir)
    matches_dir = Path(args.match_dir)
    matches_dir.mkdir(parents=True, exist_ok=True)

    heuristics = [
        HeuristicAgent(args.heuristic_seed_base + index)
        for index in range(args.heuristics)
    ]
    agents = {}
    if not args.report_only:
        for name in names:
            checkpoint = persona_dir / name
            generator = checkpoint / args.generator
            if not generator.exists():
                parser.error(f"missing {generator}")
            print(f"loading {name} ...", flush=True)
            agents[name] = load_agent(
                checkpoint,
                device=args.device,
                generator=args.generator or None,
                fallback="persona",
                persona=name,
                prefix_len=2048,
                decode=args.decode,
            )

    matches: list[dict] = []
    started = time.perf_counter()
    for p_index, name in enumerate(names):
        for h_index, heuristic in enumerate(heuristics):
            for m_index, map_path in enumerate(maps):
                for game in range(args.games_per_map):
                    seed = (
                        args.seed_base
                        + p_index * 10000
                        + h_index * 100
                        + m_index * 10
                        + game
                    )
                    side = "B" if (h_index + m_index + game) % 2 == 0 else "A"
                    other = "A" if side == "B" else "B"
                    out = matches_dir / (
                        f"mixed-{name}-vs-{heuristic.name}-{map_path.stem}-seed{seed}.json"
                    )
                    match = None
                    if args.resume:
                        match = _load_saved_match(
                            out,
                            name_a=name,
                            name_b=heuristic.name,
                            map_name=map_path.stem,
                            side=side,
                        )
                    if match is None:
                        if args.report_only:
                            continue
                        match = play_match(
                            agents[name],
                            heuristic,
                            seed=seed,
                            learner_side=side,
                            max_turns=args.max_turns,
                            max_steps=args.max_steps,
                            scenario_path=map_path,
                            out=out,
                            verify_replay=True,
                        )
                        meta_path = Path(match["meta_path"])
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        meta.update(
                            {
                                "source": "arena",
                                "label": f"{name} vs {heuristic.name} - {map_path.stem}",
                                "learner_persona": name,
                                "opponent_persona": heuristic.name,
                                "opponent_kind": "heuristic",
                                "heuristic_seed": heuristic.seed,
                                "heuristic_params": heuristic.params,
                                "personas": {side: name, other: heuristic.name},
                                "batch": out_dir.name,
                            }
                        )
                        meta_path.write_text(
                            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                        )
                        print(
                            f"  {name} vs {heuristic.name} {map_path.stem} seed {seed}: "
                            f"{match['winner'] or 'unfinished'} turns {match['turns']} "
                            f"score {match['score_delta']:+.1f}",
                            flush=True,
                        )
                    match["ncm_persona"] = name
                    match["heuristic_name"] = heuristic.name
                    match["heuristic_params"] = heuristic.params
                    match["map"] = map_path.stem
                    matches.append(match)

    report = {
        "ncm_personas": names,
        "heuristics": [
            {"name": item.name, "seed": item.seed, "params": item.params}
            for item in heuristics
        ],
        "maps": [path.stem for path in maps],
        "games": len(matches),
        "generator": args.generator,
        "decode": args.decode,
        "max_turns": args.max_turns,
        "seconds": round(time.perf_counter() - started, 1),
        "ncm": {
            name: _aggregate(matches, name, "learner", "ncm_persona")
            for name in names
        },
        "heuristic": {
            item.name: _aggregate(
                matches, item.name, "opponent", "heuristic_name"
            )
            for item in heuristics
        },
    }

    pairwise = {
        name: {
            item.name: {"games": 0, "wins": 0, "losses": 0, "unfinished": 0}
            for item in heuristics
        }
        for name in names
    }
    for match in matches:
        name = match["ncm_persona"]
        heuristic = match["heuristic_name"]
        entry = pairwise[name][heuristic]
        entry["games"] += 1
        winner = match.get("winner") or ""
        if winner == match.get("learner_side"):
            entry["wins"] += 1
        elif winner:
            entry["losses"] += 1
        else:
            entry["unfinished"] += 1
    report["pairwise"] = pairwise

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# NCM vs randomized conventional AI",
        "",
        f"{len(matches)} games, generator `{args.generator}`, decode `{args.decode}`, "
        f"max turns `{args.max_turns}`.",
        "",
        "## NCM personas",
        "",
        "| persona | W-L-U | win% | score | attack% | capture% | hold% | legal% | fallback% | tactical% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        row = report["ncm"][name]
        lines.append(
            f"| {name} | {row['wins']}-{row['losses']}-{row['unfinished']} | "
            f"{row['win_rate']:.0%} | {row['avg_score_delta']:+.1f} | "
            f"{row['actions'].get('attack', 0):.0%} | "
            f"{row['actions'].get('capture', 0):.0%} | "
            f"{row['actions'].get('hold', 0):.0%} | "
            f"{row['legal_share']:.0%} | {row['fallback_rate']:.0%} | "
            f"{row['tactical_share']:.0%} |"
        )
    lines += [
        "",
        "## Conventional AIs",
        "",
        "| AI | seed | W-L-U | win% | score | attack% | capture% | hold% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in heuristics:
        row = report["heuristic"][item.name]
        lines.append(
            f"| {item.name} | {item.seed} | "
            f"{row['wins']}-{row['losses']}-{row['unfinished']} | "
            f"{row['win_rate']:.0%} | {row['avg_score_delta']:+.1f} | "
            f"{row['actions'].get('attack', 0):.0%} | "
            f"{row['actions'].get('capture', 0):.0%} | "
            f"{row['actions'].get('hold', 0):.0%} |"
        )
    lines += [
        "",
        "## Pairwise (NCM row vs conventional AI column, W-L-U)",
        "",
        "| NCM | " + " | ".join(item.name for item in heuristics) + " |",
        "|---|" + "|".join("---" for _ in heuristics) + "|",
    ]
    for name in names:
        cells = []
        for item in heuristics:
            entry = pairwise[name][item.name]
            cell = f"{entry['wins']}-{entry['losses']}"
            if entry["unfinished"]:
                cell += f" ({entry['unfinished']}U)"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nreport: {out_dir / 'report.json'}\n        {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
