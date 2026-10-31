"""Round-robin matches between the persona NCM checkpoints.

    python -m tbs_ncm.persona_league --games-per-map 1

Every unordered pair plays on every map with sides alternating.  The report
keeps the usual win/score numbers and adds style metrics taken from the agents'
accepted orders: attack/capture/build/hold rates, build mix, average distance to
the enemy and to home, and the fallback rate.  A style distance against the
scripted teacher's training mix shows whether the clone actually learned the
persona instead of only the format.

``--resume`` reuses the save/meta/orders triple for a pair/map/seed that already
finished, so a league interrupted by a reboot continues from the last game.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .arena import load_agent, play_match
from .common import ARTIFACT_DIR, DATA_DIR
from .generate_data import MAX_STEPS_PER_TURN
from .monitor import RunLogger
from .personas import persona_names

import tbs_ncm.common  # noqa: F401

from tbs.engine.data import available_scenarios

PERSONA_DIR = ARTIFACT_DIR / "personas"
LEAGUE_DIR = PERSONA_DIR / "league"
ACTION_KEYS = ("move", "attack", "build", "capture", "hold", "load", "unload")


def _rows_mix(path: Path) -> dict:
    if not path.exists():
        return {}
    actions = Counter()
    builds = Counter()
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        target = str(json.loads(line).get("target", ""))
        parts = target.split()
        if not parts:
            continue
        action = parts[0]
        actions[{"m": "move", "a": "attack", "b": "build", "c": "capture",
                 "h": "hold", "l": "load", "u": "unload"}.get(action, action)] += 1
        if action == "b" and len(parts) >= 3:
            builds[parts[2]] += 1
        total += 1
    return {
        "actions": {key: actions.get(key, 0) / max(1, total) for key in ACTION_KEYS},
        "builds": {key: value / max(1, sum(builds.values())) for key, value in builds.items()},
        "rows": total,
    }


def _style_distance(ncm: dict, teacher: dict) -> float | None:
    if not teacher:
        return None
    ncm_actions = ncm.get("actions", {})
    teacher_actions = teacher.get("actions", {})
    keys = set(ncm_actions) | set(teacher_actions)
    return round(
        0.5 * sum(abs(ncm_actions.get(key, 0) - teacher_actions.get(key, 0)) for key in keys),
        4,
    )


def _load_saved_match(
    save_path: Path,
    *,
    name_a: str,
    name_b: str,
    map_name: str,
    side: str,
) -> dict | None:
    """Rebuild a match from save + sidecars left by an interrupted league."""
    meta_path = save_path.with_name(save_path.stem + ".meta.json")
    orders_path = save_path.with_name(save_path.stem + ".orders.json")
    if not (save_path.is_file() and meta_path.is_file() and orders_path.is_file()):
        return None
    try:
        save = json.loads(save_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(save, dict) or not isinstance(meta, dict):
        return None
    learner_side = str(meta.get("learner_side") or side)
    if learner_side not in ("A", "B"):
        return None
    other = "A" if learner_side == "B" else "B"
    from .gamefeed import save_board_stats

    board = {"A": save_board_stats(save, "A"), "B": save_board_stats(save, "B")}
    winner = save.get("winner") or ""
    status = save.get("status") or "ongoing"
    learner = meta.get("learner") if isinstance(meta.get("learner"), dict) else {}
    opponent = meta.get("opponent") if isinstance(meta.get("opponent"), dict) else {}
    units = save.get("units") if isinstance(save.get("units"), list) else []
    return {
        "seed": meta.get("seed", save.get("seed")),
        "scenario": save.get("scenario"),
        "learner_side": learner_side,
        "opponent_side": other,
        "status": status,
        "winner": winner,
        "learner_won": bool(winner and winner == learner_side),
        "finished": status != "ongoing" and winner in ("A", "B"),
        "turns": save.get("turn_number"),
        "player_turns": save.get("turn_number"),
        "orders": save.get("action_seq"),
        "board": board,
        "learner_board": board[learner_side],
        "opponent_board": board[other],
        "score_delta": round(
            board[learner_side]["score"] - board[other]["score"], 2
        ),
        "units": {
            owner: sum(1 for unit in units if unit.get("owner") == owner)
            for owner in ("A", "B")
        },
        "learner": learner,
        "opponent": opponent,
        "seconds": meta.get("seconds"),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "save": str(save_path),
        "orders_path": str(orders_path),
        "meta_path": str(meta_path),
        "persona_sides": {name_a: learner_side, name_b: other},
        "stats": {learner_side: learner, other: opponent},
        "pair": [name_a, name_b],
        "map": map_name,
    }


def _aggregate(matches: list[dict], name: str) -> dict:
    """Aggregate one persona's side of every match it played."""
    games = wins = losses = unfinished = 0
    score_delta = 0.0
    turns = 0
    actions: Counter[str] = Counter()
    builds: Counter[str] = Counter()
    model_actions: Counter[str] = Counter()
    model_builds: Counter[str] = Counter()
    enemy_distance = home_distance = 0.0
    geometry = 0
    orders = steps = fallbacks = model_orders = generated = repaired = 0
    constrained = 0
    tactical = 0
    rejected = state_too_long = 0
    repairs: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()

    for match in matches:
        if name not in match.get("persona_sides", {}):
            continue
        side = match["persona_sides"][name]
        stats = match["stats"][side]
        games += 1
        if match["winner"] == side:
            wins += 1
        elif match["winner"]:
            losses += 1
        else:
            unfinished += 1
        score_delta += match["score_delta"] if side == match["learner_side"] else -match["score_delta"]
        turns += match["turns"]
        actions.update(stats.get("actions", {}))
        builds.update(stats.get("builds", {}))
        model_actions.update(stats.get("model_actions", {}))
        model_builds.update(stats.get("model_builds", {}))
        orders += stats.get("orders", 0)
        steps += stats.get("steps", stats.get("orders", 0))
        model_orders += stats.get("model_orders", 0)
        generated += stats.get("generated", 0)
        repaired += stats.get("repaired", 0)
        constrained += stats.get("constrained", 0)
        tactical += stats.get("tactical", 0)
        fallbacks += stats.get("fallbacks", 0)
        rejected += stats.get("rejected", 0)
        state_too_long += stats.get("state_too_long", 0)
        repairs.update(stats.get("repairs", {}))
        fallback_reasons.update(stats.get("fallback_reasons", {}))
        if stats.get("mean_enemy_distance") is not None:
            enemy_distance += stats["mean_enemy_distance"]
            home_distance += stats.get("mean_home_distance") or 0.0
            geometry += 1

    total = max(1, steps or orders)
    model_total = max(1, model_orders)
    return {
        "persona": name,
        "games": games,
        "wins": wins,
        "losses": losses,
        "unfinished": unfinished,
        "win_rate": wins / games if games else 0.0,
        "finished_win_rate": wins / (wins + losses) if wins + losses else 0.0,
        "avg_score_delta": score_delta / games if games else 0.0,
        "avg_turns": turns / games if games else 0.0,
        "steps": steps,
        "orders": orders,
        "generated": generated,
        "repaired": repaired,
        "constrained": constrained,
        "tactical": tactical,
        "model_orders": model_orders,
        "fallbacks": fallbacks,
        "rejected": rejected,
        "state_too_long": state_too_long,
        "repairs": dict(repairs),
        "fallback_reasons": dict(fallback_reasons),
        "fallback_rate": fallbacks / total,
        "fallback_rate_per_order": fallbacks / max(1, orders),
        "generated_share": generated / total,
        "repaired_share": repaired / total,
        "constrained_share": constrained / total,
        "tactical_share": tactical / total,
        "legal_share": (generated + repaired + constrained) / total,
        "model_share": model_orders / total,
        "state_too_long_rate": state_too_long / total,
        "actions": {key: actions.get(key, 0) / total for key in ACTION_KEYS},
        "model_actions": {key: model_actions.get(key, 0) / model_total for key in ACTION_KEYS},
        "builds": dict(builds),
        "model_builds": dict(model_builds),
        "build_mix": {
            key: value / max(1, sum(builds.values())) for key, value in builds.items()
        },
        "model_build_mix": {
            key: value / max(1, sum(model_builds.values()))
            for key, value in model_builds.items()
        },
        "mean_enemy_distance": enemy_distance / geometry if geometry else None,
        "mean_home_distance": home_distance / geometry if geometry else None,
        "style_distance": None,
        "model_style_distance": None,
    }



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="persona NCM round-robin")
    parser.add_argument("--personas", default=",".join(persona_names()))
    parser.add_argument("--persona-dir", default=str(PERSONA_DIR))
    parser.add_argument("--maps", default="all", help="all or comma-separated map names")
    parser.add_argument("--games-per-map", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=700000)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_TURN)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--retry-temperature", type=float, default=0.5)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument(
        "--generator",
        default="",
        help="generator filename inside each persona dir (default: generator.pt)",
    )
    parser.add_argument("--fallback", choices=("persona", "greedy"), default="persona")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--decode",
        choices=("sample", "legal"),
        default="legal",
        help="sample: free generation + repair; legal: rank the view's legal orders",
    )
    parser.add_argument("--out-dir", default=str(LEAGUE_DIR))
    parser.add_argument(
        "--match-dir",
        default=str(ARTIFACT_DIR / "matches"),
        help="where saved games and meta sidecars go (the dashboard reads this dir)",
    )
    parser.add_argument(
        "--pairs",
        default="",
        help="optional comma-separated a:b pairs instead of the full round-robin",
    )
    parser.add_argument("--no-save-games", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse saved pair/map/seed games left by an interrupted run",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild the report from saved games without loading checkpoints",
    )
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--no-monitor", action="store_true")
    args = parser.parse_args(argv)

    if args.report_only:
        args.resume = True

    names = [name.strip() for name in args.personas.split(",") if name.strip()]
    persona_dir = Path(args.persona_dir)
    out_dir = Path(args.out_dir)
    matches_dir = Path(args.match_dir)
    matches_dir.mkdir(parents=True, exist_ok=True)

    available = [Path(path) for path in available_scenarios()]
    if args.maps.strip().lower() in ("", "all", "*"):
        maps = available
    else:
        wanted = {token.strip() for token in args.maps.split(",") if token.strip()}
        maps = [path for path in available if path.stem in wanted or path.name in wanted]
    if not maps:
        parser.error("no maps matched")

    agents = {}
    if not args.report_only:
        for name in names:
            checkpoint = persona_dir / name
            if not (checkpoint / "generator.pt").exists():
                parser.error(f"missing persona checkpoint {checkpoint / 'generator.pt'}")
            print(f"loading {name} ...")
            agents[name] = load_agent(
                checkpoint,
                device=args.device,
                temperature=args.temperature,
                retries=args.retries,
                retry_temperature=args.retry_temperature,
                prefix_len=args.prefix_len,
                generator=args.generator or None,
                fallback=args.fallback,
                persona=name,
                decode=args.decode,
            )

    if args.pairs:
        pairs = []
        for token in args.pairs.split(","):
            token = token.strip()
            if not token:
                continue
            left, sep, right = token.partition(":")
            if not sep or left not in names or right not in names:
                parser.error(f"bad pair {token!r}; expected one of {names} on each side")
            pairs.append((left, right))
    else:
        pairs = [
            (names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
    total_games = len(pairs) * len(maps) * args.games_per_map
    run = (
        None
        if args.no_monitor
        else RunLogger(
            "arena",
            runs_dir=args.runs_dir or None,
            argv=argv,
            total_steps=total_games,
            label=f"persona-league-{len(pairs)}pairs",
        )
    )

    matches: list[dict] = []
    started = time.perf_counter()
    for pair_index, (name_a, name_b) in enumerate(pairs):
        for map_index, map_path in enumerate(maps):
            for game in range(args.games_per_map):
                seed = args.seed_base + pair_index * 1000 + map_index * 10 + game
                side = "B" if (game + map_index) % 2 == 0 else "A"
                out = None
                if not args.no_save_games:
                    out = matches_dir / f"persona-{name_a}-vs-{name_b}-{map_path.stem}-seed{seed}.json"
                match = None
                if args.resume and out is not None:
                    match = _load_saved_match(
                        out,
                        name_a=name_a,
                        name_b=name_b,
                        map_name=map_path.stem,
                        side=side,
                    )
                resumed = match is not None
                if match is None and args.report_only:
                    parser.error(f"missing saved game {out}; cannot build the report")
                if match is None:
                    match = play_match(
                        agents[name_a],
                        agents[name_b],
                        seed=seed,
                        learner_side=side,
                        max_turns=args.max_turns,
                        max_steps=args.max_steps,
                        scenario_path=map_path,
                        out=out,
                        verify_replay=not args.no_save_games,
                    )
                    other = "A" if side == "B" else "B"
                    match["persona_sides"] = {name_a: side, name_b: other}
                    match["stats"] = {side: match["learner"], other: match["opponent"]}
                    match["pair"] = [name_a, name_b]
                    match["map"] = map_path.stem
                    if out is not None:
                        meta_path = Path(match["meta_path"])
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        meta.update(
                            {
                                "source": "arena",
                                "label": f"{name_a} vs {name_b} - {map_path.stem}",
                                "learner_persona": name_a,
                                "opponent_persona": name_b,
                                "personas": {side: name_a, other: name_b},
                                "opponent_kind": "ncm",
                                "batch": out_dir.name,
                            }
                        )
                        meta_path.write_text(
                            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                        )
                matches.append(match)
                elapsed = time.perf_counter() - started
                prefix = "resumed" if resumed else f"{match['winner'] or 'unfinished':>10s}"
                print(
                    f"  {name_a} vs {name_b} {map_path.stem} seed {seed}: "
                    f"{prefix}  "
                    f"turns {match['turns']:>2}  score {match['score_delta']:+.1f}  "
                    f"fallback {match['learner'].get('fallback_rate', 0.0):.0%}/"
                    f"{match['opponent'].get('fallback_rate', 0.0):.0%}"
                )
                if run is not None:
                    run.log(
                        step=len(matches),
                        seed=seed,
                        pair=f"{name_a}-{name_b}",
                        map=map_path.stem,
                        winner=match["winner"],
                        learner_side=side,
                        turns=match["turns"],
                        orders=match["orders"],
                        score_delta=match["score_delta"],
                        generated=match["learner"]["generated"],
                        repaired=match["learner"]["repaired"],
                        constrained=match["learner"].get("constrained", 0),
                        tactical=match["learner"].get("tactical", 0),
                        legal_share=match["learner"].get("legal_share", 0.0),
                        fallbacks=match["learner"]["fallbacks"],
                        seconds=round(elapsed, 1),
                    )
                if match.get("replay_matches") is False:
                    print("    WARNING: replay hash mismatch")

    if run is not None:
        wins = sum(match["learner_won"] for match in matches)
        finished = sum(match["finished"] for match in matches)
        run.finish(
            state="done",
            games=len(matches),
            wins=wins,
            finished=finished,
            win_rate=(wins / finished) if finished else 0.0,
            seconds=round(time.perf_counter() - started, 1),
        )

    aggregates = {name: _aggregate(matches, name) for name in names}
    for name, summary in aggregates.items():
        teacher = _rows_mix(DATA_DIR / "personas" / name / "train.jsonl")
        summary["teacher_mix"] = teacher
        summary["style_distance"] = _style_distance(summary, teacher)
        summary["model_style_distance"] = _style_distance(
            {"actions": summary.get("model_actions", {})}, teacher
        )

    reasons_by_map: dict[str, dict[str, Counter[str]]] = {}
    for match in matches:
        map_name = match["map"]
        per_persona = reasons_by_map.setdefault(map_name, {})
        for name, side in match["persona_sides"].items():
            stats = match["stats"][side]
            per_persona.setdefault(name, Counter()).update(
                stats.get("fallback_reasons", {})
            )

    report = {
        "personas": names,
        "maps": [path.stem for path in maps],
        "games_per_map": args.games_per_map,
        "matches": len(matches),
        "fallback": args.fallback,
        "decode": args.decode,
        "prefix_len": args.prefix_len,
        "generator": args.generator or "generator.pt",
        "retries": args.retries,
        "seconds": round(time.perf_counter() - started, 1),
        "aggregates": aggregates,
        "fallback_reasons_by_map": {
            map_name: {
                name: dict(counts) for name, counts in per_persona.items()
            }
            for map_name, per_persona in reasons_by_map.items()
        },
        "games": [
            {
                "pair": match["pair"],
                "map": match["map"],
                "seed": match["seed"],
                "winner": match["winner"],
                "persona_sides": match["persona_sides"],
                "turns": match["turns"],
                "score_delta": match["score_delta"],
                "status": match["status"],
                "learner_side": match["learner_side"],
                "stats": match["stats"],
            }
            for match in matches
        ],
    }

    pairwise: dict[str, dict[str, dict]] = {
        name: {other: {"games": 0, "wins": 0, "losses": 0, "unfinished": 0}
               for other in names if other != name}
        for name in names
    }
    for game in report["games"]:
        name_a, name_b = game["pair"]
        sides = game["persona_sides"]
        winner = game["winner"]
        entry_ab = pairwise[name_a][name_b]
        entry_ba = pairwise[name_b][name_a]
        entry_ab["games"] += 1
        entry_ba["games"] += 1
        if winner:
            winner_name = name_a if sides[name_a] == winner else name_b
            if winner_name == name_a:
                entry_ab["wins"] += 1
                entry_ba["losses"] += 1
            else:
                entry_ab["losses"] += 1
                entry_ba["wins"] += 1
        else:
            entry_ab["unfinished"] += 1
            entry_ba["unfinished"] += 1
    report["pairwise"] = pairwise
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Persona league",
        "",
        f"{len(matches)} games on {len(maps)} maps, fallback `{args.fallback}`, "
        f"decode `{args.decode}`, generator `{args.generator or 'generator.pt'}`, "
        f"prefix `{args.prefix_len}`, retries `{args.retries}`.",
        "",
        "`exact` = raw generated legal; `constrained` = model-ranked legal; "
        "`legal` = exact+constrained+repaired (model-controlled); "
        "`model` = all model orders; `fallback` = scripted teacher.",
        "",
        "| persona | W-L-U | win% | score | attack% | capture% | build% | hold% | exact% | constrained% | tactical% | legal% | model% | fallback% | context% | enemy dist | style dist | model style |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        row = aggregates[name]
        lines.append(
            f"| {name} | {row['wins']}-{row['losses']}-{row['unfinished']} | "
            f"{row['win_rate']:.0%} | {row['avg_score_delta']:+.1f} | "
            f"{row['actions'].get('attack', 0):.0%} | {row['actions'].get('capture', 0):.0%} | "
            f"{row['actions'].get('build', 0):.0%} | {row['actions'].get('hold', 0):.0%} | "
            f"{row['generated_share']:.0%} | {row['constrained_share']:.0%} | "
            f"{row['tactical_share']:.0%} | {row['legal_share']:.0%} | "
            f"{row['model_share']:.0%} | {row['fallback_rate']:.0%} | "
            f"{row['state_too_long_rate']:.0%} | "
            f"{row['mean_enemy_distance'] if row['mean_enemy_distance'] is not None else 0:.1f} | "
            f"{row['style_distance'] if row['style_distance'] is not None else 0:.2f} | "
            f"{row['model_style_distance'] if row['model_style_distance'] is not None else 0:.2f} |"
        )
    lines += ["", "## Fallback reasons", ""]
    for name in names:
        row = aggregates[name]
        total = max(1, row["fallbacks"])
        reasons = row["fallback_reasons"]
        if reasons:
            rendered = ", ".join(
                f"`{reason}` {count} ({count / total:.0%})"
                for reason, count in sorted(reasons.items(), key=lambda item: -item[1])
            )
        else:
            rendered = "none"
        lines.append(f"- **{name}**: {rendered}")
        if row.get("repairs"):
            mix = ", ".join(
                f"{kind} {count}"
                for kind, count in sorted(
                    row["repairs"].items(), key=lambda item: -item[1]
                )
            )
            lines.append(f"  repairs: {mix}")
    if reasons_by_map:
        lines += ["", "## Fallback reasons by map", ""]
        for map_name, per_persona in sorted(reasons_by_map.items()):
            rendered = "; ".join(
                f"{name}: " + (", ".join(f"{key} {value}" for key, value in sorted(counts.items())) or "none")
                for name, counts in sorted(per_persona.items())
            )
            lines.append(f"- **{map_name}**: {rendered}")
    lines += ["", "## Build mix", ""]
    for name in names:
        lines.append(f"- **{name}**: {aggregates[name]['builds']}")
        lines.append(f"  model-only: {aggregates[name]['model_builds']}")
    lines += ["", "## Pairwise wins (row vs column)", ""]
    lines.append("| | " + " | ".join(names) + " |")
    lines.append("|---|" + "|".join("---" for _ in names) + "|")
    for name in names:
        cells = []
        for other in names:
            if name == other:
                cells.append("-")
                continue
            entry = pairwise[name][other]
            cell = f"{entry['wins']}-{entry['losses']}"
            if entry["unfinished"]:
                cell += f" ({entry['unfinished']}U)"
            cells.append(cell)
        lines.append(f"| **{name}** | " + " | ".join(cells) + " |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines[:10 + len(names)]))
    print(f"\nreport: {out_dir / 'report.json'}\n        {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
