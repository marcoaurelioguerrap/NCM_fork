"""Collect on-policy DAgger rows for the persona checkpoints.

This is the training counterpart to ``persona_league``.  Each persona plays a
few games against the next persona; a scripted :class:`PersonaTeacher` labels
every state the NCM actually reached.  Standard DAgger uses those on-policy
states instead of only the offline teacher trajectories.

    python -m tbs_ncm.collect_dagger --games-per-persona 2 --max-turns 18
    python -m tbs_ncm.collect_dagger --games-per-persona 6 --maps all \
        --opponent-mode rotate --resume

``--opponent-mode rotate`` cycles through every persona, ``--maps`` selects the
maps to rotate, and ``--resume`` skips personas whose ``dagger.jsonl`` already
exists.  Outputs ``tbs_ncm/data/personas/<name>/dagger.jsonl`` plus a combined
``tbs_ncm/data/personas/dagger.jsonl``.  Only teacher-labelled rows are kept;
the NCM's own sampled orders are not training targets here.  Rows whose state
came from a fallback are marked ``teacher_for="fallback"`` so a fine-tune can
weight them more heavily.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .arena import load_agent
from .common import ARTIFACT_DIR, DATA_DIR
from .personas import PersonaTeacher, persona_names
from .selfplay import collect_game, dedupe, write_rows

import tbs_ncm.common  # noqa: F401

from tbs.engine.data import available_scenarios

PERSONA_DIR = ARTIFACT_DIR / "personas"


def _resolve_maps(value: str) -> list[Path]:
    available = [Path(path) for path in available_scenarios()]
    if value.strip().lower() in ("", "all", "*"):
        return available
    wanted = {token.strip() for token in value.split(",") if token.strip()}
    return [path for path in available if path.stem in wanted or path.name in wanted]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personas", default=",".join(persona_names()))
    parser.add_argument("--checkpoint-dir", default=str(PERSONA_DIR))
    parser.add_argument("--games-per-persona", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=70000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--retry-temperature", type=float, default=0.5)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--max-turns", type=int, default=18)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(DATA_DIR / "personas"))
    parser.add_argument(
        "--maps", default="all", help="all or comma-separated map names"
    )
    parser.add_argument(
        "--opponent-mode",
        choices=("next", "rotate"),
        default="next",
        help="next: always the next persona; rotate: cycle through all personas",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip personas whose dagger.jsonl already exists",
    )
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.personas.split(",") if name.strip()]
    if len(names) < 1:
        parser.error("need at least one persona")
    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    maps = _resolve_maps(args.maps)
    if not maps:
        parser.error("no maps matched")

    agents = {}
    for name in names:
        print(f"loading {name} ...", flush=True)
        agents[name] = load_agent(
            checkpoint_dir / name,
            device=args.device,
            temperature=args.temperature,
            retries=args.retries,
            retry_temperature=args.retry_temperature,
            prefix_len=args.prefix_len,
            fallback="persona",
            persona=name,
        )

    all_rows: list[dict] = []
    started = time.perf_counter()
    for persona_index, name in enumerate(names):
        learner = agents[name]
        path = out_dir / name / "dagger.jsonl"
        if args.resume and path.is_file():
            existing = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            all_rows.extend(existing)
            print(f"  {name}: resumed {len(existing)} rows from {path}", flush=True)
            continue
        persona_rows: list[dict] = []
        games: list[dict] = []
        for game in range(args.games_per_persona):
            if args.opponent_mode == "rotate":
                opponent = agents[names[(persona_index + game + 1) % len(names)]]
            else:
                opponent = agents[names[(persona_index + 1) % len(names)]]
            map_path = maps[(persona_index + game) % len(maps)]
            seed = args.seed_base + persona_index * 100 + game
            side = "B" if game % 2 == 0 else "A"
            teacher = PersonaTeacher(name, seed=seed)
            rows, stats = collect_game(
                learner,
                opponent,
                seed=seed,
                learner_side=side,
                teacher=teacher,
                teacher_mode="all",
                max_turns=args.max_turns,
                max_steps=args.max_steps,
                scenario_path=map_path,
            )
            for row in rows:
                if not str(row.get("kind", "")).startswith("teacher"):
                    continue
                row["persona"] = name
                row["seed"] = seed
                persona_rows.append(row)
            games.append(stats)
            print(
                f"  {name} game {game + 1}/{args.games_per_persona} "
                f"[{map_path.stem}]: "
                f"{stats['examples']} rows, {stats['turns']} turns, "
                f"{stats['status']} {stats['winner'] or ''}",
                flush=True,
            )

        persona_rows = dedupe(persona_rows)
        path = write_rows(out_dir / name / "dagger.jsonl", persona_rows)
        fallback_rows = sum(
            1 for row in persona_rows if row.get("teacher_for") == "fallback"
        )
        summary = {
            "persona": name,
            "games": games,
            "maps": [item.stem for item in maps],
            "opponent_mode": args.opponent_mode,
            "rows": len(persona_rows),
            "fallback_teacher_rows": fallback_rows,
            "kinds": dict(Counter(row["kind"] for row in persona_rows)),
            "seconds": round(time.perf_counter() - started, 1),
        }
        (path.parent / "dagger.summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{name}: {len(persona_rows)} teacher rows "
            f"({fallback_rows} from fallback states) -> {path}",
            flush=True,
        )
        all_rows.extend(persona_rows)

    combined = write_rows(out_dir / "dagger.jsonl", all_rows)
    print(
        f"combined: {len(all_rows)} teacher rows in {time.perf_counter() - started:.0f}s "
        f"-> {combined}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
