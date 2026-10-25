"""Held-out order quality for every persona checkpoint, in one report.

    python -m tbs_ncm.persona_eval --split test.jsonl --limit 500

Unlike ``evaluate_orders`` this keeps per-action breakdowns, which matters for
personas: a defensive model can have low exact overall while still spelling
moves and holds perfectly and only missing build strings.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from .common import ARTIFACT_DIR, DATA_DIR
from .ncm_agent import NcmSampler
from .personas import persona_names

from tbs.agents.codec import CodecError, decode_order, encode_order

PERSONA_DIR = ARTIFACT_DIR / "personas"
LETTER = {"m": "move", "a": "attack", "b": "build", "c": "capture",
          "h": "hold", "l": "load", "u": "unload"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="score persona checkpoints")
    parser.add_argument("--personas", default=",".join(persona_names()))
    parser.add_argument("--persona-dir", default=str(PERSONA_DIR))
    parser.add_argument("--split", default="test.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument(
        "--generator",
        default="",
        help="generator filename inside each persona dir (default: the config's generator.pt)",
    )
    parser.add_argument("--out", default=str(PERSONA_DIR / "eval"))
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.personas.split(",") if name.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "split": args.split,
        "limit": args.limit,
        "prefix_len": args.prefix_len,
        "generator": args.generator or "generator.pt",
        "personas": {},
    }
    started = time.perf_counter()

    for name in names:
        data_path = DATA_DIR / "personas" / name / args.split
        rows = [
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ][: args.limit]
        if not rows:
            continue
        print(f"\n{name}: {len(rows)} rows from {data_path}")
        sampler = NcmSampler(
            Path(args.persona_dir) / name,
            device=args.device,
            prefix_len=args.prefix_len,
            generator=args.generator or None,
        )
        exact = parseable = legal = checked = state_too_long = 0
        generated_actions: Counter[str] = Counter()
        generated_builds: Counter[str] = Counter()
        action_stats: dict[str, Counter] = defaultdict(Counter)
        for index, row in enumerate(rows, start=1):
            try:
                line = sampler.sample(row["prefix"])
            except ValueError:
                state_too_long += 1
                continue
            target_action = LETTER.get(str(row["target"]).split()[0], "other")
            action_stats[target_action]["n"] += 1
            generated_actions[LETTER.get(line.split()[0], "other")] += 1
            if line.startswith("b ") and len(line.split()) >= 3:
                generated_builds[line.split()[2]] += 1
            is_exact = line == row["target"]
            exact += is_exact
            action_stats[target_action]["exact"] += is_exact
            try:
                order = decode_order(line)
                parseable += 1
                action_stats[target_action]["parseable"] += 1
                allowed = row.get("legal")
                if isinstance(allowed, list):
                    checked += 1
                    legal += encode_order(order) in set(allowed)
            except CodecError:
                pass
            if index % 50 == 0:
                print(f"  {index}/{len(rows)} exact {exact/index:.2%} parseable {parseable/index:.2%}")
        total = max(1, len(rows))
        entry = {
            "rows": len(rows),
            "exact": exact / total,
            "parseable": parseable / total,
            "legal_rate": (legal / checked) if checked else None,
            "legal_n": checked,
            "state_too_long": state_too_long,
            "generated_actions": {key: value / total for key, value in generated_actions.items()},
            "generated_builds": dict(generated_builds),
            "per_target_action": {
                action: {
                    "n": stats["n"],
                    "exact": stats["exact"] / max(1, stats["n"]),
                    "parseable": stats["parseable"] / max(1, stats["n"]),
                }
                for action, stats in sorted(action_stats.items())
            },
        }
        report["personas"][name] = entry
        print(
            f"  exact {entry['exact']:.2%}  parseable {entry['parseable']:.2%}  "
            + (
                f"legal {entry['legal_rate']:.2%}"
                if entry["legal_rate"] is not None
                else "legal n/a"
            )
        )
        del sampler

    report["seconds"] = round(time.perf_counter() - started, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Persona held-out quality",
        "",
        f"Split `{args.split}`, up to {args.limit} rows, prefix {args.prefix_len}, "
        f"generator `{args.generator or 'generator.pt'}`, {report['seconds']}s.",
        "",
        "| persona | exact | parseable | move exact | attack exact | build exact | hold exact |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, entry in report["personas"].items():
        per = entry["per_target_action"]
        lines.append(
            f"| {name} | {entry['exact']:.1%} | {entry['parseable']:.1%} | "
            f"{per.get('move', {}).get('exact', 0):.1%} | "
            f"{per.get('attack', {}).get('exact', 0):.1%} | "
            f"{per.get('build', {}).get('exact', 0):.1%} | "
            f"{per.get('hold', {}).get('exact', 0):.1%} |"
        )
    lines += ["", "## Generated action mix", ""]
    for name, entry in report["personas"].items():
        lines.append(f"- **{name}**: {entry['generated_actions']}  builds {entry['generated_builds']}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nreport: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
