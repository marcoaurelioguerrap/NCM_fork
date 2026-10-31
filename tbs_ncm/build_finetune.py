"""Build a reproducible mixed dataset for a persona generator fine-tune.

The original persona data is large and mostly easy orders; the DAgger rows are
small but cover the states the NCM actually reaches.  This tool samples the
original split (optionally oversampling rare actions), adds the on-policy rows
with a configurable weight, and writes one ``finetune_train.jsonl`` per persona.

    python -m tbs_ncm.build_finetune --personas aggressive,armor
    python -m tbs_ncm.build_finetune --original 2000 --dagger-weight 2 \
        --fallback-weight 3 --repair-weight 1.5

Duplicate rows are intentional: ``train_generator.py`` consumes a plain JSONL
list, so repeating a row is how this tool applies a sample weight.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from .common import DATA_DIR
from .personas import persona_names

__all__ = ["build_persona"]


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "prefix" in row and "target" in row:
            rows.append(row)
    return rows


def _action(row: dict) -> str:
    return str(row.get("target", "")).split(" ", 1)[0]


def _sample_original(
    rows: list[dict],
    count: int,
    rng: random.Random,
    balance: bool,
    hold_ratio: float = 1.0,
) -> list[dict]:
    if count <= 0 or not rows:
        return []
    if hold_ratio < 1.0:
        holds = [row for row in rows if _action(row) == "h"]
        others = [row for row in rows if _action(row) != "h"]
        keep = int(len(holds) * max(0.0, hold_ratio))
        if keep:
            holds = rng.sample(holds, min(keep, len(holds)))
        else:
            holds = []
        rows = others + holds
    if count >= len(rows):
        return list(rows)
    if not balance:
        return rng.sample(rows, count)

    # Inverse-square-root class weights: rare actions (capture/load/unload)
    # get more draws without letting a handful of rows dominate.
    frequencies = Counter(_action(row) for row in rows)
    weights = [1.0 / math.sqrt(frequencies[_action(row)]) for row in rows]
    return rng.choices(rows, weights=weights, k=count)


def _dagger_copies(row: dict, base: int, fallback: float, repair: float) -> int:
    provenance = str(row.get("teacher_for") or row.get("kind") or "")
    factor = 1.0
    if provenance == "fallback":
        factor = fallback
    elif provenance == "repaired":
        factor = repair
    return max(1, int(round(base * factor)))


def build_persona(
    name: str,
    *,
    data_dir: Path,
    dagger_dir: Path | None,
    dagger_name: str,
    out_dir: Path,
    original: int,
    dagger_weight: int,
    fallback_weight: float,
    repair_weight: float,
    seed: int,
    balance: bool,
    hold_ratio: float,
) -> dict:
    """Write one persona's mixed fine-tune split and return its summary."""
    rng = random.Random(f"{seed}:{name}")
    source = data_dir / name
    original_rows = _read_rows(source / "train.jsonl")
    dagger_source = (dagger_dir or data_dir) / name
    dagger_rows = _read_rows(dagger_source / dagger_name)
    if hold_ratio < 1.0:
        dagger_rows = [
            row
            for row in dagger_rows
            if _action(row) != "h" or rng.random() < hold_ratio
        ]
    sampled = _sample_original(
        original_rows, original, rng, balance, hold_ratio
    )

    dagger: list[tuple[dict, str]] = []
    for row in dagger_rows:
        copies = _dagger_copies(
            row, max(1, dagger_weight), fallback_weight, repair_weight
        )
        dagger.extend((dict(row), "dagger") for _ in range(copies))

    combined: list[tuple[dict, str]] = [(row, "original") for row in sampled] + dagger
    rng.shuffle(combined)

    target_dir = out_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "finetune_train.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row, _source in combined:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    summary = {
        "persona": name,
        "seed": seed,
        "balance": balance,
        "hold_ratio": hold_ratio,
        "original_available": len(original_rows),
        "original_sampled": len(sampled),
        "dagger_available": len(dagger_rows),
        "dagger_expanded": len(dagger),
        "rows": len(combined),
        "actions": dict(Counter(_action(row) for row, _source in combined)),
        "sources": dict(Counter(source for _row, source in combined)),
        "teacher_for": dict(
            Counter(
                str(
                    row.get("teacher_for")
                    or (row.get("kind") if source == "dagger" else "original")
                    or source
                )
                for row, source in combined
            )
        ),
        "path": str(path),
    }
    (target_dir / "finetune.summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build persona fine-tune datasets")
    parser.add_argument("--personas", default=",".join(persona_names()))
    parser.add_argument("--data-dir", default=str(DATA_DIR / "personas"))
    parser.add_argument(
        "--dagger-dir",
        default="",
        help="directory holding <persona>/dagger.jsonl (default: --data-dir)",
    )
    parser.add_argument(
        "--dagger-name",
        default="dagger.jsonl",
        help="filename to read under --dagger-dir, e.g. train.jsonl for push data",
    )
    parser.add_argument("--out-dir", default=str(DATA_DIR / "personas"))
    parser.add_argument(
        "--original",
        type=int,
        default=1500,
        help="original rows sampled per persona (0 = no original rows)",
    )
    parser.add_argument("--dagger-weight", type=int, default=2)
    parser.add_argument("--fallback-weight", type=float, default=2.0)
    parser.add_argument("--repair-weight", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="oversample rare actions in the original split",
    )
    parser.add_argument(
        "--hold-ratio",
        type=float,
        default=1.0,
        help="keep this fraction of hold rows (0.1 = push-heavy dataset)",
    )
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.personas.split(",") if name.strip()]
    data_dir = Path(args.data_dir)
    dagger_dir = Path(args.dagger_dir) if args.dagger_dir else None
    out_dir = Path(args.out_dir)
    if not names:
        parser.error("no personas selected")

    summaries = []
    for name in names:
        if not (data_dir / name / "train.jsonl").exists():
            print(f"  {name}: missing {data_dir / name / 'train.jsonl'}, skipped")
            continue
        summary = build_persona(
            name,
            data_dir=data_dir,
            dagger_dir=dagger_dir,
            dagger_name=args.dagger_name,
            out_dir=out_dir,
            original=args.original,
            dagger_weight=args.dagger_weight,
            fallback_weight=args.fallback_weight,
            repair_weight=args.repair_weight,
            seed=args.seed,
            balance=args.balance,
            hold_ratio=args.hold_ratio,
        )
        summaries.append(summary)
        print(
            f"  {name}: {summary['rows']} rows "
            f"({summary['original_sampled']} original + "
            f"{summary['dagger_expanded']} dagger) -> {summary['path']}"
        )
        print(
            f"    actions {summary['actions']}  "
            f"teacher_for {summary['teacher_for']}"
        )

    if not summaries:
        print("no datasets built")
        return 1
    index = {
        "seed": args.seed,
        "personas": summaries,
        "original": args.original,
        "data_dir": str(data_dir),
        "dagger_dir": str(dagger_dir or data_dir),
        "dagger_name": args.dagger_name,
        "dagger_weight": args.dagger_weight,
        "fallback_weight": args.fallback_weight,
        "repair_weight": args.repair_weight,
        "balance": args.balance,
        "hold_ratio": args.hold_ratio,
    }
    (out_dir / "finetune.summary.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
