"""Score the trained NCM player on held-out stepwise states.

    python -m tbs_ncm.evaluate_orders --limit 500 --split test.jsonl

The score is written to ``artifacts/tbs/runs/`` so the dashboard can compare
checkpoints.  ``legal_rate`` needs ``--record-legal`` rows (see
``generate_data.py``); it is skipped when the split stores no legal sets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from .common import ARTIFACT_DIR, DATA_DIR
from .monitor import RunLogger
from .ncm_agent import NcmSampler

from tbs.agents.codec import CodecError, decode_order, encode_order


def _split_path(name: str) -> Path:
    """A split name under ``tbs_ncm/data``, or an explicit file path."""
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    return DATA_DIR / name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default=str(ARTIFACT_DIR))
    parser.add_argument(
        "--generator",
        default="",
        help="generator filename or absolute path (default: the checkpoint config)",
    )
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--no-monitor", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in _split_path(args.split).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    sampler = NcmSampler(
        args.checkpoint_dir,
        device=args.device,
        prefix_len=args.prefix_len,
        generator=args.generator or None,
    )

    exact = 0
    parseable = 0
    legal = 0
    legal_checked = 0
    state_too_long = 0
    started = time.perf_counter()
    shown = 0
    samples: list[dict] = []

    run = None
    if not args.no_monitor:
        run = RunLogger(
            "eval",
            runs_dir=args.runs_dir or None,
            argv=sys.argv,
            total_steps=len(rows),
            label=Path(args.split).stem,
        )

    try:
        with (run if run is not None else nullcontext()):
            for index, row in enumerate(rows, start=1):
                try:
                    line = sampler.sample(row["prefix"])
                except ValueError:
                    state_too_long += 1
                    continue
                is_exact = line == row["target"]
                exact += is_exact
                order = None
                error = None
                try:
                    order = decode_order(line)
                    parseable += 1
                except CodecError as exc:
                    error = str(exc)
                    if shown < args.show:
                        print(f"UNPARSEABLE {line!r} (target {row['target']!r})")
                        shown += 1

                allowed = row.get("legal")
                if isinstance(allowed, list):
                    legal_checked += 1
                    if order is not None:
                        try:
                            legal += encode_order(order) in set(allowed)
                        except CodecError:
                            pass

                if len(samples) < 80:
                    samples.append(
                        {
                            "target": row["target"],
                            "generated": line,
                            "exact": bool(is_exact),
                            "parseable": order is not None,
                            "error": error,
                        }
                    )
                if shown < args.show and not is_exact:
                    print(f"  target {row['target']!r:26s} -> {line!r}")
                    shown += 1

                if run is not None and index % max(1, args.log_every) == 0:
                    elapsed = max(1e-6, time.perf_counter() - started)
                    run.log(
                        step=index,
                        exact=exact / index,
                        parseable=parseable / index,
                        legal_rate=(legal / legal_checked) if legal_checked else None,
                        ms_per_order=round(elapsed / index * 1000, 2),
                    )

            elapsed = time.perf_counter() - started
            total = max(1, len(rows))
            ms_per_order = round(elapsed / total * 1000, 2)
            print(
                f"{len(rows)} held-out states: exact {exact}/{len(rows)} ({exact / total:.2%}), "
                f"parseable {parseable}/{len(rows)} ({parseable / total:.2%}), "
                + (
                    f"legal {legal}/{legal_checked} ({legal / max(1, legal_checked):.2%}), "
                    if legal_checked
                    else ""
                )
                + (
                    f"state-too-long {state_too_long}, "
                    if state_too_long
                    else ""
                )
                + f"{ms_per_order:.1f} ms/order"
            )
            if run is not None:
                wrong = [sample for sample in samples if not sample["exact"]]
                run.log_samples(wrong[:8] + samples[:4], limit=12)
                run.finish(
                    state="done",
                    n=len(rows),
                    exact=exact / total,
                    parseable=parseable / total,
                    legal_rate=(legal / legal_checked) if legal_checked else None,
                    legal_n=legal_checked,
                    ms_per_order=ms_per_order,
                )
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

