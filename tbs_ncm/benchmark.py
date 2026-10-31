"""Measure NCM order latency on a held-out state file.

The benchmark loads the released tokenizer and generator, runs one order per
held-out state, and reports per-order and per-turn timings.  It writes a JSON
file and prints the same payload, so a blog post can quote the exact run.

    python -m tbs_ncm.benchmark --checkpoint-dir artifacts/tbs `
      --split test_sample.jsonl --decode sample --repeat 3

``--decode legal`` also scores the legal candidate lines for the sampled
action/unit, approximating the UI's constrained decoder.  ``sample`` measures
raw generation only.  Run it on an idle GPU for publishable numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from .common import ARTIFACT_DIR
from .ncm_agent import NcmSampler

__all__ = ["benchmark", "summarize"]


def _rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def summarize(values: list[float]) -> dict[str, float | int]:
    """Mean, median, p95 and max for a list of millisecond timings."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(0.95),
        "min": min(values),
        "max": max(values),
    }


def _run_once(sampler: NcmSampler, row: dict[str, Any], decode: str) -> None:
    if decode == "legal":
        sampler.sample(row["prefix"], temperature=0.0)
        legal = row.get("legal") or []
        if legal:
            sampler.score_lines(row["prefix"], [str(line) for line in legal])
    else:
        sampler.sample(row["prefix"], temperature=0.0)


def benchmark(
    checkpoint_dir: Path | str,
    split: Path | str,
    *,
    device: str | None = None,
    prefix_len: int | None = 2048,
    decode: str = "sample",
    limit: int = 0,
    warmup: int = 5,
    repeat: int = 1,
) -> dict[str, Any]:
    rows = _rows(Path(split), limit)
    if decode == "legal":
        rows = [row for row in rows if isinstance(row.get("legal"), list) and row["legal"]]
        if not rows:
            raise ValueError("legal decoding needs rows with a non-empty 'legal' list")
    sampler = NcmSampler(checkpoint_dir, device=device, prefix_len=prefix_len)
    resolved_device = str(sampler.device)

    for row in rows[: max(0, warmup)]:
        _run_once(sampler, row, decode)

    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings: list[float] = []
    for _ in range(max(1, repeat)):
        for row in rows:
            if resolved_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            _run_once(sampler, row, decode)
            if resolved_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)

    summary = summarize(timings)
    per_order = float(summary.get("mean", 0.0))
    payload: dict[str, Any] = {
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "split": str(Path(split).resolve()),
        "decode": decode,
        "prefix_len": sampler.prefix_len,
        "device": resolved_device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if resolved_device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "gpu_peak_mb": (
            round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
            if resolved_device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "states": len(rows),
        "repeats": max(1, repeat),
        "warmup": max(0, warmup),
        "mean_tokens": round(
            statistics.mean(sampler.token_count(row["prefix"]) for row in rows), 1
        )
        if rows
        else 0,
        "ms_per_order": {key: round(value, 3) if isinstance(value, float) else value for key, value in summary.items()},
        "orders_per_second": round(1000.0 / per_order, 2) if per_order else None,
        "seconds_per_8_orders": round(8.0 * per_order / 1000.0, 3) if per_order else None,
        "seconds_per_64_orders": round(64.0 * per_order / 1000.0, 3) if per_order else None,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--split", required=True, help="JSONL with prefix/target rows")
    parser.add_argument("--device", default="")
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--decode", choices=("sample", "legal"), default="sample")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    payload = benchmark(
        args.checkpoint_dir,
        args.split,
        device=args.device or None,
        prefix_len=args.prefix_len,
        decode=args.decode,
        limit=args.limit,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
