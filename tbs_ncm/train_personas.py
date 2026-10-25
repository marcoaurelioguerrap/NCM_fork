"""Fine-tune one NCM checkpoint per persona, with bounded parallelism.

    python -m tbs_ncm.train_personas --concurrency 2

Each job starts from the released ``artifacts/tbs/generator.pt`` and writes
``artifacts/tbs/personas/<name>/{generator.pt, tokenizer.pt}``.  Logs go to
``artifacts/tbs/personas/logs/<name>.log``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .common import ARTIFACT_DIR, CONFIG_PATH, DATA_DIR
from .personas import persona_names

PERSONA_DIR = ARTIFACT_DIR / "personas"
BASE_CHECKPOINT = ARTIFACT_DIR / "generator.pt"
BASE_TOKENIZER = ARTIFACT_DIR / "tokenizer.pt"


def _command(name: str, args: argparse.Namespace) -> list[str]:
    checkpoint = f"personas/{name}/generator.pt"
    command = [
        sys.executable,
        "-u",
        "-m",
        "tbs_ncm.train_generator",
        "--train",
        str(DATA_DIR / "personas" / name / "train.jsonl"),
        "--validation",
        str(DATA_DIR / "personas" / name / "validation.jsonl"),
        "--out",
        checkpoint,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--min-lr",
        str(args.min_lr),
        "--warmup",
        str(args.warmup),
        "--eval-limit",
        str(args.eval_limit),
        "--eval-generate",
        str(args.eval_generate),
        "--log-every",
        str(args.log_every),
        "--seed",
        str(args.seed),
        "--save-every",
        "0",
    ]
    if args.resume:
        command.extend(["--resume", checkpoint])
    else:
        command.extend(["--init", str(BASE_CHECKPOINT)])
    if args.config:
        command.extend(["--config", str(args.config)])
    if args.device:
        command.extend(["--device", args.device])
    if args.max_steps:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.no_monitor:
        command.append("--no-monitor")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fine-tune the persona NCM checkpoints")
    parser.add_argument("--personas", default=",".join(persona_names()))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--eval-generate", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="retrain checkpoints that already exist")
    parser.add_argument("--resume", action="store_true", help="continue each persona checkpoint")
    parser.add_argument("--config", default="", help="model config path for new runs")
    parser.add_argument("--no-monitor", action="store_true")
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.personas.split(",") if name.strip()]
    logs = PERSONA_DIR / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    queue: list[tuple[str, subprocess.Popen, object, Path]] = []
    pending = list(names)
    started = time.perf_counter()
    failed: list[str] = []

    def launch(name: str) -> tuple[str, subprocess.Popen, object, Path]:
        target = PERSONA_DIR / name / "generator.pt"
        if target.exists() and not args.force and not args.resume:
            print(f"skip {name}: {target} exists (use --force)")
            return name, None, None, target
        (PERSONA_DIR / name).mkdir(parents=True, exist_ok=True)
        if BASE_TOKENIZER.exists():
            shutil.copyfile(BASE_TOKENIZER, PERSONA_DIR / name / "tokenizer.pt")
        config_source = Path(args.config) if args.config else CONFIG_PATH
        shutil.copyfile(config_source, PERSONA_DIR / name / "config.yaml")
        log_path = logs / f"{name}.log"
        handle = log_path.open("w", encoding="utf-8")
        print(f"start {name}: {' '.join(_command(name, args))}")
        process = subprocess.Popen(
            _command(name, args),
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        return name, process, handle, target

    while pending or queue:
        while pending and len(queue) < max(1, args.concurrency):
            name = pending.pop(0)
            if (PERSONA_DIR / name / "generator.pt").exists() and not args.force and not args.resume:
                print(f"skip {name}: checkpoint exists")
                continue
            queue.append(launch(name))
        for entry in list(queue):
            name, process, handle, target = entry
            if process is None:
                queue.remove(entry)
                continue
            code = process.poll()
            if code is None:
                continue
            handle.close()
            queue.remove(entry)
            elapsed = time.perf_counter() - started
            if code == 0 and target.exists():
                print(f"done {name} ({elapsed/60:.1f} min total)")
            else:
                failed.append(name)
                print(f"FAILED {name} exit {code}; see {logs / (name + '.log')}")
                tail = (logs / f"{name}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
                print("\n".join(tail))
        time.sleep(2)

    print(f"\nall jobs finished in {(time.perf_counter() - started)/60:.1f} min")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
