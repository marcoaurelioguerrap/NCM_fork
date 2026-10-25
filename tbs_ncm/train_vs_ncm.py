"""Train the NCM generator against a fixed NCM opponent.

One command runs the full round:

1. baseline arena: learner vs opponent on the evaluation seeds;
2. self-play data: learner vs the same frozen NCM opponent on fresh seeds;
3. mix those rows with the original greedy dataset and fine-tune the learner
   from its current checkpoint;
4. post-training arena on the same evaluation seeds;
5. a ``report.json`` with both win rates and the new checkpoint path.

The opponent is frozen by default, so this is ``learner vs NCM`` rather than a
moving-target mirror.  ``--update-opponent`` makes each iteration play against
the previous learner instead (fictitious self-play), and ``--iterations``
repeats the round.

    python -m tbs_ncm.train_vs_ncm --games 40 --eval-games 8 --epochs 1

Pass ``--teacher`` to add greedy-labelled rows on the self-play states (DAgger)
alongside the learner's own accepted orders.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from .arena import load_agent, run_arena
from .common import ARTIFACT_DIR, DATA_DIR
from .generate_data import MAX_STEPS_PER_TURN, MAX_TURNS
from .monitor import RunLogger
from .selfplay import collect_game, write_rows

from tbs.agents.greedy import GreedyAgent

__all__ = ["main", "mix_rows"]

DEFAULT_OUT = ARTIFACT_DIR / "vs_ncm"


def _read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mix_rows(
    selfplay_rows: list[dict],
    original_rows: list[dict],
    *,
    original_ratio: float,
    seed: int,
) -> list[dict]:
    """Combine self-play rows with a sample of the original greedy dataset.

    A ratio of 0.5 means "half as many original rows as self-play rows".  The
    original data keeps order-format coverage and guards against catastrophic
    forgetting; the self-play rows teach the states the learner actually visits.
    """
    if original_ratio <= 0:
        mixed = list(selfplay_rows)
    else:
        wanted = int(len(selfplay_rows) * original_ratio)
        wanted = min(wanted, len(original_rows))
        rng = random.Random(seed)
        original = rng.sample(original_rows, wanted) if wanted else []
        mixed = list(selfplay_rows) + original
    random.Random(seed + 1).shuffle(mixed)
    return mixed


def _copy_tokenizer(source: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "tokenizer.pt"
    if target.exists() and target.resolve() == source.resolve():
        return target
    shutil.copyfile(source, target)
    return target


def _release(*agents) -> None:
    for agent in agents:
        try:
            agent.sampler = None
        except Exception:
            pass
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _train_command(
    args: argparse.Namespace,
    *,
    train_path: Path,
    validation_path: Path,
    init_path: Path,
    out_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tbs_ncm.train_generator",
        "--train",
        str(train_path),
        "--validation",
        str(validation_path),
        "--init",
        str(init_path),
        "--out",
        str(out_path),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--eval-limit",
        str(args.eval_limit),
        "--eval-generate",
        str(args.eval_generate),
        "--seed",
        str(args.seed),
        "--log-every",
        str(args.log_every),
    ]
    if args.runs_dir:
        command.extend(["--runs-dir", args.runs_dir])
    if args.device:
        command.extend(["--device", args.device])
    if args.max_steps:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.no_monitor:
        command.append("--no-monitor")
    if args.tb:
        command.append("--tb")
    return command


def _summarize_arena(summary: dict) -> dict:
    return {
        key: summary[key]
        for key in (
            "games",
            "wins",
            "losses",
            "unfinished",
            "win_rate",
            "finished_rate",
            "avg_score_delta",
            "avg_turns",
            "avg_orders",
        )
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tbs_ncm.train_vs_ncm",
        description="fine-tune the NCM generator against a frozen NCM opponent",
    )
    parser.add_argument("--checkpoint-dir", default=str(ARTIFACT_DIR))
    parser.add_argument(
        "--opponent-checkpoint-dir",
        default="",
        help="frozen opponent checkpoint (default: the learner's initial checkpoint)",
    )
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games", type=int, default=40, help="self-play games per iteration")
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=40000)
    parser.add_argument("--eval-seed-base", type=int, default=50000)
    parser.add_argument("--learner-side", choices=("A", "B"), default="B")
    parser.add_argument("--no-alternate-sides", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--opponent-temperature", type=float, default=0.0)
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument(
        "--teacher",
        action="store_true",
        help="add greedy-labelled rows on the self-play states (DAgger)",
    )
    parser.add_argument("--original-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--eval-generate", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="optimizer step cap for a smoke run")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS)
    parser.add_argument(
        "--arena-max-turns",
        type=int,
        default=60,
        help="turn cap for the before/after arena matches (data uses --max-turns)",
    )
    parser.add_argument("--max-game-steps", type=int, default=MAX_STEPS_PER_TURN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--validation", default=str(DATA_DIR / "validation.jsonl"))
    parser.add_argument("--original-data", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--update-opponent", action="store_true")
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--tb", action="store_true")
    args = parser.parse_args(argv)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT / stamp
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    initial_learner = Path(args.checkpoint_dir).resolve()
    frozen_opponent = (
        Path(args.opponent_checkpoint_dir).resolve()
        if args.opponent_checkpoint_dir
        else initial_learner
    )
    validation_path = Path(args.validation).resolve()
    original_path = Path(args.original_data).resolve()
    if not validation_path.exists():
        parser.error(f"validation split not found: {validation_path}")
    if args.original_ratio > 0 and not original_path.exists():
        parser.error(f"original training split not found: {original_path}")
    if not (initial_learner / "generator.pt").exists():
        parser.error(f"missing learner generator: {initial_learner / 'generator.pt'}")
    if not (frozen_opponent / "generator.pt").exists():
        parser.error(f"missing opponent generator: {frozen_opponent / 'generator.pt'}")
    if not (initial_learner / "tokenizer.pt").exists():
        parser.error(f"missing learner tokenizer: {initial_learner / 'tokenizer.pt'}")

    eval_seeds = list(range(args.eval_seed_base, args.eval_seed_base + args.eval_games))
    run = (
        None
        if args.no_monitor
        else RunLogger(
            "vs_ncm",
            runs_dir=args.runs_dir or None,
            argv=argv,
            total_steps=args.iterations * (args.games + 2 * args.eval_games),
            label=f"games{args.games}-e{args.epochs}",
        )
    )
    report: dict = {
        "started": stamp,
        "learner": str(initial_learner),
        "frozen_opponent": str(frozen_opponent),
        "out_dir": str(out_dir),
        "config": vars(args),
        "iterations": [],
    }

    learner_ckpt = initial_learner
    opponent_ckpt = frozen_opponent
    baseline: dict | None = None
    baseline_key: tuple[str, str] | None = None
    started = time.perf_counter()
    step = 0
    try:
        with (run if run else nullcontext()):
            for iteration in range(1, args.iterations + 1):
                iter_dir = out_dir / f"iter{iteration}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"\n=== iteration {iteration}/{args.iterations}  "
                    f"learner {learner_ckpt}  opponent {opponent_ckpt} ==="
                )

                # Re-measure whenever either side changed; carrying a stale
                # baseline across an opponent update would compare two
                # different match-ups.
                key = (str(learner_ckpt), str(opponent_ckpt))
                if baseline is None or baseline_key != key:
                    learner = load_agent(
                        learner_ckpt, device=args.device, temperature=args.opponent_temperature
                    )
                    opponent = load_agent(
                        opponent_ckpt, device=args.device, temperature=args.opponent_temperature
                    )
                    print(f"\nbaseline arena ({len(eval_seeds)} seeds) ...")
                    baseline_summary = run_arena(
                        learner,
                        opponent,
                        seeds=eval_seeds,
                        learner_side=args.learner_side,
                        alternate_sides=not args.no_alternate_sides,
                        max_turns=args.arena_max_turns,
                        max_steps=args.max_game_steps,
                        out_dir=ARTIFACT_DIR / "matches",
                        label=f"vs-ncm-{stamp}-i{iteration}-baseline",
                        logger=run,
                    )
                    baseline = _summarize_arena(baseline_summary)
                    baseline_key = key
                    step += len(eval_seeds)
                    (iter_dir / "arena_baseline.json").write_text(
                        json.dumps(baseline_summary, indent=2) + "\n", encoding="utf-8"
                    )
                    _release(learner, opponent)
                    del learner, opponent
                else:
                    print(
                        f"baseline (already measured for this match-up): "
                        f"{baseline['wins']}W/{baseline['losses']}L "
                        f"({baseline['win_rate']:.1%})"
                    )

                # Self-play data against the frozen opponent.
                print(f"\nself-play data ({args.games} games) ...")
                learner = load_agent(
                    learner_ckpt, device=args.device, temperature=args.temperature
                )
                opponent = load_agent(
                    opponent_ckpt, device=args.device, temperature=args.opponent_temperature
                )
                teacher = GreedyAgent() if args.teacher else None
                rows: list[dict] = []
                games: list[dict] = []
                for index in range(args.games):
                    seed = args.seed_base + (iteration - 1) * args.games + index
                    side = args.learner_side
                    if not args.no_alternate_sides and index % 2 == 1:
                        side = "A" if side == "B" else "B"
                    game_rows, stats = collect_game(
                        learner,
                        opponent,
                        seed=seed,
                        learner_side=side,
                        teacher=teacher,
                        include_fallback=args.include_fallback,
                        max_turns=args.max_turns,
                        max_steps=args.max_game_steps,
                    )
                    rows.extend(game_rows)
                    games.append(stats)
                    print(
                        f"  seed {seed:>6} side {side}: {stats['examples']:>4} rows  "
                        f"{stats['turns']:>3} turns  {stats['status']}"
                        f"{' winner ' + stats['winner'] if stats['winner'] else ''}  "
                        f"kinds {stats['kinds']}"
                    )
                    step += 1
                    if run is not None:
                        run.log(
                            step=step,
                            iteration=iteration,
                            seed=seed,
                            rows=len(rows),
                            game_rows=stats["examples"],
                            learner_won=stats["learner_won"],
                            winner=stats["winner"],
                            generated=stats["learner"]["generated"],
                            repaired=stats["learner"]["repaired"],
                            fallbacks=stats["learner"]["fallbacks"],
                        )

                if teacher is not None:
                    teacher_rows = sum(
                        1 for row in rows if str(row.get("kind", "")).startswith("teacher")
                    )
                    print(f"  teacher rows: {teacher_rows} added")
                if not rows:
                    raise RuntimeError(
                        "self-play produced no rows; try --include-fallback or raise --temperature"
                    )
                selfplay_path = iter_dir / "selfplay.jsonl"
                write_rows(selfplay_path, rows)
                data_summary = {
                    "games": games,
                    "rows": len(rows),
                    "kinds": {},
                    "wins": sum(game["learner_won"] for game in games),
                    "finished": sum(game["status"] != "ongoing" for game in games),
                }
                for row in rows:
                    kind = row.get("kind", "")
                    data_summary["kinds"][kind] = data_summary["kinds"].get(kind, 0) + 1
                data_summary["win_rate"] = (
                    data_summary["wins"] / data_summary["finished"]
                    if data_summary["finished"]
                    else 0.0
                )
                (iter_dir / "selfplay.summary.json").write_text(
                    json.dumps(data_summary, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"  {len(rows)} rows, kinds {data_summary['kinds']}, "
                    f"learner record {data_summary['wins']}W/"
                    f"{data_summary['finished'] - data_summary['wins']}L"
                )
                _release(learner, opponent)
                del learner, opponent, teacher

                # Mix in the original dataset and fine-tune.
                train_rows = list(rows)
                if args.original_ratio > 0:
                    original = _read_rows(original_path)
                    before = len(train_rows)
                    train_rows = mix_rows(
                        train_rows,
                        original,
                        original_ratio=args.original_ratio,
                        seed=args.seed + iteration,
                    )
                    print(
                        f"  mixed {before} self-play rows with "
                        f"{len(train_rows) - before} original rows"
                    )
                else:
                    train_rows = mix_rows(
                        train_rows, [], original_ratio=0.0, seed=args.seed
                    )
                train_path = iter_dir / "train_mix.jsonl"
                write_rows(train_path, train_rows)
                print(f"  training rows: {len(train_rows)} -> {train_path}")

                ckpt_dir = iter_dir / "checkpoint"
                _copy_tokenizer(initial_learner / "tokenizer.pt", ckpt_dir)
                init_path = learner_ckpt / "generator.pt"
                out_path = ckpt_dir / "generator.pt"
                command = _train_command(
                    args,
                    train_path=train_path,
                    validation_path=validation_path,
                    init_path=init_path,
                    out_path=out_path,
                )
                print(f"\ntraining: {' '.join(command)}\n")
                result = subprocess.run(command, check=False)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"train_generator failed with exit code {result.returncode}"
                    )
                if not out_path.exists():
                    raise RuntimeError(f"train_generator did not write {out_path}")
                new_learner = ckpt_dir
                step += args.epochs

                # Post-training arena against the same opponent.
                print(f"\npost-training arena ({len(eval_seeds)} seeds) ...")
                post_learner = load_agent(
                    new_learner, device=args.device, temperature=args.opponent_temperature
                )
                post_opponent = load_agent(
                    opponent_ckpt, device=args.device, temperature=args.opponent_temperature
                )
                post_summary = run_arena(
                    post_learner,
                    post_opponent,
                    seeds=eval_seeds,
                    learner_side=args.learner_side,
                    alternate_sides=not args.no_alternate_sides,
                    max_turns=args.arena_max_turns,
                    max_steps=args.max_game_steps,
                    out_dir=ARTIFACT_DIR / "matches",
                    label=f"vs-ncm-{stamp}-i{iteration}-post",
                    logger=run,
                )
                step += len(eval_seeds)
                (iter_dir / "arena_post.json").write_text(
                    json.dumps(post_summary, indent=2) + "\n", encoding="utf-8"
                )
                _release(post_learner, post_opponent)
                del post_learner, post_opponent
                post = _summarize_arena(post_summary)

                iteration_report = {
                    "iteration": iteration,
                    "baseline": baseline,
                    "post": post,
                    "delta_win_rate": post["win_rate"] - baseline["win_rate"],
                    "delta_score": post["avg_score_delta"] - baseline["avg_score_delta"],
                    "selfplay": data_summary,
                    "checkpoint": str(new_learner),
                    "train_rows": len(train_rows),
                }
                report["iterations"].append(iteration_report)
                print(
                    f"\niteration {iteration}: win rate "
                    f"{baseline['win_rate']:.1%} -> {post['win_rate']:.1%} "
                    f"({post['win_rate'] - baseline['win_rate']:+.1%}); "
                    f"avg score {baseline['avg_score_delta']:+.1f} -> "
                    f"{post['avg_score_delta']:+.1f} "
                    f"({iteration_report['delta_score']:+.1f})"
                )

                # Prepare the next round.
                learner_ckpt = new_learner
                if args.update_opponent:
                    opponent_ckpt = new_learner
                if run is not None:
                    run.log(
                        kind="iteration",
                        step=step,
                        iteration=iteration,
                        baseline_win_rate=iteration_report["baseline"]["win_rate"],
                        post_win_rate=post["win_rate"],
                        delta=iteration_report["delta_win_rate"],
                        train_rows=len(train_rows),
                    )

            report["elapsed_s"] = round(time.perf_counter() - started, 1)
            report_path = out_dir / "report.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(
                f"\nfinal checkpoint: {learner_ckpt}\n"
                f"report: {report_path}\n"
                f"elapsed: {report['elapsed_s']:.0f}s"
            )
            if run is not None:
                run.finish(
                    state="done",
                    iterations=len(report["iterations"]),
                    final_win_rate=report["iterations"][-1]["post"]["win_rate"],
                    elapsed_s=report["elapsed_s"],
                )
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Exception as exc:
        report["error"] = f"{exc.__class__.__name__}: {exc}"
        try:
            (out_dir / "report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
