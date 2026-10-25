"""Train the multiscale VQ-VAE tokenizer on TBS order lines.

The tokenizer only ever sees the target side (one order per line).  The state is
the generator's raw-token prefix, so it is not reconstructed here.

Every run writes metrics and samples to ``artifacts/tbs/runs/``; watch it with
``python -m tbs_ncm.dashboard``.

    python -m tbs_ncm.train_tokenizer --epochs 25
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import DATA_DIR, artifact_path, load_config
from .monitor import RunLogger

from models.vqvae import VQVAE
from .chartok import CharTokenizer


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def encode_targets(rows: list[dict], tokenizer, context_length: int):
    ids = [tokenizer.encode_padded(row["target"], context_length) for row in rows]
    return torch.tensor(ids, dtype=torch.long)


def level_bounds(levels: list[int]) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for width in levels:
        bounds.append((start, start + width))
        start += width
    return bounds


def codebook_health(counts: list[torch.Tensor]) -> tuple[list[int], list[float]]:
    """Active codes and perplexity per level: the VQ collapse detector."""
    active: list[int] = []
    perplexity: list[float] = []
    for table in counts:
        table = table.double()
        used = table > 0
        active.append(int(used.sum()))
        probabilities = table[used] / table.sum().clamp(min=1.0)
        entropy = float(-(probabilities * probabilities.log()).sum())
        perplexity.append(round(math.exp(entropy), 1) if used.any() else 0.0)
    return active, perplexity


def evaluate(
    model,
    ids: torch.Tensor,
    rows: list[dict],
    tokenizer,
    context_length: int,
    batch_size: int,
    levels: list[int],
    vocab_size: int,
    example_cap: int = 240,
):
    model.eval()
    device = next(model.parameters()).device
    exact = 0
    correct = 0
    total = 0
    counts = [torch.zeros(vocab_size, dtype=torch.float64) for _ in levels]
    examples: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            stop = min(start + batch_size, len(ids))
            batch = ids[start:stop].to(device)
            codes = model.encode(batch)
            predicted = model.decode(codes).argmax(dim=-1)
            mask = batch != tokenizer.pad_token_id
            correct += int((predicted[mask] == batch[mask]).sum())
            total += int(mask.sum())
            host = codes.detach().cpu()
            for level, (low, high) in enumerate(level_bounds(levels)):
                counts[level] += torch.bincount(
                    host[:, low:high].reshape(-1), minlength=vocab_size
                ).double()
            for text, row in zip(predicted.tolist(), rows[start:stop]):
                decoded = tokenizer.decode(text)
                is_exact = decoded == row["target"]
                exact += is_exact
                if len(examples) < example_cap:
                    examples.append(
                        {"target": row["target"], "generated": decoded, "exact": bool(is_exact)}
                    )
    model.train()
    active, perplexity = codebook_health(counts)
    return exact / max(1, len(rows)), correct / max(1, total), examples, active, perplexity


def pick_samples(samples: list[dict], limit: int = 12) -> list[dict]:
    wrong = [sample for sample in samples if not sample["exact"]]
    right = [sample for sample in samples if sample["exact"]]
    return (wrong[: max(1, limit - 4)] + right[:4])[:limit]


def save_checkpoint(path: Path, model, epoch: int, best: float) -> None:
    try:
        torch.save({"model": model.state_dict(), "epoch": epoch, "best": best}, path)
    except OSError as exc:
        print(f"could not save {path}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="train the TBS order tokenizer")
    parser.add_argument("--train", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--validation", default=str(DATA_DIR / "validation.jsonl"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="tokenizer.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--runs-dir", default="")
    parser.add_argument("--tb", action="store_true")
    parser.add_argument("--no-monitor", action="store_true")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config()
    tokenizer = CharTokenizer()
    context = int(cfg.tokenizer.vqvae.context_length)
    pad_id = tokenizer.pad_token_id

    train_rows = load_rows(Path(args.train))
    val_rows = load_rows(Path(args.validation))
    if args.limit:
        train_rows = train_rows[: args.limit]
    if not val_rows:
        val_rows = train_rows[-max(1, len(train_rows) // 20) :]

    train_ids = encode_targets(train_rows, tokenizer, context)
    val_ids = encode_targets(val_rows, tokenizer, context)
    print(
        f"tokenizer data: train {len(train_rows)}, validation {len(val_rows)}, "
        f"context {context}, vocab {tokenizer.vocab_size}"
    )

    model = VQVAE(**cfg.tokenizer.vqvae).to(args.device)
    levels = [int(width) for width in model.vq.levels]
    codes_per_level = int(cfg.tokenizer.vqvae.vqvae_vocab_size[0])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, args.warmup))
        * max(
            args.min_lr / args.lr,
            0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max(1, total_steps)))),
        ),
    )

    best = -1.0
    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("step", start_epoch * steps_per_epoch))
        best = float(checkpoint.get("best", -1.0))
        scheduler.last_epoch = global_step
        print(f"resumed {args.resume} at epoch {start_epoch}")

    out_path = artifact_path(args.out)
    run = (
        None
        if args.no_monitor
        else RunLogger(
            "tokenizer",
            runs_dir=args.runs_dir or None,
            argv=sys.argv,
            total_steps=total_steps,
            tensorboard=args.tb,
            label=f"e{args.epochs}-bs{args.batch_size}",
        )
    )
    status_live = run is not None and sys.stdout.isatty()

    def status(text: str) -> None:
        if status_live:
            sys.stdout.write("\r" + text[:150].ljust(150))
            sys.stdout.flush()

    def status_clear() -> None:
        if status_live:
            sys.stdout.write("\r" + " " * 150 + "\r")
            sys.stdout.flush()

    started = time.perf_counter()
    epoch = start_epoch - 1
    try:
        with (run if run else nullcontext()):
            for epoch in range(start_epoch, args.epochs + 1):
                order = torch.randperm(len(train_rows))
                running = 0.0
                running_recon = 0.0
                running_vq = 0.0
                window_loss = 0.0
                window_batches = 0
                batches_done = 0
                cut_short = False
                for step, begin in enumerate(range(0, len(train_rows), args.batch_size)):
                    if args.max_steps and global_step >= args.max_steps:
                        cut_short = True
                        break
                    index = order[begin : begin + args.batch_size]
                    batch = train_ids[index].to(args.device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                        logits, vq_loss, *_ = model(batch)
                        recon = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            batch.reshape(-1),
                            ignore_index=pad_id,
                        )
                        loss = recon + vq_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    batches_done += 1

                    value = loss.detach().item()
                    running += value
                    running_recon += recon.detach().item()
                    running_vq += float(vq_loss.detach())
                    window_loss += value
                    window_batches += 1
                    if status_live:
                        status(
                            f"epoch {epoch}/{args.epochs}  step {global_step}/{total_steps}  "
                            f"loss {value:.4f}  recon {recon.detach().item():.4f}  "
                            f"lr {scheduler.get_last_lr()[0]:.2e}"
                        )
                    if run is not None and args.log_every > 0 and global_step % args.log_every == 0:
                        run.log(
                            step=global_step,
                            epoch=epoch,
                            loss=window_loss / max(1, window_batches),
                            recon=recon.detach().item(),
                            vq=float(vq_loss.detach()),
                            lr=scheduler.get_last_lr()[0],
                            gpu=True,
                        )
                        window_loss = 0.0
                        window_batches = 0

                if run is not None and window_batches:
                    run.log(
                        step=global_step,
                        epoch=epoch,
                        loss=window_loss / max(1, window_batches),
                        lr=scheduler.get_last_lr()[0],
                        gpu=True,
                    )
                    window_loss = 0.0
                    window_batches = 0

                exact, token_acc, examples, active, perplexity = evaluate(
                    model,
                    val_ids,
                    val_rows,
                    tokenizer,
                    context,
                    args.batch_size,
                    levels,
                    codes_per_level,
                )
                elapsed = time.perf_counter() - started
                status_clear()
                print(
                    f"epoch {epoch:>3}/{args.epochs}  loss {running / max(1, batches_done):.4f}  "
                    f"recon {running_recon / max(1, batches_done):.4f}  "
                    f"val exact {exact:.3%}  val char {token_acc:.2%}  "
                    f"codes {min(active)}-{max(active)}/{codes_per_level}  "
                    f"lr {scheduler.get_last_lr()[0]:.2e}  ({elapsed:.0f}s)"
                )

                improved = exact >= best
                if improved:
                    best = exact
                if run is not None:
                    run.log(
                        kind="epoch",
                        step=global_step,
                        epoch=epoch,
                        loss=running / max(1, batches_done),
                        recon=running_recon / max(1, batches_done),
                        vq=running_vq / max(1, batches_done),
                        val_exact=exact,
                        val_char=token_acc,
                        active_codes=active,
                        perplexity=perplexity,
                        levels=levels,
                        lr=scheduler.get_last_lr()[0],
                        best=improved,
                        minutes=round(elapsed / 60, 2),
                        gpu=True,
                    )
                    run.log_samples(pick_samples(examples), step=global_step, epoch=epoch)

                if improved:
                    save_checkpoint(out_path, model, epoch, best)
                    if epoch == 1 or exact >= 0.999:
                        for sample in examples[:4]:
                            print(f"  {sample['target']!r:32s} -> {sample['generated']!r}")
                    print(f"  saved {out_path} (best val exact {best:.3%})")
                    if run is not None:
                        save_checkpoint(run.dir / "best.pt", model, epoch, best)
                if run is not None and args.save_every and epoch % args.save_every == 0:
                    save_checkpoint(run.dir / "last.pt", model, epoch, best)

                if exact >= 1.0:
                    print("validation reconstruction is exact; stopping")
                    if run is not None:
                        run.milestone(
                            "validation reconstruction is exact; stopping early",
                            step=global_step,
                            epoch=epoch,
                        )
                    break
                if cut_short:
                    if run is not None:
                        run.milestone(
                            f"stopped after --max-steps {args.max_steps}",
                            step=global_step,
                            epoch=epoch,
                        )
                    break
            if run is not None:
                run.finish(
                    state="done",
                    step=global_step,
                    epoch=epoch,
                    best_exact=round(best, 4),
                )
    finally:
        status_clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
