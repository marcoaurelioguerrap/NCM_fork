"""Train the NCM generator to predict one order from the compact state prefix.

Teacher forcing follows the released architecture: the prefix is raw GPT-2
token embeddings through the frozen tokenizer's encoder, the target is the
frozen VQ-VAE code hierarchy (level 0 supplied by the learnable query chunk),
and one forward pass predicts every level.

Every run writes metrics and samples to ``artifacts/tbs/runs/``; watch it with
``python -m tbs_ncm.dashboard``.

    python -m tbs_ncm.train_generator --epochs 4 --batch-size 8 --grad-accum 2
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

from .common import DATA_DIR, artifact_path, load_config, load_generator_state
from .chartok import CharTokenizer
from .monitor import RunLogger

from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from tokenization.hf_tokenizer import HFTokenizer

from tbs.agents.codec import CodecError, decode_order, encode_order


def load_rows(path: Path | str) -> list[dict]:
    """Read one JSONL path or a comma-separated list of paths."""
    paths = [Path(part.strip()) for part in str(path).split(",") if part.strip()]
    rows: list[dict] = []
    for item in paths:
        rows.extend(
            json.loads(line)
            for line in item.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def build_tokenizer(cfg, device):
    vqvae = VQVAE(**cfg.tokenizer.vqvae)
    checkpoint = torch.load(artifact_path("tokenizer.pt"), map_location="cpu", weights_only=False)
    vqvae.load_state_dict(checkpoint["model"], strict=True)
    vqvae.eval()
    for parameter in vqvae.parameters():
        parameter.requires_grad_(False)
    return vqvae.to(device)


def load_prefixes(rows, gpt2, prefix_len, device):
    prefixes = []
    longest = 0
    for row in rows:
        ids = gpt2.encode(row["prefix"])
        longest = max(longest, len(ids))
        if len(ids) > prefix_len:
            raise SystemExit(
                f"a prefix needs {len(ids)} GPT-2 tokens, prefix_len is {prefix_len}; "
                "raise model.prefix_len and model.context_length"
            )
        ids = ids + [gpt2.pad_token_id] * (prefix_len - len(ids))
        prefixes.append(ids)
    return torch.tensor(prefixes, dtype=torch.long), longest


def build_targets(rows, tokenizer, context_length, device):
    ids = [tokenizer.encode_padded(row["target"], context_length) for row in rows]
    return torch.tensor(ids, dtype=torch.long)


def generate_texts(net, vqvae, tokenizer, prefixes, batch_size):
    net.eval()
    texts = []
    with torch.no_grad():
        for start in range(0, len(prefixes), batch_size):
            batch = prefixes[start : start + batch_size]
            codes = net.generate(
                None, batch, temperature=0.0, top_k=0, top_p=1.0
            )
            char_ids = vqvae.decode(codes).argmax(dim=-1)
            texts.extend(tokenizer.decode(row) for row in char_ids.tolist())
    net.train()
    return texts


def filter_prefixes(rows, gpt2, prefix_len):
    kept = []
    skipped = 0
    for row in rows:
        if len(gpt2.encode(row["prefix"])) <= prefix_len:
            kept.append(row)
        else:
            skipped += 1
    return kept, skipped


def level_bounds(levels: list[int]) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for width in levels:
        bounds.append((start, start + width))
        start += width
    return bounds


def level_accuracy(predicted: torch.Tensor, codes: torch.Tensor, levels: list[int]) -> list[float]:
    """Per-level teacher-forced accuracy over one batch, coarse to fine."""
    scores = []
    for start, end in level_bounds(levels):
        hit = predicted[:, start:end] == codes[:, start:end]
        scores.append(round(float(hit.float().mean()), 4))
    return scores


def promotion(texts, rows, parse_errors: dict[str, int]) -> dict:
    """Score generated lines: exact, parseable, legal for the stored view."""
    exact = parseable = legal = 0
    checked = 0
    samples = []
    for text, row in zip(texts, rows):
        is_exact = text == row["target"]
        exact += is_exact
        error = None
        order = None
        try:
            order = decode_order(text)
            parseable += 1
        except CodecError as exc:
            error = str(exc).split(";")[0].strip()
            key = error.split("'")[0].strip() or error[:32]
            parse_errors[key] = parse_errors.get(key, 0) + 1

        allowed = row.get("legal")
        if isinstance(allowed, list):
            checked += 1
            if order is not None:
                try:
                    legal += encode_order(order) in set(allowed)
                except CodecError:
                    pass

        samples.append(
            {
                "target": row["target"],
                "generated": text,
                "exact": bool(is_exact),
                "parseable": order is not None,
                "error": error,
            }
        )

    count = max(1, len(rows))
    return {
        "exact": exact / count,
        "parseable": parseable / count,
        "legal_rate": (legal / checked) if checked else None,
        "legal_n": checked,
        "samples": samples,
    }


def pick_samples(samples: list[dict], limit: int = 12) -> list[dict]:
    wrong = [sample for sample in samples if not sample["exact"]]
    right = [sample for sample in samples if sample["exact"]]
    return (wrong[: max(1, limit - 4)] + right[:4])[:limit]


def save_checkpoint(
    path: Path, net, epoch: int, step: int, best: float, select: str = ""
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": net.state_dict(),
            "epoch": epoch,
            "step": step,
            "best": best,
        }
        if select:
            payload["select"] = select
        torch.save(payload, path)
    except (OSError, RuntimeError) as exc:
        print(f"could not save {path}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="train the TBS NCM generator")
    parser.add_argument("--train", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--validation", default=str(DATA_DIR / "validation.jsonl"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument(
        "--select",
        choices=("auto", "legal", "exact", "loss"),
        default="auto",
        help="checkpoint metric: legal-first-try (legal), teacher-target exact "
        "(exact), validation loss (loss), or auto (legal when the validation "
        "rows carry legal sets, otherwise exact)",
    )
    parser.add_argument("--limit", type=int, default=0, help="use only the first N training rows")
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--eval-generate", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", default="", help="model config path (default tbs_ncm/config.yaml)")
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="override model.prefix_len and extend context_length by the same amount",
    )
    parser.add_argument("--out", default="generator.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps")
    parser.add_argument("--log-every", type=int, default=10, help="optimizer steps per metric event")
    parser.add_argument("--save-every", type=int, default=1, help="keep last.pt every N epochs (0 = off)")
    parser.add_argument("--runs-dir", default="", help="metric run directory (default artifacts/tbs/runs)")
    parser.add_argument("--tb", action="store_true", help="also write TensorBoard events")
    parser.add_argument("--no-monitor", action="store_true", help="do not create a run directory")
    parser.add_argument("--resume", default="", help="checkpoint to continue from")
    parser.add_argument(
        "--init",
        default="",
        help="checkpoint to initialize the weights from, with a fresh epoch/schedule "
        "(use this to fine-tune; --resume keeps the old step count)",
    )
    args = parser.parse_args()
    if args.resume and args.init:
        parser.error("--resume and --init are mutually exclusive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config) if args.config else load_config()
    if args.prefix_len:
        old_prefix = int(cfg.model.prefix_len)
        headroom = int(cfg.model.context_length) - old_prefix
        cfg.model.prefix_len = int(args.prefix_len)
        cfg.model.context_length = int(args.prefix_len) + headroom
        print(
            f"context override: prefix_len {old_prefix} -> {cfg.model.prefix_len}, "
            f"context_length {cfg.model.context_length}"
        )
    tokenizer = CharTokenizer()
    context = int(cfg.tokenizer.vqvae.context_length)
    prefix_len = int(cfg.model.prefix_len)

    gpt2 = HFTokenizer("gpt2", pad_token="<|pad|>").hf_tokenizer

    train_rows = load_rows(args.train)
    val_rows = load_rows(args.validation)
    if args.limit:
        train_rows = train_rows[: args.limit]
    print(f"generator data: train {len(train_rows)}, validation {len(val_rows)}")

    train_rows, train_skipped = filter_prefixes(train_rows, gpt2, prefix_len)
    val_rows, val_skipped = filter_prefixes(val_rows, gpt2, prefix_len)
    print(f"skipped overlong prefixes: train {train_skipped}, validation {val_skipped}")

    val_legal_rows = sum(1 for row in val_rows if isinstance(row.get("legal"), list))
    if args.select == "legal" and not val_legal_rows:
        parser.error(
            "--select legal needs validation rows with a 'legal' list; "
            "regenerate the split with --record-legal"
        )
    selection = args.select
    if selection == "auto":
        selection = "legal" if val_legal_rows else "exact"
    print(
        f"checkpoint selection: {selection} "
        f"(validation legal sets: {val_legal_rows}/{len(val_rows)})"
    )

    train_prefix, longest = load_prefixes(train_rows, gpt2, prefix_len, args.device)
    val_prefix, _ = load_prefixes(val_rows, gpt2, prefix_len, args.device)
    train_target = build_targets(train_rows, tokenizer, context, args.device)
    val_target = build_targets(val_rows, tokenizer, context, args.device)
    print(f"longest train prefix {longest} GPT-2 tokens (limit {prefix_len})")

    vqvae = build_tokenizer(cfg, args.device)
    levels = [int(width) for width in vqvae.vq.levels]
    net = NextConceptTransformer(vqvae_local=vqvae, config=cfg.model).to(args.device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total_steps = args.epochs * math.ceil(steps_per_epoch / args.grad_accum)
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

    best = -float("inf")
    start_epoch = 1
    global_step = 0
    init_path = args.resume or args.init
    if init_path:
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        load_generator_state(net, checkpoint["model"])
        if args.resume:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            global_step = int(checkpoint.get("step", 0))
            saved_select = str(checkpoint.get("select") or "exact")
            if saved_select != selection:
                print(
                    f"checkpoint was selected on {saved_select}, now selecting on "
                    f"{selection}; resetting the best score"
                )
            else:
                best = float(checkpoint.get("best", -float("inf")))
            scheduler.last_epoch = global_step
            print(f"resumed {args.resume} at epoch {start_epoch}, step {global_step}")
        else:
            print(f"initialized weights from {args.init}; fresh epoch 1 schedule")

    out_path = artifact_path(args.out)
    run = (
        None
        if args.no_monitor
        else RunLogger(
            "generator",
            runs_dir=args.runs_dir or None,
            argv=sys.argv,
            total_steps=total_steps,
            tensorboard=args.tb,
            label=f"e{args.epochs}-bs{args.batch_size}x{args.grad_accum}",
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

    ema: float | None = None
    started = time.perf_counter()
    promoted = 0

    try:
        with (run if run else nullcontext()):
            for epoch in range(start_epoch, args.epochs + 1):
                order = torch.randperm(len(train_rows))
                running = 0.0
                running_batches = 0
                window_loss = 0.0
                window_batches = 0
                epoch_loss = 0.0
                epoch_batches = 0
                optimizer.zero_grad(set_to_none=True)
                cut_short = False

                for step, begin in enumerate(range(0, len(train_rows), args.batch_size)):
                    if args.max_steps and global_step >= args.max_steps:
                        cut_short = True
                        break
                    index = order[begin : begin + args.batch_size]
                    prefix = train_prefix[index].to(args.device)
                    target = train_target[index].to(args.device)
                    with torch.no_grad():
                        codes = vqvae.encode(target)
                        x_target = vqvae.vq.idx_to_ncm_input(codes)
                    l0 = net.l0_queries.expand(prefix.shape[0], -1, -1)
                    x = torch.cat([l0, x_target], dim=1)
                    pad_mask = prefix == cfg.model.prefix_pad_token_id
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"):
                        logits = net(x, prefix_ids=prefix, prefix_pad_mask=pad_mask)
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]), codes.reshape(-1)
                        )
                    (loss / args.grad_accum).backward()
                    value = loss.detach().item()
                    running += value
                    running_batches += 1
                    window_loss += value
                    window_batches += 1
                    epoch_loss += value
                    epoch_batches += 1

                    if (step + 1) % args.grad_accum == 0 or step + 1 == steps_per_epoch:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        ema = value if ema is None else 0.98 * ema + 0.02 * value
                        if status_live:
                            status(
                                f"epoch {epoch}/{args.epochs}  step {global_step}/{total_steps}  "
                                f"loss {value:.4f}  ema {ema:.4f}  "
                                f"lr {scheduler.get_last_lr()[0]:.2e}"
                            )
                        if run is not None and args.log_every > 0 and global_step % args.log_every == 0:
                            run.log(
                                step=global_step,
                                epoch=epoch,
                                loss=window_loss / max(1, window_batches),
                                loss_ema=ema,
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
                        loss_ema=ema,
                        lr=scheduler.get_last_lr()[0],
                        gpu=True,
                    )
                    window_loss = 0.0
                    window_batches = 0

                # Teacher-forced validation loss and per-level accuracy.
                net.eval()
                val_losses = []
                correct = [0] * len(levels)
                totals = [0] * len(levels)
                eval_count = min(len(val_rows), args.eval_limit)
                with torch.no_grad():
                    for begin in range(0, eval_count, args.batch_size):
                        stop = min(begin + args.batch_size, eval_count)
                        prefix = val_prefix[begin:stop].to(args.device)
                        target = val_target[begin:stop].to(args.device)
                        codes = vqvae.encode(target)
                        x_target = vqvae.vq.idx_to_ncm_input(codes)
                        l0 = net.l0_queries.expand(prefix.shape[0], -1, -1)
                        x = torch.cat([l0, x_target], dim=1)
                        pad_mask = prefix == cfg.model.prefix_pad_token_id
                        with torch.autocast(
                            "cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"
                        ):
                            logits = net(x, prefix_ids=prefix, prefix_pad_mask=pad_mask)
                        val_losses.append(
                            F.cross_entropy(
                                logits.reshape(-1, logits.shape[-1]), codes.reshape(-1)
                            ).item()
                        )
                        predicted = logits.argmax(dim=-1)
                        for level, (start, end) in enumerate(level_bounds(levels)):
                            hit = predicted[:, start:end] == codes[:, start:end]
                            correct[level] += int(hit.sum())
                            totals[level] += hit.numel()
                val_loss = sum(val_losses) / max(1, len(val_losses))
                level_acc = [
                    round(hit / max(1, total), 4) for hit, total in zip(correct, totals)
                ]

                count = min(eval_count, args.eval_generate)
                parse_errors: dict[str, int] = {}
                scores = {"exact": 0.0, "parseable": 0.0, "legal_rate": None, "legal_n": 0, "samples": []}
                if count:
                    texts = generate_texts(
                        net, vqvae, tokenizer, val_prefix[:count].to(args.device), args.batch_size
                    )
                    scores = promotion(texts, val_rows[:count], parse_errors)
                exact, parseable = scores["exact"], scores["parseable"]
                legal_rate = scores["legal_rate"]
                epoch_select = selection
                if epoch_select == "legal" and legal_rate is None:
                    print("  warning: no legal_rate this epoch; selecting on exact instead")
                    epoch_select = "exact"
                if epoch_select == "legal":
                    score = float(legal_rate) + exact * 1e-4 - val_loss * 1e-6
                elif epoch_select == "exact":
                    score = exact * 2.0 - val_loss
                else:
                    score = -val_loss

                epoch_avg = epoch_loss / max(1, epoch_batches)
                elapsed = time.perf_counter() - started
                status_clear()
                print(
                    f"epoch {epoch:>2}/{args.epochs}  loss {epoch_avg:.4f}  "
                    f"val loss {val_loss:.4f}  generated exact {exact:.2%}  "
                    f"parseable {parseable:.2%}  "
                    + (f"legal {legal_rate:.2%}  " if legal_rate is not None else "")
                    + f"select {epoch_select} {score:.4f}  "
                    + f"({elapsed:.0f}s)"
                )
                for text, row in list(zip([s["generated"] for s in scores["samples"]], val_rows))[:4]:
                    print(f"  {row['target']!r:26s} -> {text!r}")

                improved = score > best
                if improved:
                    best = score
                    promoted += 1
                if run is not None:
                    run.log(
                        kind="epoch",
                        step=global_step,
                        epoch=epoch,
                        loss=epoch_avg,
                        loss_ema=ema,
                        val_loss=val_loss,
                        exact=exact,
                        parseable=parseable,
                        legal_rate=legal_rate,
                        legal_n=scores["legal_n"],
                        select_metric=epoch_select,
                        select_score=score,
                        level_acc=level_acc,
                        lr=scheduler.get_last_lr()[0],
                        best=improved,
                        promoted=promoted,
                        eval_n=count,
                        minutes=round(elapsed / 60, 2),
                        gpu=True,
                    )
                    run.log_samples(
                        pick_samples(scores["samples"]), step=global_step, epoch=epoch
                    )
                    if parse_errors:
                        run.log(
                            kind="note",
                            step=global_step,
                            epoch=epoch,
                            text="parse errors: "
                            + ", ".join(f"{key} x{value}" for key, value in parse_errors.items()),
                            parse_error_kinds=len(parse_errors),
                        )

                if improved:
                    save_checkpoint(
                        out_path, net, epoch, global_step, best, select=epoch_select
                    )
                    legal_text = f"{legal_rate:.2%}" if legal_rate is not None else "n/a"
                    print(
                        f"  saved {out_path} (select {epoch_select} {score:.4f}, "
                        f"exact {exact:.2%}, legal {legal_text}, val loss {val_loss:.4f})"
                    )
                    if run is not None:
                        save_checkpoint(
                            run.dir / "best.pt",
                            net,
                            epoch,
                            global_step,
                            best,
                            select=epoch_select,
                        )
                if run is not None and args.save_every and epoch % args.save_every == 0:
                    save_checkpoint(
                        run.dir / "last.pt",
                        net,
                        epoch,
                        global_step,
                        best,
                        select=epoch_select,
                    )

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
                    epoch=args.epochs,
                    best_score=round(best, 4),
                    promoted=promoted,
                )
    finally:
        status_clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
