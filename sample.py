"""Interactive sampler for released NCM checkpoints.

Mirrors the loading path of scripts/eval_shard.py, but takes text from the
command line or stdin instead of the evaluation pickles, and can show the
coarse-to-fine refinement one level at a time.

    python sample.py --checkpoint-dir artifacts/xsum --text "The BBC's ... passage ..."

    python sample.py --checkpoint-dir artifacts/xsum --file passage.txt \
        --temperature 0.9 --no-levels

    python sample.py --checkpoint-dir artifacts/xsum --data-dir data/xsum \
        --dataset-index 7
"""

import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer


def load_allowance(ckpt, gen_path, spec, ckpt_dir):
    """Resolve a `file.pt:a.b.c` reference (eval_shard.load_allowance)."""
    if not spec:
        return None
    if ":" in spec:
        path, ref = spec.split(":", 1)
    else:
        path, ref = spec, ""
    path = os.path.join(ckpt_dir, path)
    obj = ckpt if os.path.abspath(path) == os.path.abspath(gen_path) else torch.load(path, map_location="cpu", weights_only=False)
    for key in [k for k in ref.split(".") if k]:
        obj = obj[key]
    return obj.double()


def pick_tokens(logits, allow=None, freq=0.0, mu=0.0, adj=None):
    """Greedy decode with the frequency/adjacency penalties used by eval_shard."""
    if freq <= 0 and not adj:
        return logits.argmax(-1)
    B, T, V = logits.shape
    lg = logits.double()
    allowance = (allow[:V] * T)[None, :] if allow is not None else None
    a_val = adj[0] if adj else 0.0
    a_win = int(adj[1]) if adj and len(adj) > 1 else 1
    a_decay = adj[2] if adj and len(adj) > 2 else 1.0
    cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist = []
    for t in range(T):
        step = lg[:, t, :].clone()
        if freq > 0:
            excess = (cnt - allowance).clamp_min(0.0) if allowance is not None else cnt
            step -= freq * excess
            if mu > 0 and allowance is not None:
                step += mu * ((T - t) / T) * (allowance - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-a_win:])):
            step.scatter_add_(1, h[:, None], torch.full((B, 1), -a_val * a_decay ** j, dtype=torch.float64, device=lg.device))
        pred = step.argmax(-1)
        out[:, t] = pred
        hist.append(pred)
        cnt.scatter_add_(1, pred[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))
    return out


class Sampler:
    def __init__(self, cfg_path, ckpt_dir, device="cuda"):
        self.cfg = OmegaConf.load(cfg_path)
        self.cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ckpt_dir, self.cfg.checkpoints.tokenizer)
        for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
            if self.cfg.checkpoints.get(name):
                self.cfg.tokenizer.pre_tokenizer[key] = os.path.join(ckpt_dir, self.cfg.checkpoints[name])

        self.device = device
        self.tok = VQVAETokenizer.from_config(self.cfg.tokenizer)
        self.tok.vqvae = self.tok.vqvae.to(device).eval()

        gen_path = os.path.join(ckpt_dir, self.cfg.checkpoints.generator)
        ckpt = torch.load(gen_path, map_location="cpu", weights_only=False)
        self.net = NextConceptTransformer(vqvae_local=self.tok.vqvae, config=self.cfg.model).to(device)
        self.net.load_state_dict(ckpt["model"], strict=True)
        self.net.eval()

        gen_cfg = self.cfg.generation
        self.allow = load_allowance(ckpt, gen_path, str(gen_cfg.frequency_allowance), ckpt_dir) if gen_cfg.get("frequency_allowance") else None
        if self.allow is not None:
            self.allow = self.allow.to(device)
        self.freq = float(gen_cfg.get("frequency_penalty", 0.0) or 0.0)
        self.mu = float(gen_cfg.get("frequency_deficit", 0.0) or 0.0)
        self.adj = [float(v) for v in str(gen_cfg.get("adjacency_penalty", "") or "").split(",")] if gen_cfg.get("adjacency_penalty") else None

        self.prefix_len = int(self.cfg.model.prefix_len)
        self.pad_id = self.cfg.model.get("prefix_pad_token_id", self.tok.pre_tokenizer.pad_token_id)
        self.eos = self.tok.pre_tokenizer.eos_token_id
        self.levels = list(self.tok.vqvae.vq.levels)
        self.unconditional = not bool(self.cfg.data)

    def prefix(self, text):
        ids = self.tok.pre_tokenizer.encode(text)
        return (list(ids)[: self.prefix_len] + [self.pad_id] * self.prefix_len)[: self.prefix_len]

    def decode(self, ids):
        ids = list(ids)
        if self.eos in ids:
            ids = ids[: ids.index(self.eos)]
        return self.tok.pre_tokenizer.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def __call__(self, text="", temperature=1.0, top_k=1, top_p=0.0, cfg_scale=0.0,
                 raw=False, show_levels=True):
        if self.unconditional:
            idx = torch.full((1, 1), self.eos, device=self.device, dtype=torch.long)
            prefix = idx
        else:
            idx = None
            prefix = torch.tensor([self.prefix(text)], device=self.device, dtype=torch.long)

        t0 = time.perf_counter()
        codes = self.net.generate(idx, prefix, temperature=temperature, top_k=top_k,
                                  top_p=top_p, cfg_scale=cfg_scale,
                                  cfg_null_eos=self.eos if self.eos is not None else 50256)
        _, zlevels = self.tok.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
        allow, freq, mu, adj = (None, 0.0, 0.0, None) if raw else (self.allow, self.freq, self.mu, self.adj)

        previews = []
        for i, z in enumerate(zlevels):
            ids = pick_tokens(self.tok.vqvae.decoder(z), allow, freq, mu, adj)
            previews.append((self.levels[i], self.decode(ids[0].tolist())))
        elapsed = time.perf_counter() - t0

        summary = previews[-1][1]
        if show_levels:
            for i, (n_codes, txt) in enumerate(previews):
                tag = "final" if i == len(previews) - 1 else f"level {i + 1}/{len(previews)}"
                print(f"  [{tag:>12}] {n_codes:>3} codes | {txt.strip()[:300]}")
        return {"text": summary, "seconds": elapsed, "levels": previews,
                "tokens": self.levels[-1]}


def main():
    ap = argparse.ArgumentParser(description="sample from a released NCM checkpoint")
    ap.add_argument("--config", default="", help="defaults to <checkpoint-dir>/config.yaml, else config/eval/<task>.yaml")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--file", default="", help="read the passage from a file")
    ap.add_argument("--data-dir", default="", help="needed by --dataset-index")
    ap.add_argument("--dataset-index", type=int, default=None,
                    help="sample from input N of the config's eval split and print the reference")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--cfg", type=float, default=None, help="classifier-free guidance scale")
    ap.add_argument("--raw", action="store_true", help="plain argmax, skip frequency/adjacency penalties")
    ap.add_argument("--no-levels", action="store_true", help="print only the final text")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg_path = args.config
    if not cfg_path:
        local = os.path.join(args.checkpoint_dir, "config.yaml")
        cfg_path = local if os.path.exists(local) else ""
    if not cfg_path:
        sys.exit("pass --config config/eval/<task>.yaml")

    s = Sampler(cfg_path, args.checkpoint_dir)
    g = s.cfg.generation
    temperature = args.temperature if args.temperature is not None else float(g.get("temperature", 1.0))
    top_k = args.top_k if args.top_k is not None else int(g.get("top_k", 1))
    top_p = args.top_p if args.top_p is not None else float(g.get("top_p", 0.0))
    cfg_scale = args.cfg if args.cfg is not None else float(g.get("cfg_scale", 0.0) or 0.0)

    def run(text):
        if args.seed is not None:
            torch.manual_seed(args.seed)
        print(f"\n>>> {text[:120]}{'...' if len(text) > 120 else ''}")
        out = s(text, temperature=temperature, top_k=top_k, top_p=top_p,
                cfg_scale=cfg_scale, raw=args.raw, show_levels=not args.no_levels)
        print(f"\n  SUMMARY: {out['text'].strip()}")
        print(f"  ({out['tokens']} tokens in {out['seconds'] * 1e3:.0f} ms)")
        return out

    if args.text:
        run(args.text)
    elif args.file:
        run(open(args.file, encoding="utf-8").read())
    elif args.dataset_index is not None:
        import pickle
        split = s.cfg.eval.get("split", "test")
        dd = args.data_dir or os.path.join(os.environ.get("NCM_DATA_ROOT", "data"), s.cfg.task)
        items = pickle.load(open(os.path.join(dd, s.cfg.data[split]), "rb"))
        refs = pickle.load(open(os.path.join(dd, s.cfg.data[f"{split}_refs"]), "rb"))
        i = args.dataset_index
        item = items[i]
        passage = item if isinstance(item, str) else s.tok.pre_tokenizer.decode(item, skip_special_tokens=True)
        out = run(passage)
        print(f"\n  REFERENCE: {refs[i] if isinstance(refs[i], str) else s.tok.pre_tokenizer.decode(refs[i], skip_special_tokens=True)}")
    else:
        print("Paste a passage and press Enter (blank line to quit).")
        while True:
            try:
                line = input("\npassage> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            run(line)


if __name__ == "__main__":
    main()
