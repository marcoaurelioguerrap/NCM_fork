# Local setup notes (Windows + RTX 5070 Ti)

Working notes for running the released NCM checkpoints on this machine. Upstream
docs are in `README.md`; this file only records what differs locally.

## Environment

Blackwell (sm_120) needs a CUDA 13 build of PyTorch, and PyPI's Windows `torch`
wheels are CPU-only, so torch must come from the PyTorch CUDA index first:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Verified: `torch 2.14.1+cu130`, `torch.version.cuda == "13.0"`,
`get_device_capability() == (12, 0)`.

`requirements.txt` resolves `transformers` to 5.x (upstream only asks for >=4.51).
The eval and sampling paths work unmodified on 5.20.2, so no pin was added.

Set `HF_HOME` (or leave the default cache) before downloading; downloads here use
`C:\AIs\NCM\.hf`:

```powershell
$env:HF_HOME="C:\AIs\NCM\.hf"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
```

## Checkpoints downloaded

| Task | Size | Path |
|---|---|---|
| sudoku-hard | 163 MB | `artifacts/sudoku/hard`, `data/sudoku/hard` |
| xsum | 654 MB | `artifacts/xsum`, `data/xsum` |

```powershell
hf download hazyresearch/ncm-checkpoints --include "xsum/*" --local-dir artifacts
hf download hazyresearch/ncm-datasets --repo-type dataset --include "xsum/*" --local-dir data
```

`hf` exits with a non-zero status under PowerShell even on success (it writes
progress/warnings to stderr); confirm by checking the files on disk.

## Eval

```powershell
.\.venv\Scripts\python.exe evaluate.py --config config/eval/xsum.yaml `
  --checkpoint-dir artifacts/xsum --data-dir data/xsum --gpus 0 --n 256 --out eval/xsum-256
```

`--n` caps the number of inputs so a first run finishes quickly; drop it for the
full 11,334-document test set. Sudoku uses `--config config/eval/sudoku-hard.yaml`
with `--checkpoint-dir artifacts/sudoku/hard --data-dir data/sudoku/hard`.

Observed on one GPU:

| Run | Result |
|---|---|
| sudoku-hard, 32 boards | solved 75.0%, cell 97.3%, valid 75.0% |
| sudoku-hard, all 2000 boards | solved 86.25% (1725/2000), cell 98.36%, valid 86.25% |
| xsum, 32 docs | R-1 34.59, R-2 12.97, R-L 27.88, genPPL 81.93 |
| xsum, 256 docs | R-1 34.54, R-2 12.74, R-L 27.90, genPPL 80.51 |
| xsum, 256 docs, `--cfg 1.0` | R-1 34.88, R-2 13.32, R-L 28.12, genPPL 109.84 |

The full sudoku board set lands on the 86.1% reported in the Hazy blog, so the
Windows path reproduces the published numbers. The 32-board run is only useful as
a fast smoke test; the per-board variance is large.

Guided decoding (`--cfg 1.0`) buys ~0.3 R-1 / 0.2 R-L on XSum but raises generative
perplexity from ~80 to ~110, so the unguided config is the better default here.

## Sampler

`sample.py` mirrors `scripts/eval_shard.py`'s loading path but takes text from the
command line, a file, or a test-set index. It reproduces eval output bit-for-bit,
including the frequency/adjacency penalties that `pick_tokens` applies outside the
model -- plain `argmax` on the decoder logits gives visibly worse summaries.

```powershell
# one passage from the CLI, showing each coarse-to-fine level
python sample.py --checkpoint-dir artifacts/xsum --text "..."

# a known test item, with its reference summary
python sample.py --checkpoint-dir artifacts/xsum --data-dir data/xsum --dataset-index 7

# interactive: paste a passage per line
python sample.py --checkpoint-dir artifacts/xsum

# plain argmax (no penalties), no intermediate levels
python sample.py --checkpoint-dir artifacts/xsum --text "..." --raw --no-levels
```

Flags: `--temperature/--top-k/--top-p/--cfg` override the config's `generation:`
block; `--seed` fixes sampling.

## Behaviour worth knowing

- The XSum model expects full news articles truncated to 512 GPT-2 tokens; every
  test item is exactly 560 tokens before truncation. Short, hand-written
  paragraphs still produce a summary, but noticeably weaker and sometimes
  repetitive -- that is the checkpoint being out of distribution, not a bug in the
  sampler.
- Decoding is 9 levels over 48 codes, ~0.5 s per passage on the 5070 Ti.
- Perplexity-scored tasks (`tinystories`, `owt`, `lm1b-*`) additionally download
  `gpt2-large` (~3 GB) for `evaluation/gen_ppl.py`.
- `owt` and `lm1b-large` are the heavy checkpoints (folder fetches are ~1-3 GB of
  weights plus multi-GB tokenized data).

## TBS player (`tbs_ncm/`)

The released checkpoints cannot play the turn-based strategy game: the XSum
tokenizer mangles order lines (`'m A2 3,4'` reconstructs as `'A A2 3,4.'`), and
the generator was trained to summarize news. The game therefore gets its own
two-stage NCM model.

Data comes from 200 greedy self-play games on `border_skirmish`, recording the
compact state text and one accepted order per step (54,525 train examples;
legality is checked by the engine). The target side uses a character-level
tokenizer (69 symbols, nine levels `[1,2,3,4,6,8,12,16,28]`, 4096 codes per
level). **Validation reconstruction is 100% exact on all 5,274 held-out orders**
after four epochs.

The generator is an 8-layer, 512-wide NCM transformer (53M parameters). The
prefix is 832 GPT-2 tokens of compact state text through a separate
`ctx_encoder` embedding; the target is the 28-character order hierarchy.

Measured on this machine:

| Check | Result |
|---|---|
| Tokenizer reconstruction (5,274 held-out orders) | 100% exact |
| Generator, 128 held-out states, epoch 4 | 35.9% exact, 100% parseable |
| Generator, 500 held-out states | 25.2% exact, 99.4% parseable |
| Inference | 60 ms per order |
| NCM (B) vs greedy (A), seed 7 | NCM victory by capturing A's HQ, turn 25 |
| NCM (B) vs 74B LLM (A), seed 7 | NCM victory by capturing A's HQ, turn 18 |

In the greedy match, 78 orders came straight from the model, 86 were repaired by
snapping a coordinate or target to a legal option for the chosen unit, and 139
were greedy fallbacks (46% of 303 steps). Against the 74B the split was 66
generated, 100 repaired and 112 fallbacks (40% of 278 steps); the LLM was down to
a single unit by turn 14 and lost its HQ on B's turn 18. The fallback is doing
real work in both games.

```powershell
# from C:\AIs\NCM
.\.venv\Scripts\python.exe -m tbs_ncm.generate_data --train-games 200 --val-games 20 --test-games 20
.\.venv\Scripts\python.exe -m tbs_ncm.train_tokenizer --epochs 25 --batch-size 128
.\.venv\Scripts\python.exe -m tbs_ncm.train_generator --epochs 4 --batch-size 8 --grad-accum 2
.\.venv\Scripts\python.exe -m tbs_ncm.play --opponent greedy --ncm-side B --seed 7
.\.venv\Scripts\python.exe -m tbs_ncm.play --opponent council --ncm-side B --seed 60001 --council-members vanguard,anvil,quartermaster --council-parallel --max-turns 40
.\.venv\Scripts\python.exe -m tbs_ncm.play_ui --no-menu --agent-side B
```

`play_ui` needs pygame-ce in the NCM venv (`uv pip install --python .venv/Scripts/python.exe pygame-ce`).
It runs the game repo's UI with the NCM agent injected and the stepwise session.

## Dashboard

Every stage writes `artifacts/tbs/runs/<timestamp>-<stage>/` with `run.json`,
`metrics.jsonl`, `samples.jsonl`, `status.json` and its checkpoints. The viewer is
stdlib only:

```powershell
.\.venv\Scripts\python.exe -m tbs_ncm.dashboard          # http://127.0.0.1:8765
.\.venv\Scripts\python.exe -m tbs_ncm.dashboard --open   # same, opens the browser
```

The **Training** tab shows live loss/val-loss, exact/parseable/legal, LR,
steps/s, ETA, VRAM, sample diffs, tokenizer codebook usage, arena/self-play
per-game metrics and `vs_ncm` baseline-vs-post iterations. The **Games** tab
lists every saved match plus arena/self-play games; click a saved game to open
per-turn charts, a filterable action list and an action-by-action SVG board
replay. Matchups are written out explicitly (`A · LLM vs B · NCM`, or learner
vs opponent NCM for arena/self-play), and the board legend maps blue/orange to
the player names. A killed process shows as `stalled`, an exception as `crashed`. `--tb`
adds TensorBoard events on top when `tensorboard` is installed; the JSONL log is
unaffected.

For the "legal on the first try" metric, record the legal sets on the held-out
splits (same seeds, same rows, ~5 KB extra per row):

```powershell
.\.venv\Scripts\python.exe -m tbs_ncm.generate_data --splits validation,test --val-games 20 --test-games 20 --record-legal
```
The 74B server must be stopped while training the generator (it used ~10.6 GB of
the 16 GB card); restart it afterward with the command in the project notes.
