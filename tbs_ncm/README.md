# NCM player for Turn-Based Strategy

This directory trains and runs an NCM player for the Advance-Wars-style engine
in the separate game repository,
[marcoaurelioguerrap/turn_based_strategy_vibe](https://github.com/marcoaurelioguerrap/turn_based_strategy_vibe).
Point `TBS_GAME_REPO` at that checkout; it is separate from the game repository
so torch never leaks into the engine's stdlib-only test suite.

## Quickstart

Download the base checkpoint from the [fork release](https://github.com/marcoaurelioguerrap/NCM_fork/releases),
extract it into `artifacts/tbs`, and point `TBS_GAME_REPO` at the game checkout:

```powershell
git clone https://github.com/marcoaurelioguerrap/turn_based_strategy_vibe.git
git clone https://github.com/marcoaurelioguerrap/NCM_fork.git
cd NCM_fork
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt pygame-ce
# NVIDIA Windows: install the CUDA torch wheel first; see SETUP-LOCAL.md.
# Extract the release ZIP into artifacts\tbs before running the player.
$env:TBS_GAME_REPO = "..\turn_based_strategy_vibe"
python -m tbs_ncm.play --opponent greedy --ncm-side B --seed 7
python -m tbs_ncm.play_ui --no-menu --agent-side B
```

The base release contains the tokenizer, the base generator, `config.yaml`, the
learned ferry action selector (`action_clf.pt`, optional) and a held-out
evaluation sample (`test_sample.jsonl`). The five persona checkpoints used in
the league are not part of the base release. The tokenizer is shared, so a
persona checkpoint only needs its own `generator.pt` and `config.yaml`.

## Why a new tokenizer

The released XSum tokenizer cannot represent orders. Round-tripping order lines
through `artifacts/xsum/tokenizer.pt` mangles them:

```
'm A2 3,4'          -> 'A A2 3,4.'
'l A6 A2'           -> 'l A6 A2m.'
'u A6 A2 2,3'       -> 'A A6 A2 2,33 in'
```

The game orders are a small closed alphabet, so the target side uses a
character-level tokenizer (like NCM's Sudoku checkpoints) and the state prefix
stays raw GPT-2 tokens. The tokenizer only has to reconstruct one order line;
the generator handles the state.

## Pipeline

```
tbs_ncm/generate_data.py   greedy self-play -> (state text, order line) JSONL
tbs_ncm/selfplay.py        NCM-vs-NCM self-play -> the same JSONL schema
tbs_ncm/train_tokenizer.py multiscale VQ-VAE -> artifacts/tbs/tokenizer.pt
tbs_ncm/train_generator.py NCM transformer -> artifacts/tbs/generator.pt
tbs_ncm/train_vs_ncm.py    one full NCM-vs-NCM data + fine-tune + arena round
tbs_ncm/ncm_agent.py       NcmSampler + NcmAgent (one order per step)
tbs_ncm/play.py            headless NCM vs greedy/LLM, replayable order log
tbs_ncm/arena.py           NCM vs an NCM checkpoint, win rate + board score
tbs_ncm/evaluate_orders.py exact/parseable rate on held-out states
tbs_ncm/monitor.py         run directories + metrics.jsonl for every stage
tbs_ncm/gamefeed.py        match index + order-replay frames for the game browser
tbs_ncm/dashboard.py       live local dashboard over runs, games and replays
```

The game-side codec (`tbs.agents.codec`) and the stepwise coordinator
(`run_turn_stepwise`) supply the state text and drive the model one order at a
time.

## Data

`generate_data.py` plays the bundled `border_skirmish` map with the engine's
greedy agent, plus a 30% chance to build at a free site and a 12% chance to pick
a different legal move, so the dataset is not one deterministic line. Games run
for at most 40 rounds; every accepted order becomes one example.

The default run (seeds 1-200 train, 10000-10019 validation, 20000-20019 test):

| split | examples | games |
|---|---|---|
| train | 54,525 | 200 |
| validation | 5,274 | 20 |
| test | 5,425 | 20 |

Action mix in train: 23,715 moves, 19,590 attacks, 7,806 builds, 3,414 captures.
The original border-skirmish run predates the ferry teacher: greedy never loaded,
unloaded or held there, so those actions are absent from that split.

For island maps, use the game repository's ferry-aware generator instead of
regenerating the border-skirmish line:

```powershell
py scripts/generate_ferry_data.py --maps archipelago,archipelago_m --games 50 `
  --amphibious-repeat 4 --out tbs_ncm/data/ferry.jsonl
```

It writes rows with an `amphibious` flag on load/unload orders, so a fine-tune
can oversample the crossings the original dataset never contained.  The
scripted `GreedyAgent` and `PersonaTeacher` both follow the engine's `ferry`
hints now, including the pickup, sail, landing and empty-transport return.

State text is the compact codec (about 500 GPT-2 tokens, max 870 in this run;
prefixes longer than `prefix_len` are skipped at training time).

## Tokenizer

`config.yaml` uses nine levels `[1, 2, 3, 4, 6, 8, 12, 16, 28]`, 4096 codes per
level, character vocabulary (69 symbols including pad and EOS), context 28.

**Validation reconstruction is 100% exact** on all 5,274 held-out orders after
four epochs. That is the gate: a player that cannot spell an order cannot play.

```
python -m tbs_ncm.train_tokenizer --epochs 25 --batch-size 128
```

## Generator

The generator is `NextConceptTransformer` with 8 layers, 8 heads, 512 wide
(53M parameters). The target is the 28-character order hierarchy; the prefix is
the state text as raw GPT-2 ids through the model's `ctx_encoder` (a separate
embedding, `n_layer: 0`), so the tokenizer's char vocabulary and the state
vocabulary do not collide.

```
python -m tbs_ncm.train_generator --epochs 4 --batch-size 8 --grad-accum 2
```

Training uses the frozen tokenizer's code hierarchy as teacher forcing, one
forward pass per batch, bf16 autocast, AdamW and a cosine schedule. Every epoch
it generates orders for 128 held-out states and reports exact match and
parseability; the best checkpoint is saved to `artifacts/tbs/generator.pt`.

## Measured

| Check | Result |
|---|---|
| Tokenizer reconstruction (5,274 held-out orders) | **100% exact** |
| Generator, 128 held-out states, epoch 4 | 35.9% exact, 100% parseable |
| Generator, 500 held-out states | 25.2% exact, 99.4% parseable |
| Inference | 60 ms per order on the 5070 Ti |
| NCM (B) vs greedy (A), seed 7, 40-turn cap | **NCM victory by capturing A's HQ, turn 25** |
| NCM (B) vs 74B Nemotron (A), seed 7, 20-turn cap | **NCM victory by capturing A's HQ, turn 18** (earlier run) |
| 60-game all-AI league vs Qwen3.8-27B IQ3_S | NCM Vanguard 17-13, NCM Siege 8-22, Qwen 16-14 (persona checkpoints) |

In the greedy match the NCM side submitted 78 orders exactly as generated, 86
after the legalizer snapped a coordinate/target to a legal option for the chosen
unit, and fell back to greedy 139 times (46% of its 303 steps). In the LLM match
it submitted 66 generated + 100 repaired orders and fell back 112 times (40% of
its 278 steps); the LLM was down to a single unit by turn 14 and lost its HQ on
B's turn 18. Those two matches predate the nearest-legal pass; the fallback rate
below is the current number.

On identical seeds (25 turns, `border_skirmish` and `archipelago`, infantry vs
armor) the repair/rerank pass plus the 2048 prefix produced 14.3%/14.8% and
16.7%/8.7% fallback, with 83-91% of steps coming from the model (exact plus
repaired) and 9.7-18.0% exact. Every remaining fallback was `no-legal-sample`;
there were no `state-too-long` fallbacks. The old 10-match league, before the
legalizer, was 70.6-79.0% fallback and 21-29% model share.

After the fix, a 10-match `border_skirmish` league (20-turn cap, armor on its
DAgger fine-tune) measured 17.5% / 28.2% / 19.6% / 26.2% / 38.0% fallback and
62-83% model share for aggressive / defensive / infantry / armor / artillery,
or **25.3% fallback and 74.7% model share overall** (up from 74.9% / 25.1%).
All remaining fallbacks were `no-legal-sample`; zero were `state-too-long`.
The league caps at 20 turns, so it is not a direct extension of the old 40-turn
run, but the fallback share is measured over steps in both.

## Training against an NCM opponent

`selfplay.py` plays the learner against a second NCM checkpoint and writes the
same `(prefix, target)` rows `train_generator.py` consumes. `arena.py` runs
seeded NCM-vs-NCM matches; because mirror games often reach the turn cap without
an HQ capture, it reports the win rate when a game finishes plus a cap-safe
board score (`2*sites + units + hp/100`). `train_vs_ncm.py` wraps one round:
baseline arena -> self-play data -> mix with the original greedy split ->
fine-tune from the learner's checkpoint (`--init`) -> post arena.

NCM vs NCM data:

```
python -m tbs_ncm.selfplay --games 30 --seed-base 40000 --max-turns 20 \
  --temperature 0.8 --out artifacts/tbs/selfplay/selfplay_train.jsonl
```

`--teacher` adds a greedy-labelled row for every self-play state (DAgger);
`--teacher-persona <name> --teacher-mode fallback` labels only the states where
the NCM had to fall back, using that persona's scripted teacher.
`--include-fallback` keeps orders that came from the fallback instead of
dropping them. The default keeps only `generated` and `repaired` learner orders.

NCM vs NCM arena:

```
python -m tbs_ncm.arena --games 6 --seed-base 50000 \
  --learner-checkpoint-dir artifacts/tbs --opponent-checkpoint-dir artifacts/tbs
```

One full training round:

```
python -m tbs_ncm.train_vs_ncm --games 30 --eval-games 6 \
  --max-turns 20 --arena-max-turns 30 --epochs 1 --lr 5e-5 --original-ratio 0.5
```

The frozen opponent defaults to the learner's starting checkpoint.
`--update-opponent` makes the next iteration play against the previous learner;
`--iterations N` repeats the round.

A first pure self-imitation round (30 games, 3,373 self-play rows mixed with
1,686 original rows, one epoch) is in `artifacts/tbs/vs_ncm/run1/`. It did not
improve the held-out policy: exact on the first 64 validation states went from
56.3% (original) to 53.1% (fine-tuned), and on the first 128 test states from
32.8% to 30.5%; the post arena board score also dropped (+10.2 -> -48.0 over six
seeds, five unfinished). The original `artifacts/tbs/generator.pt` remains the
production checkpoint. Use `--teacher` (DAgger labels on NCM-generated states)
or filter self-play rows by score/outcome before fine-tuning.

## Persona NCMs

A persona is a scripted decision policy (`tbs_ncm/personas.py`) used to label
on-policy games, then cloned into its own checkpoint. All personas share the
tokenizer and start from the released generator, so they only move the policy.

```
python -m tbs_ncm.persona_data --games 8 --self-games 20 --val-games 2 \
  --test-games 2 --workers 8 --hold-cap 0.30
python -m tbs_ncm.train_personas --concurrency 2
python -m tbs_ncm.persona_league --games-per-map 1
```

Personas: `aggressive`, `defensive`, `infantry`, `armor`, `artillery`. Data is
written to `tbs_ncm/data/personas/<name>/`, checkpoints to
`artifacts/tbs/personas/<name>/`, and the round-robin report to
`artifacts/tbs/personas/league/report.{json,md}`.

`persona_data` cycles every available map, records both sides, and caps hold
orders at 30%: mirrors on defensive maps can otherwise be 90% holds, which
teaches the clone to pass instead of developing a style. `train_personas` runs
at most two jobs at once on the 16 GB card. `persona_league` uses each persona's
scripted teacher as the recovery policy, because the greedy fallback would
otherwise erase 40-60% of the style in the match numbers.

`collect_dagger` replays a few games per persona and keeps only the scripted
teacher's labels on the on-policy states, which is the useful subset for a
short fine-tune:

```
python -m tbs_ncm.collect_dagger --games-per-persona 2 --max-turns 18
```

The DAgger run collected 1,079 teacher rows. Each of the five fine-tunes mixed
`2x dagger + 2x fallback-origin + 1,500 sampled original` rows and trained three
epochs from that persona's release weights with `--prefix-len 2048`. Only
`armor` was promoted: on the same A/B seeds its fallback fell from 14.8%/8.7% to
7.0%/6.0% and model share rose to 93-94%, while held-out exact stayed at 1.3%
and parseable at 99.0%. `infantry` regressed on the A/B (14.3%/16.7% ->
33.7%/25.7% fallback) despite a tiny held-out exact gain, and
aggressive/defensive/artillery were neutral or worse, so those four keep their
release `generator.pt`; the DAgger checkpoints are kept beside them as
`dagger_generator.pt` for comparison. The armor release checkpoint is backed up
as `generator_release.pt`.

Inference defaults to `prefix_len: 2048`. Held-out persona states reach 1,814
GPT-2 tokens, and 10-29% of them exceed 832, so 832 was truncating a real part
of the late game. The checkpoint loader drops the derived
`attn_mask_for_masking` buffer, and RoPE lets the 832-prefix release checkpoint
run at 2048 with unchanged outputs on short states. The training entry point
accepts `--prefix-len 2048` to apply the same override to a fine-tune.

## Playing

Headless against greedy:

```
python -m tbs_ncm.play --opponent greedy --ncm-side B --seed 7
```

Against the local Qwen3.8-27B baseline used in the 60-game league (start
Unsloth Studio first; the server lists the loaded `IQ3_S` weights under the id
without the quant suffix):

```
$env:TBS_LLM_MODEL = "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"
$env:TBS_LLM_API_KEY = "<the key Unsloth prints>"
python -m tbs_ncm.play --opponent llm --ncm-side B --seed 7
```

The earlier 74B Nemotron match in the table above is a historical run; the
current league baseline is the Qwen model.

The NCM side plays one generated order per step through
`run_turn_stepwise`. It builds the legal set from the same view an LLM sees
(affordable builds and units with a usable action), decodes each sample, and
accepts an exact legal order immediately. On maps where the view exposes a
`ferry` operation, `NcmAgent` first applies the same deterministic ferry
advisor as the scripted teachers; the released checkpoint maps `l`/`u` to `m`
even after a short fine-tune, so the advisor is what guarantees the multi-turn
crossing while the model keeps every other decision. A longer fine-tune can
internalize the action later. Otherwise it resamples up to six
times and keeps the best candidate by repair kind and snap distance: same-unit
coordinate/target repair, same-unit alternate action, unit-id snap, then a
global nearest-legal line. Fuzzy unit ids only get a same-action repair, so a
mistyped id cannot be turned into an arbitrary hold. If no sample yields a
legal candidate, the scripted recovery policy runs. Engine feedback suppresses
the last rejected order on the retry. Reports split fallback reasons into
`state-too-long`, `generation-error`, `rejected`, and `no-legal-sample`, and
count exact, repaired, and fallback orders separately.

## Playing in the window

Play in the window (the NCM virtualenv has torch and pygame-ce):

```
python -m tbs_ncm.play_ui --no-menu --agent-side B
```

That launches the game repo's UI with the NCM agent injected and the stepwise
session (`--ncm-max-steps` bounds the batches per turn). The launcher changes to
the game repository first, so saves land in its `saves/` directory.

Held-out scoring (the release ZIP includes `test_sample.jsonl`):

```
python -m tbs_ncm.evaluate_orders --split test_sample.jsonl --limit 500
```

## Benchmark

Measure per-order latency on the released sample.  Run it on an idle GPU for
numbers you intend to publish; `legal` also scores the legal candidate lines,
which is the UI's default path.

```powershell
python -m tbs_ncm.benchmark --split test_sample.jsonl --decode sample --repeat 3 `
  --out artifacts/tbs/benchmark_sample.json
python -m tbs_ncm.benchmark --split test_sample.jsonl --decode legal --limit 20 `
  --out artifacts/tbs/benchmark_legal.json
```

## Learned ferry action selector

The generator is strong at order arguments but collapses the rare `l`/`u`
actions to `m`.  `tbs_ncm/action_model.py` adds a tiny character-n-gram
classifier that predicts just the action character from the state prefix; when
`artifacts/tbs/action_clf.pt` exists, `NcmAgent` uses it on maps with a `ferry`
block.  The generator still supplies the arguments through legal candidate
lines.

```powershell
python -m tbs_ncm.train_action_clf `
  --train "..\saves\ferry_data_train20.jsonl,tbs_ncm/data/train.jsonl" `
  --validation "..\saves\ferry_data_val.jsonl,tbs_ncm/data/validation.jsonl" `
  --out action_clf.pt --epochs 15 --device cuda
```

On the 759-row ferry validation split, the first released classifier reached
84.7% overall, 100% on load (21/21) and unload (14/14), 96.6% on moves and
43.2% on attacks.  The selector is optional; without the file the scripted
ferry advisor is used.

## Watching a run

Every stage writes one run directory:

```
artifacts/tbs/runs/20261020-2231-generator-e4-bs8x2/
  run.json       stage, argv, git commit, host, total steps
  config.yaml    snapshot of tbs_ncm/config.yaml
  metrics.jsonl  one JSON event per line, flushed after every write
  samples.jsonl  target/generated pairs from the epoch evaluation
  status.json    running | done | crashed, pid, heartbeat
  best.pt        best checkpoint of that run
  last.pt        checkpoint from the last finished epoch
  tb/            TensorBoard events, only when started with --tb
```

Live view:

```
python -m tbs_ncm.dashboard            # http://127.0.0.1:8765
python -m tbs_ncm.dashboard --open --port 8766
```

The page polls every 2 s. It is standard library only: no dependency, no build
step, no CDN.

The page has three tabs. **Training** keeps the run explorer; **Games** is a
browser over every saved match plus the arena and self-play summaries;
**Overview** aggregates every saved game into cross-game plots.

| panel | shows |
|---|---|
| KPI row | progress, step/epoch, loss, val loss, exact/parseable/legal, lr, steps/s, ETA, VRAM |
| Loss | train (raw + EMA) and teacher-forced validation on one axis |
| Generation quality | exact match, parseable, legal-on-first-try |
| Samples | target vs generated, wrong characters in red |
| Codebook (tokenizer only) | active codes and perplexity per level; a collapsed level is one bar |
| Games per run | win/loss, score delta, turns, orders and NCM order quality for `arena`, `vs_ncm` and `selfplay` runs |
| Iterations (`vs_ncm`) | baseline vs post win rate and score delta from `vs_ncm/*/report.json` |
| Games tab | search/filter every saved game, result, learner side, score delta and generated/repaired/fallback counts |
| Overview tab | all saved games: results by day and scenario, final-score and game-length histograms, rolling order-quality shares, and evaluation-batch comparisons (outcome, win rate, score delta, provenance, action mix and game length per run) |
| Game detail | per-turn board-score/unit/HP/site/fund charts, objective and kill markers, action-by-action SVG board replay, and NCM/learner quality per turn |

The Overview tab treats one evaluation or league report as a **batch**. It reads
`artifacts/tbs/personas/*/report.json` and the `vs_ncm` arena/self-play summaries,
matches saved games to the run that produced them, and shows the same outcome,
score, provenance, action-mix and turn-length metrics per batch. A batch selector
limits the charts to the latest 8, latest 12 (default), or all runs. Standalone
play games with no run report are still included in the all-games charts but are
excluded from the batch charts.

The Games tab replays the accepted orders in each `*.orders.json` through the
game engine, so a saved game becomes a scrubable board without re-running the
model. Matchups are labelled explicitly: play games show `A · LLM vs B · NCM`
(or `Greedy`) and arena/self-play games show `Learner NCM` versus
`Opponent NCM`, with the winner side always written out. `arena.py` and
`play.py` now write a `<game>.meta.json` sidecar with the scenario path, NCM
side, per-turn provenance and fallback reasons. Older games still get the board
replay and aggregate arena quality, but not per-turn provenance.

The badge reads `live` while the heartbeat is under 30 s, `stalled` when a run
stops without writing (killed process), `done` on a clean finish and `crashed`
when the run context manager caught an exception.

| flag | effect |
|---|---|
| `--log-every N` | optimizer steps per metric event (10 generator, 25 tokenizer) |
| `--save-every N` | write `last.pt` every N epochs (0 disables) |
| `--max-steps N` | stop after N optimizer steps, for smoke runs |
| `--runs-dir DIR` | put run directories somewhere else |
| `--tb` | also write TensorBoard events when `tensorboard` is installed; JSONL is always written |
| `--no-monitor` | skip the run directory entirely |
| `--resume PATH` | continue from a checkpoint (weights, epoch, step and schedule position; AdamW moments restart) |

`evaluate_orders` writes an `eval` run too, and accepts a path for `--split`.

### Legal on the first try

The compact state text deliberately drops `unit_actions`, so this metric needs
the legal order set recorded next to each state. Regenerate only the held-out
splits (same seeds, same rows, plus a `legal` list):

```
python -m tbs_ncm.generate_data --splits validation,test --val-games 20 --test-games 20 --record-legal
```

That adds roughly 5 KB per row (~28 MB per split). The train split is untouched
unless you name it; with `--record-legal` on the 54k-row train split expect
~290 MB extra. Without legal sets the metric and its chart are skipped and the
scorer prints no `legal` figure.

Metrics for the four-epoch run behind the current `generator.pt` were printed to
the console only, so the curves start with the next run. The Games tab is
populated from the existing saves; per-turn provenance starts with the next
game played after the meta-sidecar change.

## Notes and limits

- One scenario (`border_skirmish`) and one scripted teacher (greedy + builds).
  The player imitates that policy, not a strong search.
- No load/unload/hold targets yet. The fallback covers the first two if the
  model emits them by accident.
- The state codec deliberately omits `unit_actions`; legality is enforced by
  the engine and corrected through feedback.
- Training the 74B server must be stopped while the generator trains (it used
  about 10.6 GB of the 16 GB card). Restart it with the command in
  `SETUP-LOCAL.md`.
