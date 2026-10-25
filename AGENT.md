# Agent Instructions

## Save every game for the dashboard

Any game that is played or generated must be saved so it can be reviewed later
in the dashboard's **Games** tab. Never run a match only for its console output,
and never overwrite an earlier save.

- Put match saves in `artifacts/tbs/matches/` together with their
  `.orders.json` and `.meta.json` sidecars. The dashboard indexes saved matches
  from that directory, plus arena and self-play summaries under
  `artifacts/tbs/vs_ncm/`.
- `tbs_ncm.play` defaults to `artifacts/tbs/matches/ncm_match.json`, which is
  overwritten on every run. Always pass a unique, descriptive `--out` path, for
  example:
  `python -m tbs_ncm.play --opponent greedy --seed 7 --out artifacts/tbs/matches/ncm_vs_greedy_seed7.json`
- Keep the saves written by `arena.py`, `selfplay.py`, and the persona tools
  under `artifacts/tbs/`; do not delete them after reporting results.
- Games played in the window through `play_ui.py` save under the game
  repository's `saves/` directory, which the dashboard does not index. Copy or
  export the game into `artifacts/tbs/matches/` (with its sidecars) if it needs
  a later check.
- Before reporting a match as finished, run `python -m tbs_ncm.dashboard` and
  confirm the game appears in the Games tab.

## Always train with the monitor on

Every training, data or evaluation run must write a run directory under
`artifacts/tbs/runs/` so it appears in the dashboard's **Training** tab. Never
pass `--no-monitor`, and do not delete run directories after reporting results.

- `train_generator.py`, `train_tokenizer.py`, `train_personas.py`,
  `train_vs_ncm.py`, `generate_data.py`, `arena.py`, `selfplay.py`,
  `evaluate_orders.py`, `persona_data.py` and `persona_league.py` all accept
  `--no-monitor`; leave it off so the run is tracked.
- `--runs-dir` only relocates runs; do not use it to skip monitoring.
- Before reporting a run as finished, open `python -m tbs_ncm.dashboard` and
  confirm the Training tab shows the new run with a `live` or `done` badge.
