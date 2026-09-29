# DD-IQL / IQL diagnostic table scripts for the 2026-09-25 logs

These scripts are an updated version of `dd_iql_fast_diagnostics_four_settings`.
They were adapted to the two different log layouts in the supplied archives:

- **DD-IQL (local logs)**: `logs/iclr2027/iql_based/ogbench/diagnostics_dd_iql`
  - `bias_diagnostics.npz`
  - `delayed_diagnostics.npz`
  - `eval_logs.npz`
  - `config.yaml`
  - `training_status.yaml`
- **IQL (W&B download)**: `logs/wandb_logs_2026_09_25_orl_smooth_144`
  - group / run-id directories containing `history.jsonl` and `config.json`

The code intentionally does **not** assume that the two algorithms log at the same cadence or in the same file format.

## Recommended command

Run from the repository/project root:

```bash
python 00_run_all_diagnostics.py \
  --iql-log-root logs/wandb_logs_2026_09_25_orl_smooth_144 \
  --ddiql-log-root logs/iclr2027/iql_based/ogbench/diagnostics_dd_iql \
  --out-root paper_diagnostic_tables_20260925 \
  --early-start 5000 \
  --early-end 100000 \
  --diag-points 20 \
  --eval-step 1000000
```

`--diag-points 20` first selects checkpoints that are approximately evenly spaced in the requested training-step window *within each seed*, then averages within a seed and finally reports mean/std across seeds. This makes the comparison less sensitive to the fact that the IQL runs were logged with different diagnostic frequencies (250 or 2500 steps) while the DD-IQL common diagnostic stream uses its own cadence.

For W&B histories the parser uses the explicit training step (`training/timestep`, then `timestep`) whenever available. It only falls back to W&B `_step * diagnostic_freq` for older logs that do not contain an explicit training step. This is important because W&B `_step` is a history-row index and is not comparable across runs with different logging cadences.

If `orjson` is installed it is used automatically to parse the large W&B JSONL archive quickly; otherwise the code falls back to Python's standard `json` module.

## What each script produces

- `01_delayed_interval_coupling.py`
  - DD-IQL-only delay-phase analysis.
  - Uses the high-frequency `delayed_diagnostics.npz` stream.
  - Reports `diag/filter_receiver_temporal_corr` / covariance at delay ages inferred from the configured hard delay (for the supplied runs, D=100 -> ages 0, 25, 50, 75).
- `02_receiver_td_coupling.py`
  - IQL vs DD-IQL filter--TD-residual correlation/covariance in the early diagnostic window.
- `03_source_receiver_decoupling.py`
  - DD-IQL estimator-coupling and filtering-signal diagnostics (LOO proxies, source/receiver advantage correlations, estimator-error correlation, mean filter weight).
- `04_value_stability_performance.py`
  - Evaluation success and rollout Q/V calibration diagnostics.
  - By default, a seed is included only when it has an evaluation at `--eval-step` (default 1,000,000).
- `05_mechanism_controls.py`
  - Filter--TD coupling together with TD-residual RMSE and filtering-signal/weight controls.

Every analysis writes both CSV summaries and LaTeX row files.

## Important difference from the older analysis 01

The old script formed a delay-age table for receiver TD correlation. That exact table cannot be reconstructed from the current DD-IQL files at nonzero delay ages: the common `bias_diagnostics.npz` stream logs receiver TD correlation every 2500 steps, while the supplied DD-IQL delay period is D=100, so those coarse checkpoints land at delay age 0. The current files *do* contain a high-frequency delayed stream every 25 steps with `diag/filter_receiver_temporal_corr` and `diag/filter_receiver_temporal_cov`. Therefore analysis 01 now reports those explicitly logged temporal-coupling diagnostics instead of fabricating unavailable TD-correlation values for ages 25/50/75.

## Final-evaluation handling

The updated code is deliberately strict for the evaluation table. Interrupted DD-IQL runs often contain an evaluation at 100k/200k/300k even though the intended training horizon is 1M. Treating the last available partial evaluation as the final result would make the IQL/DD-IQL comparison inconsistent. Therefore:

- default: require `--eval-step 1000000`;
- seeds without that evaluation are excluded from the final mean/std;
- `--allow-earlier-eval` is available only when you explicitly want a non-final/partial comparison.

The output CSV contains an `*_n` column so you can see how many seeds actually contributed to each reported statistic.

## Coverage found in the supplied archives

The DD-IQL archive contains the full **144-config grid** (9 environments x 4 gamma/tau settings x 4 seeds), although some runs are interrupted or incomplete at 1M steps.

The supplied IQL W&B archive contains **110 runs across 7 environments**, not the nominal 144-run / 9-environment grid indicated by the folder name. In the supplied archive:

- Puzzle 4x4 is absent;
- Scene is absent;
- Puzzle 3x3 / original gamma / tau=0.9 has only seeds 0 and 1.

The scripts keep the full DD-IQL condition set and emit `NaN` for unavailable IQL cells rather than silently dropping those environments/conditions. If the missing W&B run directories are added later under the same IQL root, they will be discovered automatically and the tables can be regenerated without code changes.

For strict 1M-step evaluation, the current DD-IQL archive also has no final 1M evaluation for the interrupted AntMaze-Large and AntSoccer-Arena runs; some Cube-Single tau=0.9 seeds are incomplete as well. These seeds are intentionally excluded from analysis 04 unless `--allow-earlier-eval` is supplied.

## Notes on diagnostic interpretation

The scripts preserve the original diagnostic meanings. In particular, TD residuals are Bellman-consistency proxies rather than `Q-Q*`; LOO disagreement is an estimator-specific proxy; and rollout Q-return gaps are calibration/stability diagnostics rather than oracle value bias.
