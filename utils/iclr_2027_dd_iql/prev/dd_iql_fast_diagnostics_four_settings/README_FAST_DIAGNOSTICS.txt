Fast diagnostic scripts for IQL / DD-IQL
========================================

Main changes
------------
1. history.jsonl loading is optimized:
   - skip groupby when timesteps are already unique
   - use pandas groupby.last() only when duplicate timesteps exist
   - avoid repeated DataFrame copies

2. W&B _step is converted to the real training timestep:
   training_timestep = _step * config["diagnostic_freq"]
   when an explicit "timestep" is not present.

3. Diagnostics 01/02/03/05 use at most 20 checkpoints per seed by default.
   The checkpoints are selected across the full 5k--100k diagnostic window.
   Fixed-step subsampling is intentionally avoided because it can repeatedly
   hit the same delayed-filter phase when D=1000 and diagnostic_freq=250.

4. Analysis 01 samples up to 20 checkpoints independently for each delay age
   (0, 250, 500, 750), preserving the delay-age comparison.

5. Analysis 04 is unchanged conceptually: it uses final evaluation values.

Usage
-----
Run one analysis:

  python 02_receiver_td_coupling.py

Run with 40 diagnostic checkpoints per seed:

  python 02_receiver_td_coupling.py --diag-points 40

Use every logged diagnostic checkpoint (full calculation):

  python 02_receiver_td_coupling.py --diag-points 0

Run all five analyses in one Python process so the in-memory history cache is
shared across analyses:

  python 00_run_all_diagnostics.py

You can also pass common options, for example:

  python 00_run_all_diagnostics.py \
      --log-root logs/wandb_logs_2026_09_22_orl_160 \
      --diag-points 20 \
      --early-start 5000 \
      --early-end 100000

Recommended workflow for the paper
----------------------------------
- Use --diag-points 20 for fast iteration.
- Before finalizing tables, rerun the important analyses with --diag-points 0
  and verify that the qualitative conclusions are unchanged.

Four-setting table order
------------------------
All LaTeX tables now include all four settings for every environment in this
fixed order:

  1. Original, tau=0.9
  2. Original, tau=0.99
  3. High-gamma, tau=0.9
  4. High-gamma, tau=0.99

Original gamma is environment-specific:
  - HumanoidMaze Medium: gamma=0.995
  - AntMaze Large / Cube / Puzzle / Scene: gamma=0.99
High-gamma is gamma=0.999 for every environment.

Each output row now includes an explicit Setting column (Original or
High-$\\gamma$).  The recommended files for direct Overleaf copy/paste are
named latex_*_table_rows.txt.  The older latex_*_primary_rows.txt filenames
are also overwritten with the same four-setting rows for backward
compatibility, so they no longer contain only the bias-stress setting.
