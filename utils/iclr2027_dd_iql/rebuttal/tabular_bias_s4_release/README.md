# Four-state tabular bias experiment

This review artifact contains **DF-IQL, SD-IQL and DD-IQL only**. The entire
implementation is in `tabular_bias_s4.py`; it has no imports from a research
repository and requires no datasets, checkpoints or external accounts.

## Installation and execution

The tested environment is Python 3.10.9, NumPy 1.23.5, SciPy 1.10.0,
Numba 0.56.4 and Matplotlib 3.7.0. From this directory:

```bash
python -m pip install -r requirements.txt

# Full experiment: 48 conditions, 20 seeds each, 1,000,000 updates per seed.
python tabular_bias_s4.py --workers 4 --output results_s4

# Short run to check installation; these are not the 1M-step results.
python tabular_bias_s4.py --steps 20000 --seeds 200 201 --workers 2 --output smoke_s4

# Inspect the default configuration without running or writing results.
python tabular_bias_s4.py --dry-run

# Render the supplied 20-seed reference results without training.
python tabular_bias_s4.py --summary-csv reference_summary.csv --output reference_preview
```

Parallel execution uses independent worker processes and supports the `spawn`
start method. One full noise array takes 256 MB per active worker, in addition
to temporary arrays and the Python/scientific-library runtime. Reduce
`--workers` on machines with less memory. Completed raw results are reused only
after checksum verification. The configuration and source hash must match to
resume an existing output directory.

## Controlled example

- Four states, four actions, two Q estimators; all tables start at zero.
- The initial state is 0. Action `a` deterministically leads to state `a`.
  There are no terminal states and every reward is zero. The behavior action is
  sampled uniformly at every update, with the same sequence for all methods.
- The known noise-free references are `Q*=V*=F*=0` everywhere. This construction
  isolates estimation bias; it does not evaluate differences in policy return.
- Discount `gamma=0.999`; Q/V learning rates `0.1`; target-Q EMA rate `0.005`.
- Expectiles `tau=0.9,0.99`; refresh periods `D=1,10,100,1000`;
  seeds `200,...,219`; one million updates per trajectory.
- Two synthetic unit-variance Gaussian readout-error settings: IID errors over
  time/state/action/head, and a correlated random-feature field with 64 features,
  length scale `0.35`, action scale `1`, temporal AR(1) coefficient `0.99`, and
  no shared latent noise between heads. The latter is a controlled NN-like
  error field, not the output of a trained neural network.

The random-feature construction includes an unused coordinate anchor to fix
the feature layout. The MDP itself has exactly four states. An auxiliary
random stream is drawn but not mixed into either head. These conventions fix
the random-number streams used for the supplied reference results.

## Updates and refresh convention

Write `Y_i = target_Q_i + epsilon_i` for a noisy critic readout.

| Method | V tables | Receiver for V | Filter source |
|---|---:|---|---|
| DF-IQL | One shared V | `min_i Y_i` | Its own noisy minimum and shared V |
| SD-IQL | One V per Q | `Y_i` | The same index `i` |
| DD-IQL | One V per Q | `Y_i` | The other index `1-i` |

At updates `1,D+1,2D+1,...`, form the entire filter from the **current**
target-Q readouts and V tables before either training update. Each weight is
`tau` for a nonnegative source advantage and `1-tau` otherwise. Use that filter
immediately and reuse it for the following `D-1` updates. Thus its readout-noise
lags are exactly `0,...,D-1`. This is the only refresh convention in this code.

For each sampled transition, both updates read the pre-update tables:

```text
Q_i(s,a) <- 0.9 Q_i(s,a) + 0.1 gamma V_i(s')
V_i(s)   <- (1 - 0.1 w_i) V_i(s) + 0.1 w_i Y_i(s,a)
target_Q <- 0.995 target_Q + 0.005 Q
```

DF uses its shared V and minimum receiver in these expressions. Its two clean
Q tables therefore remain equal under this deterministic tabular setup, while
their injected readout errors are distinct. SD and DD have paired Q/V tables.
All methods consume identical behavior and exogenous error streams per seed.
The final reported policy is the argmax of the ensemble-mean clean Q table.

## Measurements

**Q-bias** is `mean_s max_a mean_i Q_i(s,a)` from the final clean learned tables.
Ensemble averaging precedes action maximization.

**Bias against the noise-free filtering target** is the signed mean of

```text
F_i(s) = sum_a w_i(s,a) Y_i(s,a) / sum_a w_i(s,a)
```

relative to `F*=0`. This normalized diagnostic is not a replacement for the
expectile gradient update and is not the learned V table. Negative values
mean underestimation. Filtering RMSE and separate over/under magnitudes are
also saved. Its clean accumulated-error and readout-error contributions sum
to the filtering bias.

**Coupling** in the figures/tables is the within-seed average of absolute
temporal Pearson correlations between the unnormalized filter weights and
the receiver readout errors, at each fixed receiver/state/action. For SD/DD
the readout error is `epsilon_i`; for DF it is
`min_i(target_Q_i+epsilon_i) - min_i target_Q_i`.

At 1M updates, filtering metrics use 101 pre-update observations from the final
100k updates. Each seed first averages its own observations; reported means
and sample standard deviations are then computed across training seeds.
Short runs use the final 10% of updates and a shortened diagnostic schedule.
The complete 1M schedule samples 1,003 observations and covers every delay phase.

The CSV also includes signed correlations, covariance, valid-unit fractions,
and within-checkpoint state/action correlations for these errors/proxies:

- `readout`: the injected receiver readout error defined above.
- `accumulated`: the clean target-Q receiver's error relative to zero.
- `oracle`: total receiver error, accumulated plus readout.
- `td_update` / `td_observed`: clean-online / observed Bellman residuals.
- `temporal_clean` / `temporal_observed`: drift from the receiver's own snapshot.
- `loo_clean` / `loo_observed`: disagreement with the other head.

These correlations are descriptive diagnostics. Mean absolute sample
correlation can be near `0.08` under independence with 101 observations. It
does not measure the mean bias, imply conditional unbiasedness, or by itself
establish the cause of changes in accuracy. Constant-series correlations are
undefined and excluded from the average; valid fractions are retained.
Single-seed runs have undefined sample SD, shown without SD error bars.

## Outputs and reference values

- `summary.csv`: seed mean/SD for each of the 48 default conditions.
- `per_seed_metrics.csv`: measurements for all 960 default trajectories.
- `raw/<noise>/seed_<seed>/`: Q/V tables, training curves, and actual pre-update
  inputs/gates/targets/normalized filtering outputs in numeric NPZ files;
  accompanying JSON files contain checksums and per-seed metrics.
- `figures/`: 12 title-free single-panel PNGs: three main metrics for each
  noise/expectile combination. All display only DF-IQL, SD-IQL and DD-IQL.
- `tables/`: three LaTeX tables, each with four noise/expectile blocks and D columns.
- `config.json` and `run_metadata.json`: settings and numerical-library versions.

`reference_summary.csv` contains the existing 20-seed, 1M-step measurements for
these three algorithms only. It is an optional comparison/rendering input;
the simulation does not read it. Rendering it writes metadata explicitly
identifying that no training was run. `verification.json` records the checks
performed on this standalone implementation. The 1M-step results are finite
training-time measurements, not a claim that every condition has converged.

Validation included 30 independent NumPy update/timing checks, 12 zero-noise
checks, and a full 1M-step comparison for seed 200 in all 48 conditions. All
1,584 saved-array comparisons were bitwise equal to the existing experiment;
4,368 per-seed diagnostic/accuracy comparisons agreed within absolute tolerance
`1e-12`. Both full noise streams also matched exactly. A two-seed, 20k-step
parallel smoke run generated all 12 figures and three LaTeX tables. These checks
do not constitute a fresh rerun of all 20 seeds at 1M steps.
