# Four-state oracle diagnostics

기존 4-state/4-action 실험과 같은 업데이트를 실행하고 업데이트 직전의 진단값을 추가합니다.

## Scope and reproduction

- S=A=4, 1M updates, 20 seeds (200–219), N=2, gamma=0.999.
- IID and NN-like correlated synthetic errors; tau=0.9/0.99; D=1/10/100/1000.
- Clipped Double Q (`max_min`), DF-min, SD-IQL and DD-IQL: 50 conditions.
- DF-min has two Q tables and one shared V; SD/DD have two paired Q/V estimators.
  SD versus DD preserves the estimator structure and changes only the filter source.
- Default output: `logs/iclr2027_dd_iql/rebuttal/tabular_bias_diagnostics_s4_1m/`.
- The original `tabular_bias_1m/raw/` files are required for exact reproduction checks.

From the repository root:

```bash
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_1m/diagnostics/run_replay.py
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_1m/diagnostics/report.py
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_1m/diagnostics/validate.py
```

`--workers` and `--noise-workers` control replay parallelism. Completed files are
verified and skipped on resume. `--validate-only` runs the short capture checks.
All commands accept `--results-dir`.

`capture.py` inserts observation-only statements into the unchanged `engine.run_seed`
source. Training operations, arithmetic order, input actions and noise are retained.
The generated function is saved in the output provenance directory. The replay
requires bitwise equality for **every array** in each original NPZ, including
the learning curves, 100k checkpoints and final Q/V/target/delayed arrays.

## What is recorded

All diagnostic snapshots are taken **before update t and before its snapshot refresh**:
online Q, target Q, current noise, delayed observed Q and its noise, V, delayed V,
sampled state/action, and the weight/backup actually used by that transition.
Full state/action gates are reconstructed from those same observations and checked
against the recorded transition's actual weight and backup.
The instantaneous `diagnostic_operator`, its accumulated/readout components,
`diagnostic_weights` and `diagnostic_snapshot_age` are also stored explicitly.

One snapshot is taken in each 1000-step block. A separate fixed RNG permutes the
1000 possible offsets exactly once, covering every snapshot age for D=10/100/1000.
Updates 1, 100k and 1M are additionally included. No training RNG is consumed.
Pre-update age is `(t-1) % D`. At D=1 the actual gate uses current observations,
not the previous update's stored delayed arrays.

All rewards are zero, so Q-circ=V-circ=F-circ=0. SD/DD receiver readout is
`Y_i = target_Q_i + epsilon_i`. DF uses `Y = min_i(target_Q_i + epsilon_i)`.
The diagnostic filtering output is `F_i = sum_a(w_i * Y_i) / sum_a(w_i)`.
The training update remains expectile SGD; it is **not replaced** by this ratio.

The ratio is split exactly into `operator_accumulated` and `operator_readout`.
For SD/DD these use target_Q and epsilon, respectively. For DF, the clean reference
is `min_i target_Q_i` and the effective readout error is `Y - min_i target_Q_i`.
The latter includes min selection and is not assumed to be centered independent noise.
The clipped baseline has no expectile filter: its diagnostic output is its actual
`max_a min_i(Q_i + epsilon_i)` aggregation. Its backup is gamma times that output.
Its clean/noisy split describes aggregation effects rather than an expectile filter.

## Accuracy and coupling definitions

Signed error, RMSE, positive error magnitude and negative error magnitude are
computed for the filtering/aggregation output, online Q, ensemble-mean Q and V.
RMSE squares errors before averaging receiver/state/action/time, preventing signed
cancellation. Final post-update Q/V errors are also saved separately.
In the new CSV tables, `q_rmse` and `value_rmse` evaluate individual estimators;
`q_ensemble_rmse` evaluates the mean Q. The original NPZ keys retain their original
definitions (including ensemble-mean `q_rmse`) for exact reproduction checks.

For each fixed receiver/state/action, calculate temporal covariance and Pearson
correlation between its applied weight and the following quantities:

- `oracle`: full receiver error Y minus the known reference zero.
- `accumulated`: clean target-Q error (DF: clipped clean target).
- `readout`: injected/effective readout error as defined above.
- `td_update`: online clean Q minus gamma times its own V at the successor;
  this is the actual critic-update residual, with DF's shared V.
- `td_observed`: receiver readout minus gamma times its own next V.
- `temporal_observed`: current receiver readout minus its gate-aligned snapshot.
- `temporal_clean`: corresponding clean target drift. Snapshot clean Q is recovered
  from stored observed Q minus its captured noise; exact age-zero drift is set to zero.
- `loo_observed`: current head's observed readout minus the other head's readout.
- `loo_clean`: corresponding clean target-Q disagreement.

DF has a shared gate and a min-aggregated receiver; its LOO orientation is head 0
minus head 1. SD/DD use receiver minus peer. With N=2, same-time LOO quantities
have opposite signs, so their absolute correlations cannot independently establish
source/receiver separation. DF's clean twin Q tables coincide, making clean LOO constant.
Zero-variance correlations remain NaN with a valid-unit fraction; they are never
silently assigned zero. D=1 gate-aligned temporal drift is zero and its correlation
is undefined by construction.

Correlations are calculated over the entire run and each 100k window, then absolute
correlations are averaged across receiver/state/action units **within each seed**.
These are the unsuffixed temporal metrics. In addition, `*_cross_section_*`
computes correlation across state/action pairs at a fixed receiver and snapshot,
then averages absolute correlations across snapshots/receivers within each seed.
The two aggregation modes are kept separate, including an `aggregation` column
in the proxy–accuracy calibration tables and a separate scatter figure for each mode.
Covariances and signed correlations are also reported. Error statistics average
sampled times within each seed; tables report mean and sample SD across 20 seeds.
The last100k table is a window statistic, not an instantaneous final-step statistic.
For Q/V terminal values, use `final_q_v_per_seed.csv`.

`proxy_accuracy_tracking_*` computes descriptive Pearson/Spearman relationships
across the 12 IQL method/delay settings, separately for each noise/tau and seed,
using the last100k window. These are empirical calibrations, not causal tests or a
proof of conditional unbiasedness. Baseline coupling statistics are not applicable.
