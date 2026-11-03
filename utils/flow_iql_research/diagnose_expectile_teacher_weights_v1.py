"""Inspect an expectile-weighted teacher on existing frozen critic outputs.

This is an offline numerical what-if probe, not a new trained policy. The
identity E_p[Q]=e_tau holds for these nine values with p proportional to
tau above the empirical expectile and1-tau below it. It does not hold for
Q(E_p[action]), or automatically for a distilled/projected deployment actor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(values, tau=.9):
    values = np.asarray(values, np.float64)
    assert values.shape == (4, 9, 64) and np.isfinite(values).all()
    lower, upper = values.min(1), values.max(1)
    for _ in range(80):
        root = (lower + upper) / 2
        residual = values - root[:, None, :]
        score = np.sum(np.where(residual >= 0, tau, 1 - tau) * residual, axis=1)
        lower = np.where(score > 0, root, lower)
        upper = np.where(score <= 0, root, upper)
    root = (lower + upper) / 2
    raw = np.where(values >= root[:, None, :], tau, 1 - tau)
    probabilities = raw / raw.sum(1, keepdims=True)
    error = np.sum(probabilities * values, axis=1) - root
    assert np.max(np.abs(error)) < 1e-10
    assert np.max(np.abs(probabilities.sum(1) - 1)) < 1e-12
    bound = tau / (tau + 8 * (1 - tau))
    assert probabilities.max() <= bound + 1e-12
    return probabilities, root, float(np.max(np.abs(error)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    result = json.loads(a.input_result.read_text())
    assert result['status'] == 'pass' and result['training_seed'] == 0
    raw_path = Path(result['raw_predictions'])
    assert sha(raw_path) == result['raw_predictions_sha256']
    rows = []
    with np.load(raw_path, allow_pickle=False) as values:
        for step in [800000, 900000, 1000000]:
            for criterion in ['online_minQ', 'target_meanQ']:
                key = 'online' if criterion == 'online_minQ' else 'target'
                source = [values[f'{step}_{model}_{key}'] for model in ['baseline', 'MDN']]
                qs = [q.min(0) if key == 'online' else q.mean(0) for q in source]
                probs, roots, errors = [], [], []
                for q in qs:
                    p, root, error = weights(q)
                    probs.append(p)
                    roots.append(root)
                    errors.append(error)
                il, ir = [q.argmax(1) for q in qs]
                disagreement = float(np.mean(il != ir))
                rows.append(dict(step=step, criterion=criterion,
                    hard_best9_disagreement_fraction=disagreement,
                    hard_policy_L1_difference_mean=2 * disagreement,
                    expectile_weighted_policy_L1_difference_mean=float(np.mean(np.sum(np.abs(probs[0] - probs[1]), axis=1))),
                    baseline_maxQ_minus_expectile_mean=float(np.mean(qs[0].max(1) - roots[0])),
                    MDN_maxQ_minus_expectile_mean=float(np.mean(qs[1].max(1) - roots[1])),
                    baseline_effective_actions_mean=float(np.mean(np.exp(-np.sum(probs[0] * np.log(probs[0]), axis=1)))),
                    MDN_effective_actions_mean=float(np.mean(np.exp(-np.sum(probs[1] * np.log(probs[1]), axis=1)))),
                    weighted_Q_identity_max_abs_errors=errors,
                    probability_max_observed=float(max(p.max() for p in probs))))
    out = dict(time=time.time(), status='pass', source_sha256={str(Path(__file__).resolve()): sha(__file__)},
        input_result_sha256=sha(a.input_result), raw_predictions_sha256=sha(raw_path),
        tau=.9, samples=9, probability_upper_bound=.9 / (.9 + 8 * .1), rows=rows,
        unique_reset_states=64, independent_noise_banks_per_state=4, training_replicates=0,
        optimizer_updates=0, performance_evaluation_episodes=0, GPU_jobs_allocated=0,
        formal_candidate_or_original_scores_changed=False,
        identity='For an exact finite root e: p_i=w_i/sum(w),w_i=tau if Q_i>=e else1-tau;sum(p_i*Q_i)=e. Numerical values here use float64 bisection on existing float32 predictions.',
        possible_future_loss='-E[sum_i stopgrad(p_i)*log pi_theta(a_i|s)]/D, using target-meanQ weights for consistency with current backup. Not implemented or trained here.',
        limitations=['Only one Ant case, training seed0, reset-state probes.',
            'Online-minQ and target-meanQ are separately labeled. Only the latter matches the backup aggregation.',
            'A lower distribution difference does not establish better expected returns or training stability.',
            'Weighted teachers are less greedy than hard best9; increased lower-Q action mass may harm performance.',
            'The expectation identity belongs to the finite teacher distribution. Gaussian-mixture approximation and zero-noise deployment need separate performance validation.',
            'No action average is executed; expected Q is not Q of the mean action.',
            'Current best9 likelihood actor and all native GPU experiments continue unchanged.'])
    with a.output.open('x') as stream:
        json.dump(out, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status='pass', rows=rows)))


if __name__ == '__main__':
    main()
