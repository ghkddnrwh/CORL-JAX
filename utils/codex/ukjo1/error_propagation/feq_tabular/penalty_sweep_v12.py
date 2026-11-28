"""Registered ridge-strength sweep of canonical IQL, IQL-9 and FEQ-N9.

Reuses the independently validated v11 update kernel without changing it.
The penalty is an explicit loss intervention, not a default algorithm setting.
All generated artifacts must be placed in the user-selected logs directory.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numba
import numpy as np
import scipy

from .behavior_flow import fit, offline_counts
from .matched_error_v11 import ridge_reference, save_csv, save_json
from .mdp import chain, evaluate_policy, finite_roots, multinomial_counts, roots, solve
from .score_controls_v11 import train
from .validate_matched_v11 import reference


PENALTIES = (0., .0001, .0003, .001, .0015, .002, .003, .005, .007, .01, .02, .03)
METHODS = {'iql': 2, 'iql9': 0, 'feq9': 1}
MAIN_SEEDS = tuple(range(2600, 2616))
LONG_SEEDS = tuple(range(2650, 2654))
MODES = {'iql': ('q', 'v', 'qv'), 'iql9': ('q', 'v', 'qv'), 'feq9': ('q',)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in (
        'penalty_sweep_v12.py', 'score_controls_v11.py', 'matched_error_v11.py',
        'mdp.py', 'behavior_flow.py', 'validate_matched_v11.py')}


def make_jobs():
    jobs = []
    for stage, seeds, steps, strengths in (
        ('main', MAIN_SEEDS, 2000000, PENALTIES),
        ('long', LONG_SEEDS, 5000000, (0., .0015, .003, .01)),
    ):
        for seed in seeds:
            for strength in strengths:
                for algorithm in METHODS:
                    # At zero all penalty locations are the same computation.
                    for mode in (('none',) if strength == 0 else MODES[algorithm]):
                        jobs.append(dict(stage=stage, algorithm=algorithm,
                            penalty_mode=mode, strength=strength, seed=seed, steps=steps,
                            lq=strength if mode in ('q', 'qv') else 0.,
                            lv=strength if mode in ('v', 'qv') else 0.))
    return jobs


def config_key(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def preflight():
    """Check the requested update paths against the independent NumPy model."""
    m = chain(3, .25, .999, .9, .9)
    checks = 0
    maximum = 0.
    for algorithm, method in METHODS.items():
        for mode in MODES[algorithm]:
            for strength in (0., .003, .03):
                lq = strength if mode in ('q', 'qv') else 0.
                lv = strength if mode in ('v', 'qv') else 0.
                actual = train(m.reward, m.successor, m.continuation, m.behavior,
                    m.tau, m.gamma, method, 0, 0., 173, 2600, lq, lv, log_every=1)
                q, v = reference(m, 173, 2600, 'score', method=method,
                    channel=0, eta=0., lq=lq, lv=lv)
                maximum = max(maximum, float(abs(q-actual[1]).max()),
                              float(abs(v-actual[2]).max()))
                checks += 1
    assert maximum < 1e-11
    return dict(status='passed', independent_traces=checks, max_error=maximum)


def worker(payload):
    output, config = payload
    c = config
    directory = Path(output)/'raw'/config_key(c)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory/'result.json'
    if metadata.exists():
        stored = json.loads(metadata.read_text())
        assert stored['config'] == c
        assert digest(directory/'data.npz') == stored['data_sha256']
        return stored['metrics']
    m = chain(8, .25, .999, .9, .9)
    counts = offline_counts(m.behavior, c['seed'])
    m.behavior = fit(counts)
    truth, value_truth, _ = solve(m)
    t, q, v = train(m.reward, m.successor, m.continuation, m.behavior,
        m.tau, m.gamma, METHODS[c['algorithm']], 0, 0., c['steps'], c['seed'],
        c['lq'], c['lv'])
    finite = c['algorithm'] == 'feq9'
    q_fixed, v_fixed = ridge_reference(m, c['lq'], c['lv'], finite)
    tail = t >= .9*c['steps']
    mid = (t >= .7*c['steps']) & (t <= .8*c['steps'])
    selected = tail | mid
    evaluated_steps = t[selected]
    scores = np.array([evaluate_policy(m, qq) for qq in q[selected]])
    tail_scores = scores[evaluated_steps >= .9*c['steps']]
    mid_scores = scores[(evaluated_steps >= .7*c['steps']) & (evaluated_steps <= .8*c['steps'])]
    qr = np.sqrt(np.mean((q-truth)**2, axis=(1, 2)))
    exact_return, exact_success = evaluate_policy(m, q_fixed)
    current_expectile = roots(q_fixed, m.behavior, m.tau)
    difference = q_fixed-v_fixed[:, None]
    g = np.sum(m.behavior*np.where(difference >= 0, m.tau, 1-m.tau)*difference, axis=-1)
    kappa = np.sum(m.behavior*np.where(q_fixed-current_expectile[:, None] >= 0,
                                     m.tau, 1-m.tau), axis=-1)
    residual_q = (1+c['lq'])*q_fixed-m.reward-m.gamma*m.continuation*v_fixed[m.successor]
    if finite:
        multinomial, coefficients = multinomial_counts(9, 4)
        residual_v = v_fixed-finite_roots(q_fixed, m.behavior, m.tau, multinomial, coefficients)
    else:
        residual_v = g-c['lv']*v_fixed
    equation_error = max(float(abs(residual_q).max()), float(abs(residual_v).max()))
    assert equation_error < 1e-10 and np.isfinite(q).all() and np.isfinite(v).all()
    row = {**c, 'success': float(tail_scores[:, 1].mean()),
        'return': float(tail_scores[:, 0].mean()), 'q_rmse': float(qr[tail].mean()),
        'q_bias': float((q[tail]-truth).mean()),
        'mid_success': float(mid_scores[:, 1].mean()),
        'mid_return': float(mid_scores[:, 0].mean()),
        'mid_q_rmse': float(qr[mid].mean()),
        'exact_success': exact_success, 'exact_return': exact_return,
        'exact_q_rmse': rms(q_fixed-truth),
        'learned_to_fixed_q_rmse': rms(q[tail]-q_fixed),
        'q_local_fit_rms': rms(c['lq']*q_fixed),
        'v_score_residual_rms': rms(g) if not finite else np.nan,
        'v_output_fit_rms': rms(v_fixed-current_expectile) if not finite else np.nan,
        'finite_expectile_bias_rms': rms(v_fixed-current_expectile) if finite else np.nan,
        'kappa_mean': float(kappa.mean()),
        'root_action_gap': float((q[tail, 0, 0]-q[tail, 0, 1]).mean()),
        'exact_root_action_gap': float(q_fixed[0, 0]-q_fixed[0, 1]),
        'fixed_equation_error': equation_error, 'tail_checkpoints': int(tail.sum())}
    temporary = directory/'data.tmp.npz'
    np.savez_compressed(temporary, step=t, q=q, v=v, q_truth=truth, v_truth=value_truth,
        q_fixed=q_fixed, v_fixed=v_fixed, counts=counts, beta=m.behavior,
        evaluated_steps=evaluated_steps, evaluated_return=scores[:, 0],
        evaluated_success=scores[:, 1], q_rmse=qr)
    temporary.replace(directory/'data.npz')
    save_json(directory/'result.tmp.json', dict(config=c, metrics=row,
                                               data_sha256=digest(directory/'data.npz')))
    (directory/'result.tmp.json').replace(metadata)
    return row


def audit(root, jobs):
    registration = json.loads((root/'registration.json').read_text())
    assert registration['source_sha256'] == source_hashes()
    assert {p.parent.name for p in (root/'raw').glob('*/result.json')} == {config_key(c) for c in jobs}
    by_seed = {}
    maximum = 0.
    for c in jobs:
        directory = root/'raw'/config_key(c)
        record = json.loads((directory/'result.json').read_text())
        assert record['config'] == c
        assert record['data_sha256'] == digest(directory/'data.npz')
        with np.load(directory/'data.npz') as data:
            assert data['step'][-1] == c['steps']
            assert np.isfinite(data['q']).all()
            if c['seed'] not in by_seed:
                by_seed[c['seed']] = (data['beta'].copy(), data['q_truth'].copy())
            else:
                np.testing.assert_array_equal(data['beta'], by_seed[c['seed']][0])
                np.testing.assert_array_equal(data['q_truth'], by_seed[c['seed']][1])
            maximum = max(maximum, record['metrics']['fixed_equation_error'])
    return dict(status='passed', trajectories=len(jobs),
                total_updates=sum(c['steps'] for c in jobs), paired_dataset_seeds=len(by_seed),
                fixed_equation_max_residual=maximum)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--audit-only', action='store_true')
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    jobs = make_jobs()
    registration = dict(version=12, source_sha256=source_hashes(), jobs=jobs,
        settings=dict(states=9, actions=4, length=8, progress=.25, safe_fraction=.9,
            gamma=.999, expectile_tau=.9, alpha_q=.1, alpha_v=.1, target_ema=.005,
            target_timing='old online Q', counts_per_state=1024, log_every=10000,
            primary_tail_fraction=.1, deployment_samples=9, success_horizon=1000,
            q_truth='unregularized population-expectile fixed point under empirical beta',
            penalty='half squared value, same lambda for each active regression',
            external_noise=False, flow_model='categorical empirical beta; no extra model perturbation',
            baseline_deduplication='zero penalty run once per algorithm/seed, shared by its plot curves'),
        main_strengths=PENALTIES, main_seeds=MAIN_SEEDS, long_seeds=LONG_SEEDS,
        selection='Grid fixed before this sweep; previous v11 selected this MDP family.')
    # Canonical JSON representation avoids tuple/list mismatches when resuming.
    registration = json.loads(json.dumps(registration))
    if args.dry_run:
        print(json.dumps(dict(trajectories=len(jobs),
            total_updates=sum(c['steps'] for c in jobs), **registration['settings']), indent=2))
        return
    args.root.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        result = audit(args.root, jobs)
        save_json(args.root/'artifact_validation.json', result)
        print(json.dumps(result))
        return
    path = args.root/'registration.json'
    if path.exists():
        if json.loads(path.read_text()) != registration:
            raise ValueError('Existing output uses different source or settings; use a new directory')
    else:
        save_json(path, registration)
    started = time.monotonic()
    save_json(args.root/'preflight_validation.json', preflight())
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, (str(args.root), job)) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 64 == 0 or len(rows) == len(jobs):
                print(json.dumps(dict(completed=len(rows), total=len(jobs),
                    seconds=round(time.monotonic()-started, 1))), flush=True)
    rows.sort(key=lambda r: (r['stage'], r['algorithm'], r['penalty_mode'], r['strength'], r['seed']))
    save_csv(args.root/'per_seed.csv', rows)
    save_json(args.root/'artifact_validation.json', audit(args.root, jobs))
    save_json(args.root/'run_metadata.json', dict(status='complete', trajectories=len(jobs),
        seconds=time.monotonic()-started, workers=args.workers, python=sys.version,
        platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        numba=numba.__version__))
    print(json.dumps(dict(status='complete', trajectories=len(jobs), seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
