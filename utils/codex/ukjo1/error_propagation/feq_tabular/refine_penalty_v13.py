"""Actual paired-seed training at finer penalties around v12 transitions.

Each refinement round is recorded before launching its jobs. The learning
kernel, seeds, data, evaluation and training budget are unchanged from v12.
No synthetic/interpolated observations are added to the training results.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .matched_error_v11 import save_csv, save_json
from .penalty_sweep_v12 import (MAIN_SEEDS, METHODS, MODES, config_key, digest,
                               source_hashes, worker)


MAX_CHANGE = .05
MIN_SPACING = {'q': .000025, 'v': .0000125, 'qv': .00000625}
MAX_ROUNDS = 7


def verify_base(base):
    registration = json.loads((base/'registration.json').read_text())
    if registration['source_sha256'] != source_hashes():
        raise ValueError('The original v12 training sources have changed')
    if json.loads((base/'artifact_validation.json').read_text())['status'] != 'passed':
        raise ValueError('Original artifact audit did not pass')
    return registration


def load_records(root, base):
    records = []
    for directory, origin in ((base, 'v12'), (root, 'v13')):
        for path in sorted((directory/'raw').glob('*/result.json')):
            obj = json.loads(path.read_text())
            if obj['config']['stage'] == 'main':
                records.append(dict(config=obj['config'], metrics=obj['metrics'],
                    artifact=str(path.parent.resolve()), origin=origin,
                    data_sha256=obj['data_sha256']))
    keys = [config_key(r['config']) for r in records]
    assert len(keys) == len(set(keys)), 'Duplicate seed/condition observations'
    return records


def grouped(records):
    groups = defaultdict(list)
    for obj in records:
        r = obj['metrics']
        modes = MODES[r['algorithm']] if r['strength'] == 0 else (r['penalty_mode'],)
        for mode in modes:
            groups[r['algorithm'], mode, r['strength']].append(r)
    for key, rs in groups.items():
        assert len(rs) == 16 and {r['seed'] for r in rs} == set(MAIN_SEEDS), key
    return groups


def candidate_points(records, first_round=False):
    """Share new x coordinates among all algorithms using one penalty mode."""
    groups = grouped(records)
    requested = defaultdict(set)
    triggers = []
    for algorithm in METHODS:
        for mode in MODES[algorithm]:
            curve = sorted((strength, np.mean([r['success'] for r in rs]))
                for (alg, pmode, strength), rs in groups.items()
                if alg == algorithm and pmode == mode)
            for (left, yl), (right, yr) in zip(curve, curve[1:]):
                if abs(yr-yl) <= MAX_CHANGE:
                    continue
                if right-left < 2*MIN_SPACING[mode]-1e-13:
                    continue
                triggers.append(dict(algorithm=algorithm, penalty_mode=mode,
                    left=left, right=right, left_success=float(yl), right_success=float(yr)))
                values = np.linspace(left, right, 11)[1:-1] if first_round else [(left+right)/2]
                for value in values:
                    requested[mode].add(round(float(value), 12))
    jobs = []
    for mode, values in requested.items():
        for strength in sorted(values):
            for algorithm in METHODS:
                if mode not in MODES[algorithm] or (algorithm, mode, strength) in groups:
                    continue
                for seed in MAIN_SEEDS:
                    jobs.append(dict(stage='main', algorithm=algorithm, penalty_mode=mode,
                        strength=strength, seed=seed, steps=2000000,
                        lq=strength if mode in ('q', 'qv') else 0.,
                        lv=strength if mode in ('v', 'qv') else 0.))
    return jobs, triggers


def unresolved_changes(records):
    groups = grouped(records)
    rows = []
    for algorithm in METHODS:
        for mode in MODES[algorithm]:
            points = sorted((strength, np.mean([r['success'] for r in rs]))
                for (alg, pmode, strength), rs in groups.items()
                if alg == algorithm and pmode == mode)
            for (left, yl), (right, yr) in zip(points, points[1:]):
                if abs(yr-yl) > MAX_CHANGE:
                    rows.append(dict(algorithm=algorithm, penalty_mode=mode,
                        left=left, right=right, change_pp=float(100*abs(yr-yl)),
                        width=right-left, min_spacing=MIN_SPACING[mode],
                        reason='minimum spacing reached' if right-left < 2*MIN_SPACING[mode]-1e-13 else 'round limit'))
    return rows


def audit(root, base, records):
    verify_base(base)
    expected = {config_key(c) for p in root.glob('round_*.json')
                for c in json.loads(p.read_text())['jobs']}
    actual = {config_key(r['config']) for r in records if r['origin'] == 'v13'}
    assert expected == actual
    grouped(records)
    reference_data = {}
    largest_fixed_residual = 0.
    for obj in records:
        c = obj['config']
        p = Path(obj['artifact'])/'data.npz'
        assert digest(p) == obj['data_sha256']
        with np.load(p) as data:
            assert data['step'][-1] == c['steps'] == 2000000
            assert np.isfinite(data['q']).all() and np.isfinite(data['v']).all()
            if c['seed'] not in reference_data:
                reference_data[c['seed']] = (data['beta'].copy(), data['q_truth'].copy())
            else:
                np.testing.assert_array_equal(data['beta'], reference_data[c['seed']][0])
                np.testing.assert_array_equal(data['q_truth'], reference_data[c['seed']][1])
            selected = data['evaluated_steps'] >= .9*c['steps']
            np.testing.assert_allclose(data['evaluated_success'][selected].mean(), obj['metrics']['success'], atol=1e-13)
            np.testing.assert_allclose(data['evaluated_return'][selected].mean(), obj['metrics']['return'], atol=1e-13)
        largest_fixed_residual = max(largest_fixed_residual, obj['metrics']['fixed_equation_error'])
    return dict(status='passed', new_runs=len(actual), reused_runs=len(records)-len(actual),
        total_main_runs=len(records), new_updates=2000000*len(actual),
        fixed_equation_max_residual=largest_fixed_residual,
        checks='source/checksum/full seed coverage/paired data/end step/finiteness/summary from raw values')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.base.resolve() == args.root.resolve():
        parser.error('Positive workers and a separate output root are required')
    registration = verify_base(args.base)
    plan = dict(version=13, base=str(args.base.resolve()),
        base_registration_sha256=digest(args.base/'registration.json'),
        source_sha256={**source_hashes(), Path(__file__).name:digest(__file__)},
        settings=registration['settings'], seeds=MAIN_SEEDS, steps=2000000,
        strategy=dict(initial='10 equal subdivisions of each original >5pp interval',
            later='midpoint of remaining >5pp intervals, shared within penalty mode',
            target_max_change_pp=5, min_spacing=MIN_SPACING, maximum_rounds=MAX_ROUNDS,
            selection='Adaptive x-grid uses the previous results; no independent hypothesis test implied'),
        inference='No smoothing or synthetic points; every added x value has all 16 original seeds')
    plan = json.loads(json.dumps(plan))
    if args.dry_run:
        jobs, triggers = candidate_points(load_records(args.root, args.base), True)
        print(json.dumps(dict(first_round_runs=len(jobs), triggers=triggers, plan=plan), indent=2))
        return
    args.root.mkdir(parents=True, exist_ok=True)
    path = args.root/'registration.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Existing refinement uses different settings/source')
    save_json(path, plan)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index in range(1, MAX_ROUNDS+1):
            path = args.root/f'round_{index:02}.json'
            if path.exists():
                jobs = json.loads(path.read_text())['jobs']
            else:
                jobs, triggers = candidate_points(load_records(args.root, args.base), index == 1)
                if not jobs:
                    break
                save_json(path, dict(round=index, jobs=jobs, trigger_intervals=triggers,
                    source_sha256=digest(__file__)))
            print(json.dumps(dict(round=index, jobs=len(jobs))), flush=True)
            futures = [pool.submit(worker, (str(args.root), c)) for c in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed % 256 == 0 or completed == len(jobs):
                    print(json.dumps(dict(round=index, completed=completed, total=len(jobs),
                        seconds=round(time.monotonic()-start, 1))), flush=True)
    records = load_records(args.root, args.base)
    result = audit(args.root, args.base, records)
    save_json(args.root/'artifact_validation.json', result)
    save_json(args.root/'artifact_index.json', records)
    save_csv(args.root/'per_seed.csv', [dict(**r['metrics'], origin=r['origin'], artifact=r['artifact']) for r in records])
    remaining = unresolved_changes(records)
    save_json(args.root/'remaining_sharp_intervals.json', remaining)
    metadata = dict(status='complete', **{k:v for k,v in result.items() if k!='status'},
        seconds=time.monotonic()-start, remaining_sharp_intervals=len(remaining))
    save_json(args.root/'run_metadata.json', metadata)
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
