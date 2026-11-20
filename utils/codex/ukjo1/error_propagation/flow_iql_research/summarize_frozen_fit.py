"""Summarize fixed-Q two-step fitting without conflating it with true residuals."""
import hashlib
import json
from pathlib import Path
import statistics


def main():
    root = Path('logs/diagnosis/flow_iql_research_20261026')
    rows = []
    for path in sorted((root/'analysis/frozen_Q_V_diagnostics_v1').glob('w1_*_[0-9]*.json')):
        record = json.loads(path.read_text())
        if record['step'] not in (800000, 900000, 1000000):
            continue
        initial = next(t for t in record['trace'] if t['extra_frozen_v_steps'] == 0)
        fitted = next(t for t in record['trace'] if t['extra_frozen_v_steps'] == 2)
        changes = {}
        for name in ('same_minibatch', 'independent_train', 'validation'):
            before = initial['sets'][name]['expectile_objective']
            after = fitted['sets'][name]['expectile_objective']
            changes[name] = dict(before=before, after=after, change=after-before,
                                 relative_change=(after-before)/max(before, 1e-12))
        rows.append(dict(run=record['run'], step=record['step'], sets=changes,
                         source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    summary = {}
    for name in ('same_minibatch', 'independent_train', 'validation'):
        ds = [r['sets'][name] for r in rows]
        summary[name] = dict(snapshots_reduced=sum(d['change'] < 0 for d in ds),
                             snapshots_increased=sum(d['change'] > 0 for d in ds),
                             median_relative_change=statistics.median(d['relative_change'] for d in ds))
    result = dict(kind='auxiliary_same_batch_generalization_check',
        snapshots=len(rows), trajectories=len({r['run'] for r in rows}),
        steps=[800000, 900000, 1000000], extra_V_updates=2, summary=summary, rows=rows,
        interpretation='Extra fitting reduces the same-minibatch objective consistently; heldout objectives do not improve consistently. This motivates, but does not establish the benefit of, an independent second fitting batch.',
        limits=['One observed action per state: these objectives are not oracle conditional expectile residuals.',
                'Repeated snapshots from one trajectory are correlated and are not independent seeds.',
                'Only original Gaussian IQL value-study trajectories are included; policy-family duplicates are excluded.',
                'Different snapshots use different Q targets; raw objectives are not directly comparable between methods.',
                'The diagnostic freezes Q, uses CPU arithmetic and makes no change to coupled GPU training.'])
    path = root/'analysis/frozen_fit_last3_summary.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False)); temp.replace(path)
    print(json.dumps({k: result[k] for k in ('snapshots', 'trajectories', 'summary')}, indent=2))


if __name__ == '__main__':
    main()
