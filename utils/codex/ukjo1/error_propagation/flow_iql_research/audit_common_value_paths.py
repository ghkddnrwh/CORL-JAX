"""Check actual late Q/V identity across policy families, without re-evaluation."""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def compare(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return [x for k in a for x in compare(a[k], b[k])]
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        return [x for aa, bb in zip(a, b) for x in compare(aa, bb)]
    aa, bb = np.asarray(a), np.asarray(b)
    assert aa.shape == bb.shape
    if aa.dtype.kind not in 'biufc':
        return [dict(equal=bool(np.array_equal(aa, bb)), max_difference=None)]
    return [dict(equal=bool(np.array_equal(aa, bb)),
                 max_difference=float(np.max(np.abs(aa.astype(np.float64)-bb.astype(np.float64)))))]


def main():
    root = Path('logs/diagnosis/flow_iql_research_20261026')
    groups = []
    for short, step in (('scene', 800000), ('scene', 900000), ('scene', 1000000), ('pz3', 600000)):
        ids = [f'p1_{short}_ifql_c0_bounded_v2_g999_t90_s0',
               f'p2_{short}_fql_policy_bounded_v2_g999_t90_s0',
               f'p1_{short}_distilled_c30_bounded_v2_g999_t90_s0',
               f'm1_{short}_gaussian_bc_bounded_v2_g999_t90_s0',
               f'm1_{short}_mixture_bc_bounded_v2_g999_t90_s0']
        if short == 'scene':
            ids.insert(0, f'w1_{short}_bounded_v2_g999_t90_s0')
        records = []
        anchor = None
        anchor_config = None
        for name in ids:
            run = root/'runs'/name
            config = json.loads((run/'config.json').read_text())
            ident = config['identity']
            path = run/f'checkpoint_{step}.pkl'
            if not path.exists():
                path = root/'analysis/checkpoint_archive'/name/f'checkpoint_{step}.pkl'
            if not path.exists():
                records.append(dict(run=name, status='checkpoint_not_available'))
                continue
            with path.open('rb') as file:
                saved = pickle.load(file)
            assert saved['step'] == step
            if anchor is None:
                anchor, anchor_config = saved, ident
            for field in ('environment', 'seed', 'profile'):
                assert ident[field] == anchor_config[field]
            for field in ('discount', 'expectile', 'value_hidden_dims', 'layer_norm', 'lr', 'batch_size'):
                assert ident['config'][field] == anchor_config['config'][field]
            assert all(x['equal'] for x in compare(anchor['replay_rng'], saved['replay_rng']))
            assert all(x['equal'] for x in compare(anchor['state']['q_updates'], saved['state']['q_updates']))
            assert all(x['equal'] for x in compare(anchor['state']['v_updates'], saved['state']['v_updates']))
            model = {}
            for key in ('modules_value', 'modules_critic', 'modules_target_critic'):
                diffs = compare(anchor['state']['agent']['network']['params'][key],
                                saved['state']['agent']['network']['params'][key])
                model[key] = dict(bitwise_equal=all(d['equal'] for d in diffs),
                                  max_absolute_difference=max(d['max_difference'] for d in diffs))
            metrics = [json.loads(s) for s in (run/'evaluations.jsonl').read_text().splitlines()
                       if json.loads(s)['step'] == step]
            records.append(dict(run=name, status='compared', kind=ident['kind'],
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256((run/'config.json').read_bytes()).hexdigest(),
                same_replay_rng=True, modules=model,
                primary_success_at_this_step=metrics[0]['success_rate'] if metrics else None))
        groups.append(dict(task=short, step=step, anchor=ids[0], records=records))
    result = dict(kind='actual_checkpoint_value_identity_across_policy_families', groups=groups,
        caveat='Equal Q/V isolates the policy path, but policies can differ in density family, training loss and inference budget. This is not a last3 aggregate or an oracle critic-accuracy test.')
    path = root/'analysis/late_policy_family_value_identity.json'
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(result, indent=2)); temp.replace(path)
    for g in groups:
        print(g['task'], g['step'])
        for r in g['records']:
            print(r['run'], r['status'],
                  {k: d['bitwise_equal'] for k, d in r.get('modules', {}).items()},
                  r.get('primary_success_at_this_step'))


if __name__ == '__main__':
    main()
