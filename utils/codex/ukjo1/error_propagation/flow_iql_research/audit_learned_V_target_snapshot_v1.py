"""Separate raw/projected current-V snapshot differences on fixed replay queries.

Both are diagnostics, not original training minibatch errors or population zeta.
"""
import argparse
import json
from pathlib import Path
import time

from utils.flow_iql_research import core, policy_core
from utils.flow_iql_research.evaluate_cross_seed_temporal_v2 import read
from utils.flow_iql_research.rank_mean_bundle_v1 import sha
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan_path = args.root / 'independent_critic_factorial_registration_v1.json'
    plan = json.loads(plan_path.read_text())
    rows, evidence, arrays = [], [], {}
    for short in ('ant', 'hmmed', 'pz3'):
        job = next(j for j in plan['jobs'] if Path(j['actor_run']).name.split('_')[1] == short)
        run = Path(job['actor_run'])
        conf = json.loads((run / 'config.json').read_text())
        identity = conf['identity']
        assert identity['kind'] == 'ifql' and identity['profile'] == 'bounded_v2'
        assert (run / 'completed.json').exists() and not (run / 'stopped.json').exists()
        env, data, _ = ogbench.make_env_and_datasets(identity['environment'])
        indices = np.random.RandomState(identity['seed'] + 19401).randint(len(data['actions']), size=1024)
        fields = ('observations', 'actions', 'next_observations', 'rewards', 'masks')
        batch = {k: np.asarray(data[k][indices], np.float32) for k in fields}
        batch['actions'] = np.clip(batch['actions'], -1 + 1e-5, 1 - 1e-5)
        cfg = dict(identity['config'])
        for name in ('actor_hidden_dims', 'value_hidden_dims'):
            cfg[name] = tuple(cfg[name])
        template = policy_core.create(identity['seed'], jnp.asarray(batch['observations'][:1]),
            jnp.asarray(batch['actions'][:1]), cfg, 'ifql')
        dataset_path = run / 'dataset.json'
        dataset = json.loads(dataset_path.read_text())
        bounds = [np.float32(dataset['return_lower']), np.float32(dataset['return_upper'])]
        diagnostics = [json.loads(line) for line in (run / 'diagnostics.jsonl').read_text().splitlines()]
        for step in (800000, 900000, 1000000):
            actual_conf, saved, checkpoint = read(run, step)
            assert actual_conf == conf and int(saved['state']['v_updates']) == 2 * step
            state = flax.serialization.from_state_dict(template, saved['state'])
            values = core.probe(state, {k: jnp.asarray(v) for k, v in batch.items()})
            q, target_q, v, raw, finite = jax.device_get(values)
            assert bool(finite) and np.isfinite(q).all() and np.isfinite(raw).all()
            projected = np.clip(raw, *bounds)
            rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))
            reference = next(r for r in diagnostics if r['step'] == step)
            row = dict(short=short, environment=identity['environment'], step=step,
                queries=1024, lower=float(bounds[0]), upper=float(bounds[1]),
                current_V_unprojected_snapshot_TD_RMS=rms(q.astype(np.float64) - raw[None].astype(np.float64)),
                current_V_projected_snapshot_TD_RMS=rms(q.astype(np.float64) - projected[None].astype(np.float64)),
                target_projection_change_fraction=float(np.mean(raw != projected)),
                target_projection_change_RMS=rms(raw.astype(np.float64) - projected.astype(np.float64)),
                legacy_GPU_probe_unprojected_TD_RMS=reference['td_rms'],
                CPU_minus_GPU_unprojected_TD_RMS=None,
                conditional_expectile_residual_identified=False, actual_training_minibatch_reconstructed=False)
            row['CPU_minus_GPU_unprojected_TD_RMS'] = row['current_V_unprojected_snapshot_TD_RMS'] - reference['td_rms']
            rows.append(row)
            for name, value in dict(Q=q, target_Q=target_q, V=v, raw_y=raw, projected_y=projected).items():
                arrays[f'{short}_{step}_{name}'] = value
            evidence.append(dict(run=str(run), step=step, checkpoint_sha256=sha(checkpoint),
                config_sha256=sha(run / 'config.json'), dataset_sha256=sha(dataset_path)))
        arrays[short + '_indices'] = indices
        env.close()
    np.savez_compressed(args.output / 'raw_snapshots.npz', **arrays)
    result = dict(time=time.time(), status='complete', source_sha256=sha(__file__),
        plan_sha256=sha(plan_path), rows=rows, model_evidence=evidence,
        note='CPU current-network snapshot formulas on the exact private fixed replay indices; not the original GPU training minibatch or conditional population residual.',
        limits=['CPU/GPU and differently shaped compiled graphs may differ numerically; raw differences are retained without attributing a GPU precision fault.',
            'Generic learned-V td_rms omits projection. Finite-root branches additionally use a different actual bootstrap variable.',
            'Sample-wise score RMS is not RMS of a conditional expected score; true zeta/B_k remain unidentified.'],
        goal_complete=False, primary_scores_unchanged=True)
    (args.output / 'review.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(status='complete', models=len(evidence), rows=rows)), flush=True)


if __name__ == '__main__':
    main()
