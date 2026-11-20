"""Equal-Adam-budget frozen-Q V fitting: repeated versus fresh minibatches.

Auxiliary CPU diagnosis only. Neither procedure resumes coupled Q training or
changes a checkpoint. Heldout losses are not conditional residual oracle labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core, policy_core
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
import optax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--step', type=int, default=1000000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    verifier = Verifier(); run = args.run.resolve()
    config = json.loads((run/'config.json').read_text()); identity = config['identity']; cfg = dict(identity['config'])
    assert identity['kind'] == 'ifql' and identity['profile'] == 'bounded_v2'
    assert cfg['discount'] == .999 and cfg['expectile'] == .9
    for path, digest in config['source_sha256'].items():
        assert verifier.digest(path) == digest, path
    cp = run/f'checkpoint_{args.step}.pkl'; cp_sha = verifier.checkpoint(cp, config, args.step)
    with cp.open('rb') as stream:
        saved = pickle.load(stream)
    for key in ('actor_hidden_dims','value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    env, raw, val = ogbench.make_env_and_datasets(identity['environment']); env.close()
    template = policy_core.create(identity['seed'],jnp.zeros((1,raw['observations'].shape[-1])),
        jnp.zeros((1,raw['actions'].shape[-1])),cfg,'ifql')
    state = flax.serialization.from_state_dict(template,saved['state']); net = state.agent.network
    rng = np.random.RandomState(identity['seed']+87111)
    index_pins = {}
    fields = ('observations','actions','next_observations','rewards','masks')
    def sample(data, count, tag):
        indices = rng.randint(len(data['actions']),size=count)
        index_pins[tag] = hashlib.sha256(indices.tobytes()).hexdigest()
        batch = {k:np.asarray(data[k][indices],np.float32) for k in fields}
        batch['actions'] = np.clip(batch['actions'],-1+1e-5,1-1e-5)
        return {k:jnp.asarray(v) for k,v in batch.items()}
    fitting = sample(raw,256,'initial_fitting')
    sets = dict(same_minibatch=fitting, independent_train=sample(raw,4096,'independent_train'),
                validation=sample(val,4096,'validation'))
    fresh_batches = [fitting]+[sample(raw,256,f'fresh_fitting_{i}') for i in range(1,8)]
    @jax.jit
    def targetq(batch):
        return net.select('target_critic')(batch['observations'],actions=batch['actions']).min(0)
    targets = {k:targetq(b) for k,b in sets.items()}
    fresh_targets = [targetq(b) for b in fresh_batches]
    @jax.jit
    def per_example(vp, batch, q):
        v = net.select('value')(batch['observations'],params={**net.params,'modules_value':vp})
        diff = q-v
        return jnp.where(diff>=0,cfg['expectile'],1-cfg['expectile'])*diff**2
    tx = optax.adam(cfg['lr'])
    @jax.jit
    def fit(vp, vo, batch, q):
        grad = jax.grad(lambda p:per_example(p,batch,q).mean())(vp)
        updates,vo = tx.update(grad,vo,vp)
        return optax.apply_updates(vp,updates),vo
    initial_vp, initial_vo = net.params['modules_value'], state.v_opt
    assert int(initial_vo[0].count) == 2*args.step
    baseline = {k:np.asarray(per_example(initial_vp,b,targets[k]),np.float64) for k,b in sets.items()}
    traces = {}
    for mode in ('repeated_minibatch','fresh_minibatch'):
        vp,vo = initial_vp,initial_vo; trace=[]
        for n in range(9):
            if n in (0,1,2,4,8):
                measurements={}
                for name,batch in sets.items():
                    loss = np.asarray(per_example(vp,batch,targets[name]),np.float64)
                    assert np.isfinite(loss).all()
                    measurements[name] = dict(expectile_objective=float(loss.mean()),
                        objective_change_from_snapshot=float((loss-baseline[name]).mean()))
                trace.append(dict(extra_V_Adam_steps=n,sets=measurements,
                    actual_V_Adam_count=int(vo[0].count)))
                assert int(vo[0].count) == 2*args.step+n
            if n < 8:
                batch,q = (fitting,targets['same_minibatch']) if mode=='repeated_minibatch' else (fresh_batches[n],fresh_targets[n])
                vp,vo = fit(vp,vo,batch,q)
        traces[mode] = trace
    assert traces['repeated_minibatch'][:2] == traces['fresh_minibatch'][:2]
    prior_path = run.parent.parent/'analysis/frozen_Q_V_diagnostics_v1'/f'{run.name}_{args.step}.json'
    prior_agreement = None
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        assert prior['checkpoint_sha256'] == cp_sha and prior['config_sha256'] == verifier.digest(run/'config.json')
        assert prior['diagnostic_source_sha256'] == verifier.digest('utils/flow_iql_research/frozen_diagnostics_v3.py')
        discrepancies=[]
        for old,new in zip(prior['trace'],traces['repeated_minibatch']):
            assert old['extra_frozen_v_steps'] == new['extra_V_Adam_steps']
            for name in sets:
                a,b = old['sets'][name]['expectile_objective'],new['sets'][name]['expectile_objective']
                discrepancies.append(abs(a-b))
                assert np.isclose(a,b,rtol=1e-6,atol=1e-7),(name,a,b)
        prior_agreement=dict(max_objective_abs_difference=max(discrepancies),prior_sha256=verifier.digest(prior_path))
    assert int(state.v_opt[0].count) == 2*args.step
    result = dict(time=time.time(),kind='frozen_Q_value_sampling_equal_optimizer_budget_CPU_diagnostic',
        run=run.name,environment=identity['environment'],learner_seed=identity['seed'],step=args.step,
        modes=traces,query_index_sha256=index_pins,prior_repeated_fit_agreement=prior_agreement,
        checkpoint_sha256=cp_sha,config_sha256=verifier.digest(run/'config.json'),
        source_sha256=verifier.digest(__file__),initial_V_Adam_count=2*args.step,
        primary_training_modified=False,benchmark_evaluations_added=0,GPU_training_added=0,
        limits=['Frozen target-Q is not the coupled Q/V training trajectory.',
            'One-action-per-state heldout expectile objective is not an oracle conditional residual or B_k.',
            'Both modes take8Adamsteps of256examples; repeated reuses256draws and fresh uses2048draws including the same first batch.',
            'Fitting/train/validation query sets are identical within a model; queries depend on learner seed and are not shared across seeds.',
            'CPU arithmetic does not replay GPU updates exactly; no new trained model or policy return is produced.'],goal_complete=False)
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(args.output)
    print(json.dumps(dict(run=run.name,at8={m:t[-1]['sets'] for m,t in traces.items()},prior_agreement=prior_agreement)),flush=True)


if __name__=='__main__':
    main()
