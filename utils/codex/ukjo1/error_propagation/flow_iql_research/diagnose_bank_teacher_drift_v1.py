"""Read-only same-state/same-noise teacher drift across joint checkpoints.

Cross old/new flow banks with old/new critics. Changes are descriptive, not
an additive causal decomposition. All states are offline replay probe states.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--steps', default='100000,300000,800000')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0)) == 6
    assert all(d.platform == 'cpu' for d in jax.devices())
    steps = sorted({int(x) for x in args.steps.split(',')})
    conf = json.loads((args.run/'config.json').read_text())
    identity = conf['identity']
    assert identity['recipe'] == 'bank_bc'
    for path, digest in conf['source_sha256'].items():
        assert sha(path) == digest, path
    cfg = dict(identity['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    env, data, _ = ogbench.make_env_and_datasets(identity['environment'])
    env.close()
    indices = np.random.RandomState(identity['seed']+19401).randint(len(data['actions']),size=1024)
    obs = jnp.asarray(np.asarray(data['observations'][indices],np.float32))
    dim = data['actions'].shape[-1]
    del data
    template = flat.create(identity['seed'],obs[:1],jnp.zeros((1,dim)),cfg,identity['recipe'])
    states, hashes = {}, {}
    for step in steps:
        cp = args.run/f'checkpoint_{step}.pkl'
        hashes[str(step)] = sha(cp)
        with cp.open('rb') as f:
            saved = pickle.load(f)
        assert saved['step'] == step and saved['identity'] == identity
        assert saved['source_sha256'] == conf['source_sha256']
        state = flax.serialization.from_state_dict(template,saved['state'])
        assert int(state.base.q_updates) == int(state.actor_updates) == step
        assert int(state.base.v_updates) == 0
        assert int(state.actor.opt_state[0].count) == int(state.base.agent.network.opt_state[0].count) == step
        states[step] = state

    @jax.jit
    def generate(state):
        agent = state.base.agent
        broad,noise = flat.lean.bank(agent,obs,jax.random.PRNGKey(74191),9)
        actions = flat.lean.flow(agent,broad,noise,10)
        q = agent.network.select('critic')(broad,actions=actions).min(0)
        best = q.argmax(0)
        actor = jnp.clip(state.actor(obs,flat.packed_input(noise,cfg['onestep_specification'])),-1,1)
        return dict(bank=actions,q=q,index=best,teacher=flat.gather(actions,best),actor=actor)

    @jax.jit
    def score(state,actions):
        broad = jnp.broadcast_to(obs,(9,*obs.shape))
        return state.base.agent.network.select('critic')(broad,actions=actions).min(0)

    generated = {step:jax.device_get(generate(states[step])) for step in steps}
    mse = lambda x,y:float(np.mean((np.asarray(x)-np.asarray(y))**2))
    summaries = {}
    for step,g in generated.items():
        gap = np.sort(g['q'],axis=0)[-1]-np.sort(g['q'],axis=0)[-2]
        summaries[str(step)] = dict(actor_to_current_teacher_MSE=mse(g['actor'],g['teacher']),
            teacher_best_vs_second_Q_gap_quantiles=np.quantile(gap,[0,.1,.5,.9,1]).tolist(),
            same_state_bank_action_variance=float(g['bank'].var(0).mean()))
    comparisons = []
    for i,old in enumerate(steps[:-1]):
        for new in steps[i+1:]:
            a,b = generated[old],generated[new]
            q_new_oldbank = np.asarray(score(states[new],jnp.asarray(a['bank'])))
            q_old_newbank = np.asarray(score(states[old],jnp.asarray(b['bank'])))
            h_oldflow_newcritic = np.asarray(flat.gather(jnp.asarray(a['bank']),jnp.asarray(q_new_oldbank.argmax(0))))
            h_newflow_oldcritic = np.asarray(flat.gather(jnp.asarray(b['bank']),jnp.asarray(q_old_newbank.argmax(0))))
            comparisons.append(dict(old_step=old,new_step=new,
                same_noise_bank_flow_action_MSE=mse(a['bank'],b['bank']),
                actual_teacher_label_drift_MSE=mse(a['teacher'],b['teacher']),
                critic_only_reselection_on_oldbank_MSE=mse(a['teacher'],h_oldflow_newcritic),
                flow_change_with_oldcritic_MSE=mse(a['teacher'],h_newflow_oldcritic),
                critic_argmax_flip_fraction_on_oldbank=float(np.mean(a['index']!=q_new_oldbank.argmax(0))),
                critic_argmax_flip_fraction_on_newbank=float(np.mean(b['index']!=q_old_newbank.argmax(0))),
                new_actor_to_old_teacher_MSE=mse(b['actor'],a['teacher']),
                new_actor_to_current_teacher_MSE=mse(b['actor'],b['teacher']),
                old_actor_to_current_teacher_MSE=mse(a['actor'],b['teacher'])))
    args.output.mkdir(parents=True)
    arrays = dict(observations=np.asarray(obs),replay_indices=indices)
    for step,g in generated.items():
        arrays.update({f'{key}_{step}':value for key,value in g.items()})
    raw = args.output/'same_state_noise_drift_arrays_v1.npz'
    np.savez_compressed(raw,**arrays)
    result = dict(time=time.time(),status='complete',run=str(args.run),environment=identity['environment'],
        training_seed=identity['seed'],steps=steps,checkpoint_sha256=hashes,
        config_sha256=sha(args.run/'config.json'),replay_probe_states=1024,
        state_indices_seed='training_seed+19401',noise_seed=74191,noise_count=9,
        states_and_noise_fixed_across_checkpoints=True,actor_input='same packed9Gaussian as teacher',
        optimizer_updates_performed=0,evaluation_interactions=0,no_online_data=True,
        summaries=summaries,comparisons=comparisons,arrays_sha256=sha(raw),
        source_sha256={str(Path(p).resolve()):sha(p) for p in [__file__,*flat.source_paths()]},
        limitation='CPU same-replay-probe-state diagnostics; finite checkpoints, not fine-scale instantaneous '
                   'teacher drift, policy-return or gradient-error oracle. Cross interventions are not additive '
                   'or unique causal attributions; frozen/joint comparisons also confound budget, data, and inputs.',
        scope='Read-only description of teacher nonstationarity and actor tracking. Not evaluation, training, '
              'a success gate, or proof that nonstationarity helps.')
    for step,h in hashes.items():
        assert sha(args.run/f'checkpoint_{step}.pkl') == h
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(summaries=summaries,comparisons=comparisons)),flush=True)


if __name__ == '__main__':
    main()
