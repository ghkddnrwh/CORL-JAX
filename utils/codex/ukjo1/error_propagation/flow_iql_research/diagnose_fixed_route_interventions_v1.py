"""Diagnostic rollout interventions: learned/exact routing and generation.

The exact route is ONLY the frozen fixed32 learned-critic teacher's index,
not an oracle for true returns. Hybrid policies intentionally use Flow/Q;
they are not fast deployment candidates and cannot count toward the goal.
No optimizer, evaluation-state training, or online data improvement occurs.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import routed_fixed_actor_v1 as routed
from utils.flow_iql_research.evaluate_fixed_bank_teacher_v1 import make_policy as reference_policy
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
from utils.propagation_research.seeded_reset import prepare_seeded_action_space

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def write(path,value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    temp.replace(path)


def make_policy(template,definition,bank,mode):
    assert mode in ('learned_route_exact_flow','teacher_route_learned_head','both_exact','both_learned')

    def policy(params,obs,key):
        agent = template.base.agent.replace(network=template.base.agent.network.replace(
            params=params['teacher']['base']))
        if mode == 'both_learned':
            return definition.apply({'params':params['actor']},obs)
        p = definition.apply({'params':params['actor']},obs,return_components=True)
        if mode == 'learned_route_exact_flow':
            route = p['logits'].argmax(-1)
            return flat.lean.flow(agent,obs,bank[route],10)
        broad = jnp.broadcast_to(obs,(32,*obs.shape))
        actions = flat.lean.flow(agent,broad,bank,10)
        scores = agent.network.select('critic')(broad,actions=actions).min(0)
        route = scores.argmax(0)
        if mode == 'both_exact':
            return actions[route]
        return jnp.clip(p['means'][route],-1,1)

    return jax.jit(policy,backend='cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0)) == 6
    assert all(d.platform == 'cpu' for d in jax.devices())
    manifest = json.loads(args.manifest.read_text())
    for p,h in manifest['source_sha256'].items():
        assert sha(p) == h,p
    case = manifest['cases'][args.case]
    checkpoint = Path(case['teacher_checkpoint'])
    assert sha(checkpoint) == case['teacher_checkpoint_sha256']
    conf = json.loads((checkpoint.parent/'config.json').read_text())
    for p,h in conf['source_sha256'].items():
        assert sha(p) == h,p
    cfg = dict(conf['identity']['config'])
    for k in ('actor_hidden_dims','value_hidden_dims'):
        cfg[k] = tuple(cfg[k])
    assert sha(case['bank']) == case['bank_sha256']
    bank = jnp.asarray(np.load(case['bank']))
    assert sha(case['replay_cache']) == case['replay_cache_sha256']
    with np.load(case['replay_cache']) as archive:
        preflight_observations = archive['observations'][-4096:-4092].copy()
    obs_dim,dim = case['observation_dim'],bank.shape[-1]
    template = flat.create(conf['identity']['seed'],jnp.zeros((1,obs_dim)),jnp.zeros((1,dim)),cfg,conf['identity']['recipe'])
    with checkpoint.open('rb') as f:
        saved = pickle.load(f)
    assert saved['step'] == 1000000 and saved['identity'] == conf['identity']
    state = flax.serialization.from_state_dict(template,saved['state'])
    assert int(state.base.q_updates) == int(state.actor_updates) == 1000000
    assert int(state.base.v_updates) == 0
    teacher_params = flat.lean.policy_parameters(state.base)
    args.output.mkdir(parents=True)
    result = dict(time=time.time(),status='evaluating',environment=case['environment'],
        manifest_sha256=sha(args.manifest),teacher_checkpoint_sha256=sha(checkpoint),
        fixed_bank_sha256=sha(case['bank']),optimizer_updates=0,evaluation_used_for_learning=False,
        interventions=['learned_route_exact_flow','teacher_route_learned_head'],
        episodes_per_intervention=50,episode_seeds=list(range(42,92)),models={},
        source_sha256=manifest['source_sha256'],
        interpretation='Exact means frozen learned-critic fixed32 teacher index or its BC Flow generator; '
                       'not a true-value oracle. Joint errors and state visitation interact; no additive causal decomposition.',
        scope='Private diagnostic hybrid rollouts with Flow/Q. NOT fast actor-only deployment or native goal performance.')
    write(args.output/'status.json',result)
    for name,model in case['models'].items():
        assert sha(model['artifact']) == model['artifact_sha256']
        with Path(model['artifact']).open('rb') as f:
            actor = pickle.load(f)
        metadata = actor['metadata']
        assert metadata['components'] == 32 and metadata['architecture'] == 'state_only_route_then_one_linear_head'
        definition = routed.RoutedFixedActor(tuple(metadata['hidden_dims']),dim,32)
        params = dict(teacher=teacher_params,actor=actor['actor_params'])
        payload = jax.device_put(params,jax.devices('cpu')[0])
        policies = {mode:make_policy(state,definition,bank,mode) for mode in
                    (*result['interventions'],'both_exact','both_learned')}
        exact_reference = reference_policy(state,bank)
        primary_reference = routed.make_policy(metadata)
        # Same single-state shapes as rollout, including nonzero states.
        preflight = []
        for observation in [np.zeros(obs_dim,np.float32),*preflight_observations]:
            key = jax.random.PRNGKey(42)
            both = np.asarray(policies['both_exact'](payload,observation,key))
            ref = np.asarray(exact_reference(teacher_params,observation,key))
            learned = np.asarray(policies['both_learned'](payload,observation,key))
            primary = np.asarray(primary_reference(actor['actor_params'],observation,key))
            err1,err2 = float(np.max(np.abs(both-ref))),float(np.max(np.abs(learned-primary)))
            assert err1 < 1e-6 and err2 < 1e-6
            assert all(np.isfinite(np.asarray(policies[mode](payload,observation,key))).all() for mode in result['interventions'])
            preflight.append(dict(exact_teacher_absmax=err1,primary_actor_absmax=err2))
        result['models'][name] = dict(artifact_sha256=model['artifact_sha256'],
            original_actor_CPU50=model['original_actor_CPU50'],
            original_fixed32_teacher_CPU50=case['teacher_CPU50'],preflight=preflight,evaluations={})
        write(args.output/'status.json',result)
        for mode in result['interventions']:
            namespace = SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=params)))
            env = ogbench.make_env_and_datasets(case['environment'],env_only=True)
            reset = prepare_seeded_action_space(env,42)
            evaluated = evaluate(namespace,env,policies[mode],100000,50)
            env.close()
            evaluated.update(time=time.time(),protocol='private_fixed_route_factorial_CPU50_v1',
                mode=mode,model=name,teacher_checkpoint_sha256=sha(checkpoint),
                actor_artifact_sha256=model['artifact_sha256'],reset_metadata=reset,
                no_optimizer_updates=True,evaluation_used_for_learning=False,counts_toward_goal=False,
                not_actor_only_deployment=True)
            path = args.output/(name+'_'+mode+'_CPU50_v1.json')
            write(path,evaluated)
            result['models'][name]['evaluations'][mode] = dict(success=evaluated['success_rate'],path=str(path),sha256=sha(path))
            write(args.output/'status.json',result)
            print(json.dumps(dict(model=name,mode=mode,success=evaluated['success_rate'])),flush=True)
    assert sha(checkpoint) == case['teacher_checkpoint_sha256']
    result.update(status='complete',completed=time.time(),total_evaluation_episodes=200)
    write(args.output/'result.json',result)
    write(args.output/'status.json',result)


if __name__ == '__main__':
    main()
