"""Frozen single-checkpoint mean/min-head policy comparison, no training."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import policy_core as core
from utils.flow_iql_research import head_aggregation_policy_v1 as selector
from utils.flow_iql_research import train as primary
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--aggregation', choices=('min','mean'), default='mean')
    parser.add_argument('--episodes', type=int, choices=(1,50), default=50)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args(); assert not args.output.exists()
    config = json.loads((args.run/'config.json').read_text()); identity = config['identity']
    assert identity['kind'] == 'ifql' and identity['config']['num_samples'] == 9
    assert identity['profile'] in ('bounded_v2', 'bounded_v2_empirical_backup', 'bounded_v2_mean_empirical_backup')
    for path, digest in config['source_sha256'].items():
        assert sha(path) == digest, path
    checkpoint = args.run/f'checkpoint_{args.step}.pkl'
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    assert saved['step'] == int(saved['state']['q_updates']) == args.step
    assert json.loads(json.dumps(saved['identity'])) == identity
    assert saved['source_sha256'] == config['source_sha256']
    assert int(saved['state']['v_updates']) == int(saved['state']['v_opt']['0']['count']) == 2*args.step
    assert int(saved['state']['agent']['network']['opt_state']['0']['count']) == args.step
    cfg = dict(identity['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    env = ogbench.make_env_and_datasets(identity['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, 42)
    template = core.create(identity['seed'], jnp.zeros((1,env.observation_space.shape[-1])),
                           jnp.zeros((1,env.action_space.shape[-1])), cfg, 'ifql')
    state = flax.serialization.from_state_dict(template,saved['state'])
    assert all(np.isfinite(np.asarray(value)).all()for value in jax.tree_util.tree_leaves(state))
    policy = selector.make_policy(state.agent,args.aggregation)
    minimum = selector.make_policy(state.agent,'min')

    # The public IFQL sampler has a single-observation interface. Check many
    # independently keyed scalar observations, not its unsupported batch path.
    observations = np.random.RandomState(61712).normal(size=(32,env.observation_space.shape[-1])).astype(np.float32)
    def native(params,observation,key):
        model=state.agent.replace(network=state.agent.network.replace(params=params))
        return model.sample_actions(observation,key,0.)
    keys=jax.random.split(jax.random.PRNGKey(61713),32)
    native_actions=jax.jit(jax.vmap(native,in_axes=(None,0,0)),backend='cpu')(state.agent.network.params,observations,keys)
    selector_actions=jax.jit(jax.vmap(minimum,in_axes=(None,0,0)),backend='cpu')(state.agent.network.params,observations,keys)
    max_delta=float(np.max(np.abs(np.asarray(native_actions)-np.asarray(selector_actions))))
    assert max_delta <= 1e-6,max_delta
    result=primary.evaluate(state,env,policy,args.step,args.episodes)
    comparison=None
    if args.reference is not None:
        assert args.aggregation=='min'
        records=[json.loads(line)for line in args.reference.read_text().splitlines()]
        found=[record for record in records if record['step']==args.step]
        assert len(found)==1 and found[0]['n_episodes']==args.episodes
        comparison={key:result[key]==found[0][key]for key in ('episode_successes','episode_rewards','episode_lengths')}
        assert all(comparison.values()),comparison
    result.update(time=time.time(),kind='auxiliary_frozen_head_aggregation',run=args.run.name,
        environment=identity['environment'],training_seed=identity['seed'],profile=identity['profile'],
        head_aggregation=args.aggregation,selection_samples=9,critic_members=1,
        checkpoint_sha256=sha(checkpoint),config_sha256=sha(args.run/'config.json'),
        selector_source_sha256=sha(selector.__file__),evaluator_source_sha256=sha(__file__),
        evaluation_helper_source_sha256=sha(primary.__file__),reset_metadata=reset,
        min_selector_vs_native_32_scalar_probes_max_abs=max_delta,
        self_reference_episode_arrays_exact=comparison,primary_evaluations_modified=False,
        limits=['The same actual Q/BC parameters and proposal stream are used; only the deployed head reducer changes.',
                'This is an inference variant, not another training seed or an oracle Q-error correction.',
                'Early evaluations are exploratory; primary comparison requires800K/900K/1M and all nine tasks.',
                'Mean-head bootstrapping and mean-head deployment are separate experimental factors.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(args.output)
    print(json.dumps(dict(run=args.run.name,step=args.step,aggregation=args.aggregation,success=result['success_rate'],
                         native_action_max_delta=max_delta,self_reference=comparison)),flush=True)


if __name__=='__main__':main()
