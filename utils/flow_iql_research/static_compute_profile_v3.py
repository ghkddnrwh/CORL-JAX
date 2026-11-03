"""CPU HLO operation estimates; no learning, GPU work or policy episodes.

The operation estimates describe compiled single updates and single-observation
actions. They are neither measured latency nor evidence of equal GPU compute.
"""
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import time

from utils.flow_iql_research import core, benchmark_core, policy_core, selection_v2
from utils.flow_iql_research.independent_critic_policy_v2 import make_selector
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def size(tree):
    return sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(tree))


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def estimate(function, arguments, path):
    compiled = jax.jit(function, backend='cpu').lower(*arguments).compile()
    analyses = compiled.cost_analysis()
    assert len(analyses) == 1, analyses
    costs = {key: float(value) for key, value in analyses[0].items()
             if key in ('flops', 'transcendentals', 'bytes accessed')}
    assert costs['flops'] > 0
    hlo = compiled.as_text().encode()
    with gzip.open(path, 'wb') as stream:
        stream.write(hlo)
    result = dict(costs=costs, optimized_HLO_gzip=str(path),
                  optimized_HLO_sha256=hashlib.sha256(hlo).hexdigest())
    del compiled
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert all(device.platform == 'cpu' for device in jax.devices())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = args.root.resolve()
    plan_path = root / 'independent_critic_fresh_confirmation_registration_v1.json'
    plan = json.loads(plan_path.read_text())
    source_paths = [Path(__file__), Path(core.__file__), Path(policy_core.__file__),
                    Path(benchmark_core.__file__), Path(selection_v2.__file__),
                    Path('utils/flow_iql_research/independent_critic_policy_v2.py'),
                    Path('utils/flow_iql_research/independent_critic_policy_v1.py')]
    source_paths += sorted(Path(core.__file__).parent.joinpath('vendor').glob('*.py'))
    result = dict(time=time.time(), status='running', rows=[],
                  plan_sha256=sha(plan_path), source_sha256={str(p): sha(p) for p in source_paths},
                  jax_version=jax.__version__, backend='cpu', training_batch_size=256,
                  training_updates_per_member=1000000,
                  excluded=['environment interaction', 'evaluation episodes', 'data transfer and I/O',
                            'periodic probes', 'checkpoint writes', 'compilation', 'GPU latency'],
                  interpretation='Compiler static operation estimates; bytes accessed are compiler estimates, not measured memory traffic.',
                  goal_complete=False)
    for short, environment in zip(('ant','hmmed','hmlarge','soccer','cube','cubed','pz3','pz4','scene'), core.ENVIRONMENTS):
        identity = next(item for item in plan['expected_identities'].values()
                        if item['environment'] == environment and item['seed'] == 3)
        cfg = dict(identity['config'])
        fql_path = root / f'runs/ref_{short}_fql_g999_s0/config.json'
        fql_config = json.loads(fql_path.read_text())
        for path, digest in fql_config['source_sha256'].items():
            assert sha(path) == digest, path
        for path, digest in plan['training_source_sha256'].items():
            assert sha(path) == digest, path
        fc = dict(fql_config['identity']['config'])
        for configuration in (cfg, fc):
            for key in ('actor_hidden_dims', 'value_hidden_dims'):
                configuration[key] = tuple(configuration[key])
        env = ogbench.make_env_and_datasets(environment, env_only=True)
        ob_shape, act_shape = env.observation_space.shape, env.action_space.shape
        env.close()
        observation = jnp.zeros((1, *ob_shape), jnp.float32)
        action = jnp.zeros((1, *act_shape), jnp.float32)
        state = policy_core.create(3, observation, action, cfg, 'ifql')
        fql_state = benchmark_core.create(0, observation, action, fc, 'fql')
        batch = dict(observations=jnp.zeros((256, *ob_shape), jnp.float32),
                     next_observations=jnp.zeros((256, *ob_shape), jnp.float32),
                     actions=jnp.zeros((256, *act_shape), jnp.float32),
                     rewards=jnp.zeros((256,), jnp.float32), masks=jnp.ones((256,), jnp.float32))
        base_path = root / f'runs/p1_{short}_ifql_c0_bounded_v2_g999_t90_s0/dataset.json'
        data = json.loads(base_path.read_text())
        bounds = (data['return_lower'], data['return_upper'])
        row = dict(task=short, environment=environment, observation_shape=ob_shape,
                   action_shape=act_shape, IFQL_config=cfg, FQL_config=fc,
                   FQL_config_sha256=sha(fql_path), bounds_metadata_sha256=sha(base_path))
        def profile(name, function, arguments):
            return estimate(function, arguments, out / f'{short}_{name}.hlo.gz')
        row['IFQL_V2_single_update'] = profile('IFQL_update',
            policy_core.build_update('bounded_v2', bounds, 'ifql'), (state, batch))
        row['FQL_single_update'] = profile('FQL_update',
            benchmark_core.build_update('official', bounds, 'fql'), (fql_state, batch))
        params = state.agent.network.params
        key, obs = jax.random.PRNGKey(42), observation[0]
        for n in (9, 32):
            row[f'IFQL_best{n}_one_action'] = profile(f'IFQL_best{n}',
                selection_v2.make_policy(state.agent, 'max', n, 'behavior'), (params, obs, key))
        selector = make_selector(state.agent, 'mean_rank', 9)
        row['rank_mean_M3_N9_one_action'] = profile('rank_mean_M3_N9',
            lambda p, members, o, k: selector(p, members, o, k)[0],
            (params, tuple(params['modules_critic'] for _ in range(3)), obs, key))
        def fql_policy(p, o, k):
            agent = fql_state.agent.replace(network=fql_state.agent.network.replace(params=p))
            return agent.sample_actions(o, seed=k, temperature=0.)
        row['FQL_one_action'] = profile('FQL_action', fql_policy,
                                      (fql_state.agent.network.params, obs, key))
        count = lambda name: size(params['modules_' + name])
        row['parameters'] = dict(IFQL_single_trainable=sum(count(n) for n in ('critic','value','actor_flow')),
            primary_M3_all_trainable=3*sum(count(n) for n in ('critic','value','actor_flow')),
            primary_semantic_deployment=3*count('critic')+count('actor_flow'),
            FQL_trainable=sum(size(v) for k,v in fql_state.agent.network.params.items() if k!='modules_target_critic'),
            FQL_semantic_deployment=size(fql_state.agent.network.params['modules_actor_onestep_flow']))
        row['ratios'] = dict(
            M3_training_FLOPs_over_FQL=3*row['IFQL_V2_single_update']['costs']['flops']/row['FQL_single_update']['costs']['flops'],
            primary_action_FLOPs_over_FQL=row['rank_mean_M3_N9_one_action']['costs']['flops']/row['FQL_one_action']['costs']['flops'],
            primary_action_FLOPs_over_IFQL32=row['rank_mean_M3_N9_one_action']['costs']['flops']/row['IFQL_best32_one_action']['costs']['flops'])
        result['rows'].append(row)
        atomic(out / 'summary.json', result)
        print(json.dumps(dict(task=short, ratios=row['ratios'])), flush=True)
        del state, fql_state, params, batch, selector
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
        gc.collect()
    result['status'] = 'complete'
    result['equal_update_nine_task_training_FLOP_ratio'] = (
        sum(3*r['IFQL_V2_single_update']['costs']['flops'] for r in result['rows']) /
        sum(r['FQL_single_update']['costs']['flops'] for r in result['rows']))
    atomic(out / 'summary.json', result)


if __name__ == '__main__':
    main()
