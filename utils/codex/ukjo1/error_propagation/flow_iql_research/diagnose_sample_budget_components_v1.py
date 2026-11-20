"""Frozen Q/BC-Flow crossover for seed/time instability; no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(output):
    root = PROJECT / 'logs/diagnosis/finite_mean_sample_budget_multiseed_20261115'
    prior_path = root / 'registration_v1.json'
    prior = json.loads(prior_path.read_text())
    hm = 'humanoidmaze-medium-navigate-singletask-v0'
    pz = 'puzzle-4x4-play-singletask-v0'
    specs = {
        'hm18_s1_800K': (hm, 18, 1, 800000),
        'hm18_s1_1M': (hm, 18, 1, 1000000),
        'hm18_s2_800K': (hm, 18, 2, 800000),
        'hm18_s2_1M': (hm, 18, 2, 1000000),
        'hm36_s1_1M': (hm, 36, 1, 1000000),
        'hm36_s2_1M': (hm, 36, 2, 1000000),
        'pz27_s0_1M': (pz, 27, 0, 1000000),
        'pz27_s1_1M': (pz, 27, 1, 1000000),
        'pz27_s2_1M': (pz, 27, 2, 1000000),
    }
    pairs = [
        dict(id='hm18_s1_time', A='hm18_s1_800K', B='hm18_s1_1M'),
        dict(id='hm18_s2_time', A='hm18_s2_800K', B='hm18_s2_1M'),
        dict(id='hm36_seed', A='hm36_s1_1M', B='hm36_s2_1M'),
        dict(id='pz27_seed01', A='pz27_s0_1M', B='pz27_s1_1M'),
        dict(id='pz27_seed02', A='pz27_s0_1M', B='pz27_s2_1M'),
    ]
    sources, hashes = {}, {}
    for label, (env, n, seed, step) in specs.items():
        job = next(j for j in prior['jobs'] + prior['reused_jobs']
                   if (j['environment'], j['bootstrap_samples'], j['seed']) == (env, n, seed))
        run = Path(job['output'])
        conf_path = run / 'config.json'
        conf = json.loads(conf_path.read_text())
        cfg = conf['identity']['config']
        assert cfg['backup_samples'] == n and cfg['num_samples'] == 9
        assert cfg['discount'] == .999 and cfg['expectile'] == .9
        assert cfg['backup_Q_aggregation'] == 'mean' and cfg['target_clipping']
        for source, digest in conf['source_sha256'].items():
            assert sha(source) == digest, source
            hashes[source] = digest
        directory = Path(job['evaluation_directory']) if job.get('reused') else root / 'evaluations' / job['id']
        ref = directory / f'{step}_lean.json'
        value = json.loads(ref.read_text())
        cp = run / f'checkpoint_{step}.pkl'
        assert sha(cp) == value['checkpoint_sha256'] and sha(conf_path) == value['config_sha256']
        sources[label] = dict(run=str(run.resolve()), step=step, environment=env,
            training_seed=seed, bootstrap_samples=n, checkpoint=str(cp.resolve()),
            checkpoint_sha256=sha(cp), config_sha256=sha(conf_path),
            reference=str(ref.resolve()), reference_sha256=sha(ref),
            reference_success_rate=value['success_rate'])
    for path in (Path(__file__), Path(__file__).with_name('run_sample_budget_component_diagnosis_v1.py')):
        hashes[str(path.resolve())] = sha(path)
    jobs = [dict(id='self_' + label, q=label, flow=label, self_control=True) for label in sources]
    for pair in pairs:
        for q, flow in ((pair['A'], pair['B']), (pair['B'], pair['A'])):
            jobs.append(dict(id=f'{pair["id"]}__Q_{q}__Flow_{flow}', q=q, flow=flow, self_control=False))
    assert len(jobs) == 19
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(protocol='frozen_sample_budget_component_crossover_v1', created_unix=time.time(),
        parent_registration_sha256=sha(prior_path), sources=sources, pairs=pairs, jobs=jobs,
        source_sha256=hashes, episodes_per_cell=50, reset_seeds=list(range(42, 92)),
        deployment_candidates=9, flow_steps=10, critic_selection='min_online_twin_Q_then_argmax',
        optimizer_updates=0, new_training_runs=0, persist_mixed_training_state=False,
        trajectory_observation_stride=25,
        predeclared_followup='Compare fixed states/noise/candidate banks across the same Q/Flow sources; '
            'use replay and actual self-policy visitation states. Separate state-independent Q offsets from action rankings.',
        limits=['Cases were selected after observing failures; diagnostic, not confirmatory benchmark or a selected new algorithm.',
                'Component interventions can interact; no unique additive causal attribution is assumed.',
                'Only 1M snapshots for cross-seed pairs; not full learning curves or independent extra training seeds.',
                'Q-source intervention includes its training history, initialization and replay/noise randomness.'])
    atomic(output / 'plan_v1.json', plan)
    print(json.dumps(dict(output=str(output), jobs=len(jobs), episodes=950, pairs=len(pairs))), flush=True)


def read_source(source, restore=True):
    import flax.serialization
    import jax
    import jax.numpy as jnp
    import numpy as np
    from utils.flow_iql_research import sample_budget_core_v1 as core
    run = Path(source['run'])
    conf = json.loads((run / 'config.json').read_text())
    assert sha(run / 'config.json') == source['config_sha256']
    assert sha(source['checkpoint']) == source['checkpoint_sha256']
    with Path(source['checkpoint']).open('rb') as stream:
        saved = pickle.load(stream)
    assert saved['identity'] == conf['identity'] and saved['source_sha256'] == conf['source_sha256']
    assert saved['step'] == source['step']
    state = saved['state']
    assert int(state['base']['q_updates']) == int(state['actor_updates']) == source['step']
    assert int(state['base']['v_updates']) == 0
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(state))
    if not restore:
        return conf, state
    identity = conf['identity']
    cfg = dict(identity['config'])
    for key in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[key] = tuple(cfg[key])
    dataset = json.loads((run / 'dataset.json').read_text())
    template = core.create(identity['seed'], jnp.zeros((1, dataset['observation_dim'])),
        jnp.zeros((1, dataset['action_dim'])), cfg, identity['recipe'])
    return conf, flax.serialization.from_state_dict(template, state)


class ObservedEnv:
    """Record observations without drawing randomness or altering transitions."""
    def __init__(self, env, stride):
        self.env, self.stride = env, stride
        self.episode, self.steps = -1, 0
        self.observations, self.coordinates = [], []

    def __getattr__(self, key):
        return getattr(self.env, key)

    def record(self, obs):
        import numpy as np
        self.observations.append(np.asarray(obs, np.float32).copy())
        self.coordinates.append((self.episode, self.steps))

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self.episode += 1
        self.steps = 0
        self.record(result[0])
        if self.episode % 10 == 0:
            print(json.dumps(dict(event='episode', episode=self.episode)), flush=True)
        return result

    def step(self, action):
        result = self.env.step(action)
        self.steps += 1
        if self.steps % self.stride == 0 and not (result[2] or result[3]):
            self.record(result[0])
        return result


def evaluate_job(plan_path, job_id):
    import jax
    import numpy as np
    import ogbench
    from utils.flow_iql_research import sample_budget_core_v1 as core
    from utils.flow_iql_research.train_mean_empirical_backup_v1 import evaluate
    from utils.propagation_research.seeded_reset import prepare_seeded_action_space
    assert all(d.platform == 'cpu' for d in jax.devices())
    plan = json.loads(plan_path.read_text())
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest, path
    job = next(j for j in plan['jobs'] if j['id'] == job_id)
    output = plan_path.parent / 'evaluations' / f'{job_id}.json'
    assert not output.exists()
    qs, fs = plan['sources'][job['q']], plan['sources'][job['flow']]
    qc, qstate = read_source(qs)
    fc, fstate = (qc, qstate.base.agent.network.params['modules_actor_flow']) if qs == fs else read_source(fs, False)
    if qs != fs:
        fstate = fstate['base']['agent']['network']['params']['modules_actor_flow']
    assert qs['environment'] == fs['environment']
    for key in ('actor_hidden_dims', 'value_hidden_dims', 'actor_layer_norm', 'layer_norm', 'num_samples', 'flow_steps'):
        assert qc['identity']['config'][key] == fc['identity']['config'][key]
    payload = core.lean.policy_parameters(qstate.base)
    params = dict(payload['base'])
    params['modules_actor_flow'] = fstate
    payload = dict(payload, base=params)
    policy = core.make_teacher_policy(qstate)
    namespace = SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=payload)))
    env = ogbench.make_env_and_datasets(qs['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, 42)
    wrapped = ObservedEnv(env, plan['trajectory_observation_stride'])
    result = evaluate(namespace, wrapped, policy, qs['step'], plan['episodes_per_cell'])
    env.close()
    comparison = None
    if job['self_control']:
        reference = json.loads(Path(qs['reference']).read_text())
        assert sha(qs['reference']) == qs['reference_sha256']
        comparison = {key: result[key] == reference[key] for key in
                      ('episode_successes', 'episode_rewards', 'episode_lengths')}
        assert all(comparison.values()), comparison
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix('.npz')
    np.savez_compressed(raw_path, observations=np.stack(wrapped.observations),
                        episode_step=np.asarray(wrapped.coordinates, np.int32))
    for source in (qs, fs):
        assert sha(source['checkpoint']) == source['checkpoint_sha256']
    result.update(protocol=plan['protocol'], job=job, q_source=qs, flow_source=fs,
        reset_metadata=reset, source_sha256=plan['source_sha256'],
        self_episode_arrays_exact=comparison, optimizer_updates=0, new_training_runs=0,
        mixed_training_state_saved=False, trajectory_arrays_sha256=sha(raw_path),
        trajectory_observations=len(wrapped.observations),
        trajectory_scope='Diagnostic observations only; never appended to training data.',
        timestamp_unix=time.time())
    atomic(output, result)
    print(json.dumps(dict(event='complete', job=job_id, success=result['success_rate'],
                          seconds=result['seconds'], self_exact=comparison)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--output', type=Path, required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--plan', type=Path, required=True)
    ev.add_argument('--job', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output.resolve())
    else:
        evaluate_job(args.plan.resolve(), args.job)


if __name__ == '__main__':
    main()
