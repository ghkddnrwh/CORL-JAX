"""Supplemental real-observation resident and host-to-host policy latency.

Does not replace the original frozen zero-input benchmark or its gates.
Both policies use identical real observations, random keys, devices, and
precision flags. Models are resident; host-to-host includes float32 casting,
input transfer, policy, synchronization and a copied NumPy action. Loading,
compilation, PRNG-key preparation, simulator stepping and IO are excluded.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import statistics
import time
import numpy as np
from utils.flow_iql_research import mixture_onestep_deploy_core_v1 as core
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as portable
from utils.flow_iql_research.routed_policy_verifier_v1 import actor_path
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import flax.serialization
import jax
import jax.numpy as jnp


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def host_action(policy, weights, observation, key, device):
    observation = jax.device_put(np.asarray(observation, np.float32), device)
    return np.asarray(policy(weights, observation, key)).copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--CPU-validation-only', action='store_true')
    a = p.parse_args()
    assert not a.output.exists()
    if not a.CPU_validation_only:
        assert os.environ['NVIDIA_TF32_OVERRIDE'] == '0'
    manifest = json.loads(a.observations.read_text())
    assert manifest['status'] == 'pass' and len(manifest['environments']) == 9
    observed = {row['environment']: row for row in manifest['environments']}
    jobs = json.loads(a.input.read_text())['jobs']
    result = dict(time=time.time(), status='running', rows=[], source_sha256=sha(__file__),
        input_sha256=sha(a.input), observations_manifest_sha256=sha(a.observations),
        visible_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
        CPU_affinity=sorted(os.sched_getaffinity(0)), NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        protocol='Supplemental64realresetobservations,batch1,10warm+3x100synchronouscalls,alternatingmodeandinput-placementorder,residentweights.',
        host_to_host_includes=['float32_observation_cast', 'device_put', 'single_policy_call',
                               'device_synchronization', 'copied_numpy_action'],
        excluded=['model_loading', 'JIT_compilation', 'PRNG_key_preparation', 'environment_step', 'IO'],
        original_primary_speed_gates_changed=False, validation_only=a.CPU_validation_only)
    if not a.CPU_validation_only:
        result['GPU_kind'] = jax.devices('gpu')[0].device_kind
    for job in jobs:
        run = Path(job['run'])
        root = run.parent.parent
        conf = json.loads((run / 'config.json').read_text())
        identity = conf['identity']
        assert identity['recipe'] == 'mdn9_bc' and identity['seed'] == 0
        obs_spec = observed[identity['environment']]
        observation_path = Path(obs_spec['output'])
        assert sha(observation_path) == obs_spec['sha256']
        observations = np.load(observation_path, allow_pickle=False)
        native_obs = observations['observations_native']
        obs32 = observations['observations_float32']
        assert native_obs.shape == obs32.shape and native_obs.shape[0] == 64
        assert np.array_equal(native_obs.astype(np.float32), obs32)
        path = actor_path(root, run.name, 1000000)
        checkpoint = run / 'checkpoint_1000000.pkl'
        native = run / 'actor_1000000.pkl'
        actor_digest, checkpoint_digest, native_digest = sha(path), sha(checkpoint), sha(native)
        with path.open('rb') as stream:
            artifact = pickle.load(stream)
        with checkpoint.open('rb') as stream:
            saved = pickle.load(stream)
        assert artifact['training_identity'] == saved['identity'] == identity
        assert artifact['training_checkpoint_sha256'] == checkpoint_digest
        assert artifact['native_actor_sha256'] == native_digest
        assert artifact['training_step'] == saved['step'] == 1000000
        for source, digest in artifact['deployment_source_sha256'].items():
            assert sha(source) == digest
        for source, digest in conf['source_sha256'].items():
            assert sha(source) == digest
        metadata = artifact['metadata']
        assert obs32.shape[1] == metadata['observation_dim']
        assert obs_spec['action_dim'] == metadata['action_dim']
        cfg = dict(identity['config'])
        for key in ['actor_hidden_dims', 'value_hidden_dims']:
            cfg[key] = tuple(cfg[key])
        template = core.create(identity['seed'], jnp.zeros((1, metadata['observation_dim'])),
            jnp.zeros((1, metadata['action_dim'])), cfg, 'mdn9_bc')
        state = flax.serialization.from_state_dict(template, saved['state'])
        assert int(state.base.q_updates) == int(state.actor_updates) == 1000000
        assert int(state.base.v_updates) == 0
        row = dict(run=run.name, environment=identity['environment'], training_seed=0,
            checkpoint_sha256=checkpoint_digest, actor_artifact_sha256=actor_digest,
            native_actor_artifact_sha256=native_digest, observations_sha256=obs_spec['sha256'],
            native_observation_dtype=str(native_obs.dtype), metadata=metadata)
        backends = ['cpu'] if a.CPU_validation_only else ['cpu', 'gpu']
        for backend in backends:
            device = jax.devices(backend)[0]
            device_obs = [jax.device_put(x, device) for x in obs32]
            keys = [jax.device_put(x, device) for x in jax.random.split(jax.random.PRNGKey(733), 310)]
            policies = dict(primary=(portable.make_policy(metadata, backend), artifact['actor_params']),
                teacher=(core.make_teacher_policy(state, backend), core.lean.policy_parameters(state.base)))
            for mode, (policy, weights) in policies.items():
                weights = jax.device_put(weights, device)
                policies[mode] = (policy, weights)
                # The new timing wrapper must preserve each frozen policy's
                # action exactly at all64 observation/key pairs before timing.
                for index in range(64):
                    reference = np.asarray(policy(weights, device_obs[index], keys[index]))
                    wrapped = host_action(policy, weights, native_obs[index], keys[index], device)
                    assert np.isfinite(reference).all() and np.array_equal(reference, wrapped)
                    assert reference.shape == (metadata['action_dim'],)
                for index in range(10):
                    policy(weights, device_obs[index], keys[index]).block_until_ready()
            row[backend + '_all64_actions_wrapper_bitwise_exact'] = True
            if a.CPU_validation_only:
                continue
            records = {(scope, mode): [] for scope in ['resident', 'host_to_host'] for mode in policies}
            for repeat in range(3):
                modes = ['primary', 'teacher'] if repeat % 2 == 0 else ['teacher', 'primary']
                scopes = ['resident', 'host_to_host'] if repeat % 2 == 0 else ['host_to_host', 'resident']
                for scope in scopes:
                    for mode in modes:
                        policy, weights = policies[mode]
                        values = []
                        for index in range(10 + repeat * 100, 110 + repeat * 100):
                            observation = device_obs[index % 64] if scope == 'resident' else native_obs[index % 64]
                            key = keys[index]
                            before = time.perf_counter()
                            if scope == 'resident':
                                action = policy(weights, observation, key)
                                action.block_until_ready()
                            else:
                                action = host_action(policy, weights, observation, key, device)
                            values.append((time.perf_counter() - before) * 1000)
                        records[(scope, mode)].append(values)
            for (scope, mode), values in records.items():
                prefix = f'{backend}_{scope}_{mode}'
                medians = [statistics.median(v) for v in values]
                row[prefix + '_ms_median'] = statistics.median(medians)
                row[prefix + '_ms_repeat_medians'] = medians
                row[prefix + '_ms_p90'] = float(np.quantile(np.asarray(values).ravel(), .9))
                row[prefix + '_all_ms'] = values
            for scope in ['resident', 'host_to_host']:
                prefix = f'{backend}_{scope}'
                row[prefix + '_latency_ratio'] = row[prefix + '_primary_ms_median'] / row[prefix + '_teacher_ms_median']
        assert sha(path) == actor_digest and sha(checkpoint) == checkpoint_digest and sha(native) == native_digest
        result['rows'].append(row)
        atomic(a.output, result)
        print(json.dumps(dict(run=run.name, validation_only=a.CPU_validation_only,
            wrapper_actions_exact=True)), flush=True)
        del state, template, saved, artifact, policies, policy, weights, keys, observations, device_obs
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
        gc.collect()
    result.update(status='complete', time=time.time(), optimizer_updates_performed=0)
    atomic(a.output, result)


if __name__ == '__main__':
    main()
