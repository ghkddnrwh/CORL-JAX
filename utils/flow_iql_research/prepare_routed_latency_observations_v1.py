"""Freeze real reset observations for an additional deployment latency study.

CPU only; no policy, model, dataset loading, environment actions, training,
performance evaluation, or production queue changes. All nine environments
receive the same 64 independently seeded resets (42 through 105).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import ogbench
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    area = root / 'analysis/routed_real_observation_speed_v1'
    proposal = area / 'prospective_real_observation_speed_plan_v1.json'
    plan = json.loads(proposal.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest
    registration = root / 'registration_v19.json'
    assert sha(registration) == plan['registration_sha256']
    reg = json.loads(registration.read_text())
    jobs = [j for j in reg['jobs'] if j['args']['recipe'] == 'mdn9_bc' and j['args']['seed'] == 0]
    assert len(jobs) == len({j['args']['env'] for j in jobs}) == 9
    target = area / 'real_reset_observations_manifest_v1.json'
    assert not target.exists()
    rows = []
    for job in jobs:
        env_id = job['args']['env']
        output = area / (env_id + '_64_resets_v1.npz')
        assert not output.exists()
        env = ogbench.make_env_and_datasets(env_id, env_only=True)
        metadata = prepare_seeded_action_space(env, 42)
        observations = []
        for seed in range(42, 106):
            np.random.seed(seed)
            random.seed(seed)
            env.action_space.seed(seed)
            observation, _ = env.reset(seed=seed)
            observations.append(np.asarray(observation).copy())
        original = np.stack(observations)
        assert original.ndim == 2 and original.shape[0] == 64 and np.isfinite(original).all()
        values = original.astype(np.float32)
        with output.open('xb') as stream:
            np.savez(stream, observations_native=original, observations_float32=values,
                     reset_seeds=np.arange(42, 106, dtype=np.int32))
        rows.append(dict(environment=env_id, native_run=job['id'], output=str(output), sha256=sha(output),
            shape=list(original.shape), original_dtype=str(original.dtype), inference_dtype='float32',
            action_dim=int(np.prod(env.action_space.shape)), reset_metadata=metadata))
        env.close()
    result = dict(time=time.time(), status='pass', proposal_sha256=sha(proposal),
        source_sha256={str(Path(__file__).resolve()): sha(__file__)},
        environments=rows, observations_per_environment=64, first_reset_seed=42,
        last_reset_seed=105, environment_actions_executed=0, learned_model_trees_loaded=[],
        training_or_primary_scores_or_GPU_queue_modified=False,
        scope='Real environment reset observations for a future matched latency study. No episode score, optimizer update, or speed measurement.')
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status='pass', environments=len(rows), observations=64 * len(rows))))


if __name__ == '__main__':
    main()
