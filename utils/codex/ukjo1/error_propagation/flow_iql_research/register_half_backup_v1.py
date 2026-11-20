"""Register three uniform half-correction pilots after final-source validation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    unit_path = root/'analysis/half_backup_validation_v1/unit_final.json'
    runner_path = root/'analysis/half_backup_runner_validation_v2/runner.json'
    unit, runner = [json.loads(path.read_text()) for path in (unit_path, runner_path)]
    assert unit['serialized_two_plus_two_full_state_max_abs'] == 0
    assert max(unit['endpoint_full_state_max_abs'].values()) == 0
    assert runner['source_pinned_payload_and_replay_RNG_full_vs_resumed_exact']
    config = json.loads((runner_path.parent/'full/config.json').read_text())
    pins = config['source_sha256']
    for path, digest in pins.items(): assert sha(path) == digest, path
    jobs = []
    for short, environment in [('ant', 'antmaze-large-navigate-singletask-v0'),
                                ('cubed', 'cube-double-play-singletask-v0'),
                                ('pz3', 'puzzle-3x3-play-singletask-v0')]:
        jobs.append(dict(id=f'heb1_{short}_bounded_v2_half_empirical_backup_g999_t90_s0',
                         module='utils.flow_iql_research.train_half_empirical_backup_v1',
                         args={'env':environment, 'kind':'ifql', 'profile':'bounded_v2_half_empirical_backup',
                               'seed':0, 'discount':.999, 'expectile':.9, 'samples':9,
                               'steps':1000000, 'episodes':50}))
    plan = dict(time=time.time(), kind='uniform_half_correction_pilot_registration_before_GPU_launch',
                correction_fraction=.5, backup_samples=9, deployment_samples=9, expectile=.9, discount=.999,
                method='IQLV2 plus half displacement toward exact finite-target-minQ expectile; original IFQL BC policy',
                method_note='utils/flow_iql_research/HALF_EMPIRICAL_BACKUP_METHOD.txt',
                primary_steps=[800000, 900000, 1000000], episodes=50, first_episode_seed=42,
                final_metric='Equal last3 mean; pilots cannot form a nine-environment mean.',
                jobs=jobs, source_sha256=pins,
                validation={str(path):sha(path) for path in (unit_path, runner_path)},
                validation_amendment='Prior tiny CPU v1 validation rejected resume after a probe-source amendment; retained and superseded by frozen final-source v2. No GPU benchmark had launched.',
                scope='Alternative value-only intervention; not combined with GMM, rank-flow or target-head mean. No environment-specific correction tuning.',
                limits=['Finite proposal loss/score reduction is not an oracle residual guarantee.',
                        'Learned-value and finite-generator errors may cancel; midpoint is not proven optimal.',
                        'Nine-task expansion and fresh training-seed confirmation are required for a benchmark claim.'])
    plan_path = root/'half_empirical_backup_pilot_registration_v1.json'
    assert not plan_path.exists()
    lock = (root/'queue_mutation.lock').open('a')
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue_path = root/'queue.json'
        queue = json.loads(queue_path.read_text())
        assert not ({job['id'] for job in jobs} & {job['id'] for job in queue})
        priority = json.loads((root/'policy_family_round_robin_priority_v2.json').read_text())['priority']
        priority = priority[:8]+[jobs[0]['id'], priority[8], jobs[1]['id'], priority[9], priority[10], jobs[2]['id'], priority[11]]
        mapping = {job['id']:job for job in queue+jobs}
        ordered = [mapping[key] for key in priority]+[job for job in queue if job['id'] not in priority]
        assert len(ordered) == len(queue)+3 and len({job['id'] for job in ordered}) == len(ordered)
        previous = root/'queue_before_half_backup_registration_v1.json'
        assert not previous.exists()
        previous.write_bytes(queue_path.read_bytes())
        plan['pending_priority'] = priority
        plan['queue_before_sha256'] = sha(queue_path)
        plan_path.write_text(json.dumps(plan, indent=2))
        temp = queue_path.with_suffix('.half_backup.tmp')
        temp.write_text(json.dumps(ordered, indent=2))
        os.replace(temp, queue_path)
    print(json.dumps(dict(plan=str(plan_path), registered_jobs=[job['id'] for job in jobs],
                          queue_size=len(ordered), source_pins_verified=True)), flush=True)


if __name__ == '__main__':
    main()
