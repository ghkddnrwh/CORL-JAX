"""Failure-triggered BC/critic source crossover; frozen inference only.

Two existing diagonal evaluations plus two source swaps form a2x2 per task.
No source combination can become a benchmark winner using these observations.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from utils.flow_iql_research import core
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
import numpy as np


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def audit_job(plan, job, verifier):
    result = json.loads(Path(job['output']).read_text())
    step = job['step']
    assert result['environment'] == job['environment']
    assert result['step'] == result['learner_update_step'] == step
    assert result['aggregation_rule'] == 'mean_rank'
    assert result['selection_samples'] == 9 and result['num_independent_learners'] == 3
    assert result['num_Q_heads'] == 6
    assert result['total_Q_updates_across_members'] == 3*step
    assert result['total_V_updates_across_members'] == 6*step
    assert result['n_episodes'] == 50 and result['first_episode_seed'] == 42
    assert result['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
    assert result['eval_device'] == 'cpu'
    assert not result['primary_evaluations_modified'] and not result['mixed_joint_learning_state_saved']
    for field in ('episode_successes','episode_rewards','episode_lengths'):
        assert len(result[field]) == 50 and np.isfinite(result[field]).all()
    assert set(result['episode_successes']) <= {0.,1.}
    assert abs(statistics.mean(result['episode_successes'])-result['success_rate']) < 1e-12
    assert result['selector_source_sha256'] == plan['source_sha256']['utils/flow_iql_research/independent_critic_policy_v2.py']
    assert result['evaluator_source_sha256'] == plan['source_sha256']['utils/flow_iql_research/evaluate_independent_critic_policy_v2.py']
    configurations = []
    assert len(result['critic_members']) == len(job['member_pins']) == 3
    assert len({m['training_seed'] for m in result['critic_members']}) == 3
    for run_name, pin, evidence in zip(job['critic_runs'], job['member_pins'], result['critic_members']):
        run = Path(run_name)
        cfg, digest = audit_model(run, step, verifier)
        assert cfg['identity']['environment'] == job['environment']
        configurations.append({k:v for k,v in cfg['identity']['config'].items() if k!='num_samples'})
        assert evidence['run'] == run.name and evidence['training_seed'] == cfg['identity']['seed']
        assert evidence['actual_Q_Adam_count'] == step and evidence['actual_V_Adam_count'] == 2*step
        assert evidence['checkpoint_sha256'] == pin['checkpoint_sha256'] == digest
        assert evidence['config_sha256'] == pin['config_sha256'] == verifier.digest(run/'config.json')
        assert Path(evidence['checkpoint']).resolve() == (run/f'checkpoint_{step}.pkl').resolve()
    actor = Path(job['actor_run'])
    cfg, digest = audit_model(actor, step, verifier)
    assert cfg['identity']['environment'] == job['environment']
    assert result['actor_checkpoint_sha256'] == job['actor_pin']['checkpoint_sha256'] == digest
    assert result['actor_training_seed'] == cfg['identity']['seed']
    assert verifier.digest(actor/'config.json') == job['actor_pin']['config_sha256']
    configurations.append({k:v for k,v in cfg['identity']['config'].items() if k!='num_samples'})
    assert all(c == configurations[0] for c in configurations)
    return dict(success=result['success_rate'], result_sha256=verifier.digest(job['output']))


def audit_model(run, step, verifier):
    config = json.loads((run/'config.json').read_text())
    identity = config['identity']; cfg = identity['config']
    assert identity['kind'] == 'ifql' and identity['profile'] == 'bounded_v2'
    assert cfg['discount'] == .999 and cfg['expectile'] == .9 and cfg['num_samples'] in (9,32)
    assert identity['steps'] == 1000000 and identity['episodes'] == 50
    assert not (run/'stopped.json').exists()
    done = json.loads((run/'completed.json').read_text())
    assert done['step'] == done['q_updates'] == 1000000 and done['v_updates'] == 2000000
    for path, digest in config['source_sha256'].items():
        assert verifier.digest(path) == digest, path
    return config, verifier.checkpoint(run/f'checkpoint_{step}.pkl', config, step)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('registration', type=Path)
    args = parser.parse_args(); plan = json.loads(args.registration.read_text())
    out = Path(plan['output']); out.mkdir(exist_ok=False)
    lock = (out/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    verifier = Verifier()
    for path, digest in plan['source_sha256'].items():
        assert verifier.digest(path) == digest, path
    queue = [j for j in plan['jobs'] if j['new_evaluation']]
    assert len(queue) == 12 and len(plan['jobs']) == 24
    finished, running, failures = {}, {}, []
    for job in plan['jobs']:
        if not job['new_evaluation']:
            finished[job['id']] = audit_job(plan, job, verifier)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl')
    slots = plan['CPUslots']
    while queue or running:
        for key in list(running):
            worker = running[key]
            if worker['process'].poll() is None:
                continue
            worker['log'].close()
            if worker['process'].returncode:
                failures.append(dict(job=key, exit_code=worker['process'].returncode))
            else:
                finished[key] = audit_job(plan, worker['job'], verifier)
            del running[key]
        for slot in range(len(slots)):
            if not queue or failures or any(w['slot'] == slot for w in running.values()):
                continue
            job = queue.pop(0)
            for path, digest in plan['source_sha256'].items():
                assert verifier.digest(path) == digest, path
            command = ['taskset','-c',','.join(map(str,slots[slot])),sys.executable,'-u','-m',
                'utils.flow_iql_research.evaluate_independent_critic_policy_v2',
                '--actor-run',job['actor_run'],'--critic-runs',*job['critic_runs'],
                '--step',str(job['step']),'--rule','mean_rank','--output',job['output']]
            log = (out/(job['id']+'.log')).open('w')
            process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            running[job['id']] = dict(process=process, log=log, slot=slot, job=job,
                identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19])
        atomic(out/'status.json', dict(time=time.time(), pid=os.getpid(),
            status='failure_requires_inspection' if failures else 'running' if queue or running else 'complete',
            completed=finished, pending=len(queue), failures=failures,
            running={k:dict(pid=w['process'].pid, process_identity=w['identity'], slot=w['slot']) for k,w in running.items()},
            goal_complete=False, primary_candidate_unchanged=True))
        if failures and not running:
            return
        if queue or running:
            time.sleep(5)
    cells=[]
    for task in plan['tasks']:
        for actor_group in ('seen','fresh'):
            for critic_group in ('seen','fresh'):
                jobs = [j for j in plan['jobs'] if j['task']==task and j['actor_group']==actor_group and j['critic_group']==critic_group]
                assert len(jobs) == 3 and sorted(j['step'] for j in jobs) == [800000,900000,1000000]
                cells.append(dict(task=task, actor_group=actor_group, critic_group=critic_group,
                    per_step={str(j['step']):finished[j['id']]['success'] for j in jobs},
                    last3_mean=statistics.mean(finished[j['id']]['success'] for j in jobs)))
    atomic(out/'summary.json', dict(time=time.time(), status='complete', cells=cells,
        registration_sha256=verifier.digest(args.registration), source_sha256=verifier.digest(__file__),
        actual_unique_models_verified=len(verifier.checkpoints), evaluations=finished,
        additional_GPU_jobs=0, additional_CPU50_evaluations=12, primary_candidate_unchanged=True,
        limits=plan['limits'], goal_complete=False))


if __name__ == '__main__':
    main()
