"""Count-aware actual-model audit and secondary high-tau extraction controller.

The registered primary N9 outcomes and training trajectories are unchanged.
All comparisons use the same retained models, late steps, and CPU50 resets.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time
import numpy as np

STEPS = (800000, 900000, 1000000)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)); temporary.replace(path)


def finite_tree(value):
    if value is None:
        return True
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    array = np.asarray(value)
    return array.dtype.kind in 'biufc' and bool(np.isfinite(array).all())


def check_sources(sources):
    for path, expected in sources.items():
        assert digest(path) == expected, path


def load_parent(plan, run_id, step):
    run = Path(plan['root'])/'runs'/run_id
    conf = json.loads((run/'config.json').read_text())
    assert conf['identity'] == plan['expected_identities'][run_id]
    assert conf['source_sha256'] == plan['expected_training_sources'][run_id]
    check_sources(conf['source_sha256'])
    path = run/f'checkpoint_{step}.pkl'
    if not path.exists():
        path = Path(plan['root'])/'analysis/checkpoint_archive'/run_id/f'checkpoint_{step}.pkl'
    before = digest(path)
    with path.open('rb') as stream:
        saved = pickle.load(stream)
    assert digest(path) == before
    assert json.loads(json.dumps(saved['identity'])) == conf['identity']
    assert saved['source_sha256'] == conf['source_sha256']
    state = saved['state']; clocks = plan['expected_final_clocks'][run_id]
    assert saved['step'] == int(state['q_updates']) == step
    assert int(state['agent']['network']['opt_state']['0']['count']) == step
    assert int(state['v_updates']) == clocks['V_total']*step//1000000
    assert int(state['v_opt']['0']['count']) == clocks['separate_V_Adam']*step//1000000
    assert finite_tree(state)
    return conf, saved, path, before


def check_episodes(result, step):
    assert result['step'] == step and result['n_episodes'] == 50
    assert result['first_episode_seed'] == 42 and result['eval_device'] == 'cpu'
    assert result['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
    for key in ('episode_successes', 'episode_rewards', 'episode_lengths'):
        assert len(result[key]) == 50 and np.isfinite(result[key]).all()
    assert set(result['episode_successes']) <= {0., 1.}
    assert abs(statistics.mean(result['episode_successes'])-result['success_rate']) < 1e-12


def primary_record(run, step):
    if not (run/'evaluations.jsonl').exists():
        return None
    records = [json.loads(line) for line in (run/'evaluations.jsonl').read_text().splitlines()]
    found = [r for r in records if r['step'] == step]
    assert len(found) <= 1
    if not found:
        return None
    check_episodes(found[0], step)
    return found[0]


def evaluate(plan, run_id, step, output):
    # Import the compatibility shim before Flax/JAX's upstream dependencies.
    from utils.flow_iql_research import policy_core, train_policy_v2, tau_matched_selection_v1
    from utils.propagation_research.seeded_reset import prepare_seeded_action_space
    import flax.serialization
    import jax
    import jax.numpy as jnp
    import ogbench
    assert not output.exists()
    check_sources(plan['source_sha256'])
    conf, saved, checkpoint, checkpoint_sha = load_parent(plan, run_id, step)
    identity = conf['identity']; cfg = dict(identity['config'])
    for field in ('actor_hidden_dims', 'value_hidden_dims'):
        cfg[field] = tuple(cfg[field])
    env = ogbench.make_env_and_datasets(identity['environment'], env_only=True)
    reset = prepare_seeded_action_space(env, 42)
    template = policy_core.create(identity['seed'], jnp.zeros((1,env.observation_space.shape[-1])),
        jnp.zeros((1,env.action_space.shape[-1])), cfg, 'ifql')
    state = flax.serialization.from_state_dict(template, saved['state'])
    compiled = tau_matched_selection_v1.make_decision(state.agent)
    totals = np.zeros(4, np.float64)

    def policy(params, observations, key):
        action, prefix_q, selected_q, index = compiled(params, observations, key)
        gap = float(selected_q-prefix_q)
        assert gap >= 0 and np.isfinite(gap)
        totals[:] += (1., gap, float(index >= 9), float(gap > 0))
        return action

    result = train_policy_v2.evaluate(state, env, policy, step, 50)
    check_episodes(result, step)
    result.update(time=time.time(), kind='secondary_tau_matched_nested_best_N',
        run_id=run_id, environment=identity['environment'], learner_seed=identity['seed'],
        expectile=cfg['expectile'], selection_samples=max(9,tau_matched_selection_v1.proposal_count(cfg['expectile'])),
        original_prefix_samples=9, extra_proposal_fold=tau_matched_selection_v1.EXTRA_PROPOSAL_FOLD,
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        config_sha256=digest(Path(plan['root'])/'runs'/run_id/'config.json'),
        source_sha256=plan['source_sha256'], reset_metadata=reset,
        rollout_query_diagnostics=dict(queries=int(totals[0]),
            predicted_Q_increase_mean=float(totals[1]/totals[0]),
            extra_proposal_selected_fraction=float(totals[2]/totals[0]),
            strictly_higher_predicted_Q_fraction=float(totals[3]/totals[0])),
        limits=['Predicted Q gains are not true action advantages or conditional residuals.',
            'Diagnostics use states visited by this secondary policy, not paired primary visited states.',
            'N99 uses more inference compute; primary training and N9 scores are unchanged.'])
    output.parent.mkdir(parents=True, exist_ok=True); atomic(output, result)
    print(json.dumps(dict(run=run_id, step=step, success=result['success_rate'])), flush=True)


def controller(plan, registration):
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=False)
    lock = (output/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root = Path(plan['root']); running, failed, valid, pins = {}, {}, {}, {}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
    while True:
        check_sources(plan['source_sha256'])
        assert digest(plan['parent_registration']) == plan['parent_registration_sha256']
        for key, row in list(running.items()):
            code = row['process'].poll()
            if code is not None:
                row['log'].close(); del running[key]
                if code:
                    failed[key] = dict(exit_code=code, log=str(row['path']))
        ready = []
        rows = []
        for run_id in plan['run_ids']:
            run = root/'runs'/run_id; paired = {}; parent_complete = False
            failure = json.loads((run/'stopped.json').read_text()) if (run/'stopped.json').exists() else None
            if (run/'completed.json').exists():
                done = json.loads((run/'completed.json').read_text())
                c = plan['expected_final_clocks'][run_id]
                assert done['step'] == done['q_updates'] == 1000000 and done['v_updates'] == c['V_total']
                parent_complete = True
            for step in STEPS:
                key = f'{run_id}_step{step}'; target = output/(key+'.json')
                primary = primary_record(run, step)
                checkpoint = run/f'checkpoint_{step}.pkl'
                if primary is not None and checkpoint.exists():
                    stat = checkpoint.stat()
                    stamp = [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]
                    if key not in pins or pins[key]['stamp'] != stamp:
                        _, _, path, cp_sha = load_parent(plan, run_id, step)
                        if key in pins:
                            assert pins[key]['sha256'] == cp_sha, 'Retained checkpoint changed'
                        pins[key] = dict(path=str(path), sha256=cp_sha, stamp=stamp)
                    if target.exists():
                        record = json.loads(target.read_text()); check_episodes(record, step)
                        assert record['run_id'] == run_id and record['selection_samples'] == 99
                        assert record['checkpoint_sha256'] == pins[key]['sha256']
                        assert record['source_sha256'] == plan['source_sha256']
                        assert record['config_sha256'] == digest(run/'config.json')
                        valid[key] = record['success_rate']
                    elif key not in running and key not in failed:
                        ready.append((key,run_id,step,target))
                paired[str(step)] = dict(primary_N9=None if primary is None else primary['success_rate'],
                    secondary_N99=valid.get(key))
            complete = parent_complete and all(v['primary_N9'] is not None and v['secondary_N99'] is not None for v in paired.values())
            rows.append(dict(run_id=run_id, environment=plan['expected_identities'][run_id]['environment'],
                profile=plan['expected_identities'][run_id]['profile'], parent_complete=parent_complete,
                stopped=failure, per_step=paired, last3_N9=statistics.mean(v['primary_N9'] for v in paired.values()) if complete else None,
                last3_N99=statistics.mean(v['secondary_N99'] for v in paired.values()) if complete else None))
        occupied = {r['slot'] for r in running.values()}
        for slot, cpus in enumerate(plan['CPUslots']):
            if slot in occupied or not ready:
                continue
            key,run_id,step,target = ready.pop(0); logpath = output/(key+'.log'); log = logpath.open('a')
            command = ['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u','-m',
                'utils.flow_iql_research.tau_matched_extraction_v1',str(registration),
                '--evaluate',run_id,'--step',str(step),'--output',str(target)]
            proc = subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running[key] = dict(process=proc,log=log,path=logpath,slot=slot,
                process_identity=Path(f'/proc/{proc.pid}/stat').read_text().split(') ',1)[1].split()[19])
        complete = all(r['last3_N99'] is not None for r in rows)
        methods = []
        for profile in sorted({r['profile'] for r in rows}):
            selected = [r for r in rows if r['profile'] == profile]
            assert len(selected) == 3
            ok = all(r['last3_N99'] is not None for r in selected)
            a = statistics.mean(r['last3_N9'] for r in selected) if ok else None
            b = statistics.mean(r['last3_N99'] for r in selected) if ok else None
            methods.append(dict(profile=profile, completed_tasks=sum(r['last3_N99'] is not None for r in selected),
                three_task_N9_mean=a,three_task_N99_mean=b,delta=None if not ok else b-a))
        atomic(output/'status.json',dict(time=time.time(),pid=os.getpid(),
            status='complete' if complete else 'failure_requires_inspection' if failed else 'running',
            rows=rows,methods=methods,valid_evaluations=len(valid),actual_checkpoint_pins=pins,
            running={k:dict(pid=r['process'].pid,slot=r['slot'],process_identity=r['process_identity']) for k,r in running.items()},
            failures=failed, registration_sha256=digest(registration), goal_complete=False,
            limits=plan['limits']))
        if complete:
            return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('registration', type=Path)
    parser.add_argument('--evaluate'); parser.add_argument('--step',type=int,choices=STEPS)
    parser.add_argument('--output',type=Path); args = parser.parse_args()
    plan = json.loads(args.registration.read_text())
    if args.evaluate:
        assert args.step and args.output
        evaluate(plan,args.evaluate,args.step,args.output)
    else:
        controller(plan,args.registration)


if __name__ == '__main__':
    main()
