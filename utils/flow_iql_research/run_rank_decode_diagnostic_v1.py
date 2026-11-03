"""Run a registered three-grid frozen-model CPU diagnostic in two slots."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());out=Path(plan['output_directory']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for source,digest in plan['source_sha256'].items():assert sha(source)==digest,source
    assert sha(Path(plan['run'])/'config.json')==plan['config_sha256']
    assert sha(Path(plan['run'])/f"checkpoint_{plan['step']}.pkl")==plan['checkpoint_sha256']
    running={};failures={};slots=['60-63','64-67']
    while True:
        for name,item in list(running.items()):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();del running[name]
            if code!=0:failures[name]=dict(exit_code=code,time=time.time())
            print('FINISH',name,code,flush=True)
        valid={};pending=[]
        for job in plan['jobs']:
            path=Path(job['output'])
            if path.exists():
                result=json.loads(path.read_text())
                assert result['checkpoint_sha256']==plan['checkpoint_sha256']
                assert result['config_sha256']==plan['config_sha256']
                assert result['step']==plan['step']and result['decode_steps']==job['decode_steps']
                assert result['n_episodes']==50 and result['first_episode_seed']==42 and result['eval_device']=='cpu'
                assert result['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
                assert len(result['episode_successes'])==50
                assert abs(float(np.mean(result['episode_successes']))-result['success_rate'])<1e-12
                assert result['decoder_source_sha256']==plan['source_sha256']['utils/flow_iql_research/rank_decode_policy_v1.py']
                assert result['evaluator_source_sha256']==plan['source_sha256']['utils/flow_iql_research/evaluate_rank_decode_v1.py']
                valid[job['id']]=result
            elif job['id']not in running and job['id']not in failures:pending.append(job)
        occupied={item['slot']for item in running.values()}
        for slot,cpus in enumerate(slots):
            if slot in occupied or not pending:continue
            job=pending.pop(0);path=Path(job['output'])
            command=['taskset','-c',cpus,sys.executable,'-u','-m','utils.flow_iql_research.evaluate_rank_decode_v1',plan['run'],'--step',str(plan['step']),'--decode-steps',str(job['decode_steps']),'--episodes','50','--queries',str(plan['queries']),'--samples',str(plan['samples']),'--output',str(path)]
            if job['decode_steps']==10:command+=['--reference',str(Path(plan['run'])/'evaluations.jsonl')]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
            log=path.with_suffix('.log').open('a');process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running[job['id']]=dict(process=process,log=log,slot=slot,started_at=time.time());print('START',job['id'],process.pid,flush=True)
        atomic(out/'status.json',dict(pid=os.getpid(),time=time.time(),running={name:dict(pid=item['process'].pid,slot=item['slot'],started_at=item['started_at'])for name,item in running.items()},completed=len(valid),pending=len(pending),failures=failures))
        if not running and(len(valid)==len(plan['jobs'])or failures):
            if failures:return
            ten=next(value for value in valid.values()if value['decode_steps']==10)
            assert all(ten['self_reference_episode_arrays_exact'].values())
            teacher=json.loads(Path(plan['teacher_result']).read_text());assert teacher['checkpoint_sha256']==plan['checkpoint_sha256']
            rows=[];teacher_arrays=[]
            for job in plan['jobs']:
                value=valid[job['id']]
                with np.load(Path(job['output']).with_suffix('.npz'))as data:teacher_arrays.append(data['teacher_actions'].copy())
                rows.append(dict(decode_steps=job['decode_steps'],CPU50_success=value['success_rate'],reward_mean=value['reward_mean'],summary=value['summary'],
                    only_ten_succeeds=int(np.sum((np.asarray(ten['episode_successes'])==1)&(np.asarray(value['episode_successes'])==0))),
                    only_this_grid_succeeds=int(np.sum((np.asarray(ten['episode_successes'])==0)&(np.asarray(value['episode_successes'])==1))),
                    output_sha256=sha(job['output'])))
            atomic(out/'summary.json',dict(time=time.time(),run=plan['run'],step=plan['step'],rows=rows,
                own_fixed_ten_step_BC_best9_CPU50=teacher['success_rate'],
                teacher_probe_action_max_delta_between_graphs=max(float(np.max(np.abs(array-teacher_arrays[0])))for array in teacher_arrays),
                actual50_ten_step_episode_arrays_exact=True,plan_sha256=sha(args.plan),
                scope='Posthocfrozen100Kdiagnostic,notlast3,newtrainingreplicateorwinningdecodeselection.',goal_complete=False))
            return
        time.sleep(15)


if __name__=='__main__':main()
