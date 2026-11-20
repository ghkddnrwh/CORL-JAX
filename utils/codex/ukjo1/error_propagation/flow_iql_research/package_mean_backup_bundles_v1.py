"""Package predefined seed9/1M policies as parents finish; no model selection."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from utils.flow_iql_research import tau_matched_extraction_v1 as helper
from utils.flow_iql_research.mean_backup_bundle_v1 import sha,RECIPE


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path);args=parser.parse_args()
    plan=json.loads(args.registration.read_text());output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    lock=(output/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=Path(plan['root']);running={};complete={};failures=[];pins={}
    parent_plan=json.loads(Path(plan['parent_registration']).read_text())
    assert sha(plan['parent_registration'])==plan['parent_registration_sha256']
    assert len(plan['jobs'])==len({j['environment']for j in plan['jobs']})==9
    assert all(j['seed']==9 and j['step']==1000000 for j in plan['jobs'])
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        MUJOCO_GL='egl',PYTHONUNBUFFERED='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',XLA_PYTHON_CLIENT_MEM_FRACTION='0.1')
    def parent(job):
        rid=job['run_id'];run=root/'runs'/rid
        if not(run/'completed.json').exists():return False
        if rid not in pins:
            done=json.loads((run/'completed.json').read_text());assert done['step']==done['q_updates']==1000000 and done['v_updates']==2000000
            assert not(run/'stopped.json').exists()
            _,_,path,digest=helper.load_parent(parent_plan,rid,1000000)
            pins[rid]=dict(path=str(path),sha256=digest)
        return True
    for job in plan['jobs']:
        prior=plan['reused_development'].get(job['id'])
        if prior is None:continue
        assert parent(job)
        for path,digest in prior['file_sha256'].items():assert sha(path)==digest,path
        result=json.loads(Path(prior['evaluation']).read_text())
        assert result['environment']==job['environment']and result['training_seed']==9 and result['step']==1000000
        assert all(result['reference_episode_arrays_exact'].values())and result['no_source_directory_reads_during_load_or_rollout']
        assert result['no_optimizer_updates']and result['source_checkpoint_sha256']==pins[job['run_id']]['sha256']
        shutil.copy2(prior['bundle'],job['bundle']);assert sha(job['bundle'])==result['bundle_sha256']
        complete[job['id']]=dict(environment=job['environment'],seed=9,step=1000000,bundle=job['bundle'],
            bundle_sha256=sha(job['bundle']),bytes=Path(job['bundle']).stat().st_size,
            reference_episode_arrays_exact=result['reference_episode_arrays_exact'],evaluation=prior['evaluation'],
            evaluation_sha256=sha(prior['evaluation']),reused_development=True,source_checkpoint_sha256=pins[job['run_id']]['sha256'])
    def launch(job,stage,slot):
        path=output/(job['id']+'_'+stage+'.log');log=path.open('a')
        command=['nice','-n','10','taskset','-c',','.join(map(str,plan['CPUslots'][slot])),sys.executable,'-u','-m']
        if stage=='export':command+=['utils.flow_iql_research.export_mean_backup_bundle_v1',str(root/'runs'/job['run_id']),'--output',job['bundle']]
        else:command+=['utils.flow_iql_research.evaluate_mean_backup_bundle_v2',job['bundle'],'--reference-run',str(root/'runs'/job['run_id']),'--output',job['evaluation']]
        process=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        running[job['id']]=dict(process=process,log=log,job=job,stage=stage,slot=slot,
            process_identity=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19])
    while True:
        helper.check_sources(plan['source_sha256'])
        for key,row in list(running.items()):
            code=row['process'].poll()
            if code is None:continue
            row['log'].close();del running[key];job=row['job']
            if code:failures.append(dict(job=key,stage=row['stage'],exit_code=code));continue
            if row['stage']=='export':
                with Path(job['bundle']).open('rb')as stream:bundle=pickle.load(stream)
                assert bundle['recipe']==RECIPE and bundle['training_seed']==9 and bundle['step']==1000000
                assert bundle['environment']==job['environment']and bundle['identity']==parent_plan['expected_identities'][job['run_id']]
                assert bundle['provenance']['checkpoint_sha256']==pins[job['run_id']]['sha256']
                launch(job,'evaluate',row['slot'])
            else:
                result=json.loads(Path(job['evaluation']).read_text());helper.check_episodes(result,1000000)
                assert result['environment']==job['environment']and result['training_seed']==9
                assert result['source_checkpoint_sha256']==pins[job['run_id']]['sha256']
                assert all(result['reference_episode_arrays_exact'].values())and result['no_source_directory_reads_during_load_or_rollout']
                assert result['no_optimizer_updates']and result['bundle_sha256']==sha(job['bundle'])
                complete[key]=dict(environment=job['environment'],seed=9,step=1000000,bundle=job['bundle'],bundle_sha256=sha(job['bundle']),
                    bytes=Path(job['bundle']).stat().st_size,reference_episode_arrays_exact=result['reference_episode_arrays_exact'],
                    evaluation=job['evaluation'],evaluation_sha256=sha(job['evaluation']),reused_development=False,
                    source_checkpoint_sha256=result['source_checkpoint_sha256'])
        occupied={r['slot']for r in running.values()}
        if not failures:
            ready=[j for j in plan['jobs']if j['id']not in complete and j['id']not in running and parent(j)]
            for slot in range(len(plan['CPUslots'])):
                if slot not in occupied and ready:launch(ready.pop(0),'export',slot)
        finished=len(complete)==9
        result=dict(time=time.time(),pid=os.getpid(),status='failure_requires_inspection'if failures else'complete'if finished else'running',
            completed_tasks=len(complete),artifacts=list(complete.values()),failures=failures,actual_parent_model_pins=pins,
            running={k:dict(pid=v['process'].pid,process_identity=v['process_identity'],stage=v['stage'],slot=v['slot'])for k,v in running.items()},
            registration_sha256=sha(args.registration),source_sha256=sha(__file__),new_GPU_jobs=0,new_training_replication=False,
            primary_metric_modified=False,goal_complete=False,
            scope='Nine predefined seed9/fixed1M deployment policies. CPU50 replays verify serialization, not additional benchmark observations or last3 averages.')
        helper.atomic(output/'status.json',result)
        if finished:helper.atomic(output/'manifest.json',result);return
        if failures and not running:return
        time.sleep(10)


if __name__=='__main__':main()
