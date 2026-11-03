"""Private CPU integration: actual-width training, resume, export and rollout API.

Every learner performs eight updates. These checks are not efficacy experiments
and do not modify the registered GPU queue or any protected reference model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def leaves(tree,path=''):
    if tree is None:return
    if isinstance(tree,dict):
        for key,value in sorted(tree.items()):yield from leaves(value,path+'/'+key)
    elif isinstance(tree,(list,tuple)):
        for index,value in enumerate(tree):yield from leaves(value,path+'/'+str(index))
    else:yield path,np.asarray(tree)


def invoke(directory,label,cpus,module,arguments):
    command=['nice','-n','15','taskset','-c',cpus,sys.executable,'-u','-m',module,*arguments]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',MUJOCO_GL='egl')
    with (directory/(label+'.log')).open('ab') as log:
        child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        birth=Path('/proc',str(child.pid),'stat').read_text().rsplit(') ',1)[1].split()[19]
        atomic(directory/(label+'_launch.json'),dict(time=time.time(),pid=child.pid,birth=birth,command=command))
        code=child.wait()
    assert code==0,(label,code,str(directory/(label+'.log')))


def check_case(root,environment,recipe,cpus):
    directory=root/recipe;directory.mkdir();full=directory/'full8';resume=directory/'resume4plus4'
    common=['--env',environment,'--recipe',recipe,'--seed','6','--steps','8','--chunk','2',
            '--probe-every','2','--train-device','cpu']
    runner='utils.flow_iql_research.train_set_onestep_deploy_v1'
    invoke(directory,'full8',cpus,runner,[*common,'--output',str(full)])
    invoke(directory,'part4',cpus,runner,[*common,'--output',str(resume),'--stop-at','4'])
    invoke(directory,'resume8',cpus,runner,[*common,'--output',str(resume)])
    states=[]
    for run in (full,resume):
        with (run/'checkpoint_8.pkl').open('rb') as stream:saved=pickle.load(stream)
        state=saved['state'];assert saved['step']==int(state['base']['q_updates'])==int(state['actor_updates'])==8
        assert int(state['base']['v_updates'])==0
        # Vendor TrainState starts its display step at one; Adam starts at zero.
        assert int(state['actor']['step'])==int(state['base']['agent']['network']['step'])==9
        for optimizer in (state['actor']['opt_state'],state['base']['agent']['network']['opt_state']):
            counts=[int(value) for key,value in leaves(optimizer) if key.endswith('/count')]
            assert counts and all(value==8 for value in counts),counts
        assert all(np.isfinite(value).all() for _,value in leaves(state))
        with (run/'actor_8.pkl').open('rb') as stream:artifact=pickle.load(stream)
        assert artifact['schema']=='set_actor_deployment_v1'
        assert not {'critic','base','state'}&artifact.keys()
        exported=dict(leaves(artifact['actor_params']));trained=dict(leaves(state['actor']['params']))
        assert exported.keys()==trained.keys() and all(np.array_equal(value,trained[key]) for key,value in exported.items())
        states.append(saved)
    left,right=(dict(leaves(saved['state'])) for saved in states)
    assert left.keys()==right.keys() and all(np.array_equal(value,right[key]) for key,value in left.items())
    left,right=(dict(leaves(saved['replay_rng'])) for saved in states)
    assert left.keys()==right.keys() and all(np.array_equal(value,right[key]) for key,value in left.items())
    invoke(directory,'portable',cpus,'utils.flow_iql_research.validate_standalone_set_actor_v1',
           ['--actor',str(full/'actor_8.pkl'),'--output',str(directory/'portable.json')])
    result=dict(environment=environment,recipe=recipe,full_state_resume_bitwise_equal=True,
        actor_export_params_bitwise_equal=True,Q_Adam_steps=8,actor_Adam_steps=8,V_updates=0,
        portable_client=json.loads((directory/'portable.json').read_text()),run=str(full))
    atomic(directory/'case_result.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();assert not args.output_dir.exists();args.output_dir.mkdir(parents=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cases=[('antmaze-large-navigate-singletask-v0','set_q_a1','36-41'),
           ('cube-double-play-singletask-v0','set_bc','42-47')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(check_case,args.output_dir,*case) for case in cases]
        rows=[future.result() for future in futures]
    # Separate portable clients above imported no learner or environment modules.
    # Now compare their outputs to the training definition in this CPU process.
    import jax
    from utils.flow_iql_research import set_onestep_deploy_core_v1 as core
    assert all(device.platform=='cpu' for device in jax.devices())
    for row in rows:
        run=Path(row['run'])
        with (run/'actor_8.pkl').open('rb') as stream:artifact=pickle.load(stream)
        meta=artifact['metadata'];obs=np.linspace(-.1,.1,meta['observation_dim'],dtype=np.float32)
        policy=core.make_actor_policy(meta,'cpu');single=np.asarray(policy(artifact['actor_params'],obs,jax.random.PRNGKey(42)))
        batch=np.asarray(policy(artifact['actor_params'],np.broadcast_to(obs,(16,len(obs))),jax.random.PRNGKey(43)))
        portable=row['portable_client'];assert np.array_equal(single,np.asarray(portable['action'],np.float32))
        assert np.array_equal(batch,np.asarray(portable['batch_actions'],np.float32))
        row['portable_vs_training_output_bitwise_equal']=True
    cube=next(row for row in rows if row['recipe']=='set_bc');directory=args.output_dir/'set_bc'
    for mode in ('primary','teacher'):
        output=directory/('evaluation_'+mode+'.json')
        invoke(directory,'evaluation_'+mode,'42-47','utils.flow_iql_research.evaluate_set_onestep_deploy_v1',
            ['--run',cube['run'],'--step','8','--mode',mode,'--episodes','10',
             '--implementation-only','--output',str(output)])
        result=json.loads(output.read_text());assert result['no_performance_claim'] and len(result['episode_successes'])==10
        assert result['n_episodes']==10 and result['step']==8
    result=dict(time=time.time(),status='pass',cases=rows,
        evaluator_actor_and_teacher_CPU10_API_passed=True,no_performance_claim=True,
        no_registered_queue_or_protected_model_changes=True)
    atomic(args.output_dir/'result.json',result)


if __name__=='__main__':main()
