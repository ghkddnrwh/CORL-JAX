"""Frozen-Q fitting and exact success-boundary calibration, never a policy score.

Raw return predictions stay in cost units. Sigmoid-head predictions are decoded
using the registered known-bound affine map. Private V-only CPU counterfits reset
Adam, use shared fresh training minibatches, and never touch live model files.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np
from utils.flow_iql_research import tau_matched_extraction_v1 as helper

COUNTERFIT_STEPS=(2,8,32)
QUERY_ROWS=8192
QUERY_SEED=605131
TRAIN_SEED=609193


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def metrics(q,tq,v,reward,mask,tau):
    fit=tq-v;w=np.where(fit>=0,tau,1-tau)
    result=dict(rows=len(v),Q_mean=float(q.mean()),target_Q_mean=float(tq.mean()),V_mean=float(v.mean()),
        fitted_expectile_MSE=float(np.mean(w*fit**2)),sample_score_mean=float(np.mean(w*fit)),
        sample_score_RMS=float(np.sqrt(np.mean((w*fit)**2))),
        Q_twin_disagreement_RMS=float(np.sqrt(np.mean((q[0]-q[1])**2))))
    if np.all(mask==0):
        assert np.all(reward==0)
        # These are exact relabelled absorbing-boundary labels, not an oracle
        # for arbitrary nonterminal action values or conditional expectiles.
        result.update(known_boundary_Q_label=0.,
            online_Q_min_boundary_RMSE=float(np.sqrt(np.mean(q.min(0)**2))),
            online_Q_mean_boundary_RMSE=float(np.sqrt(np.mean(q.mean(0)**2))),
            target_Q_min_boundary_RMSE=float(np.sqrt(np.mean(tq**2))),
            V_boundary_RMSE=float(np.sqrt(np.mean(v**2))),
            online_Q_min_penalty_mean=float(np.mean(q.min(0)-q.mean(0))))
    return result


def run_job(registration,job,output):
    from utils.flow_iql_research import policy_core,probability_value_core_v1 as scoring
    from utils.propagation_research.reward_bounds import REWARD_COMPONENTS
    import flax.serialization
    import jax
    import jax.numpy as jnp
    import optax
    import ogbench
    assert not output.exists()
    conf,saved,checkpoint,checkpoint_sha=helper.load_parent(registration,job['run_id'],job['step'])
    cfg=dict(conf['identity']['config']);environment=conf['identity']['environment']
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    np.random.seed(0)
    env,raw,validation=ogbench.make_env_and_datasets(environment)
    fields=('observations','actions','next_observations','rewards','masks')
    train={k:np.asarray(raw[k],np.float32)for k in fields}
    val={k:np.asarray(validation[k],np.float32)for k in fields}
    train['actions']=np.clip(train['actions'],-1+1e-5,1-1e-5)
    val['actions']=np.clip(val['actions'],-1+1e-5,1-1e-5)
    index=np.random.RandomState(QUERY_SEED).choice(len(val['actions']),QUERY_ROWS,replace=False)
    goals=np.flatnonzero(val['masks']==0)
    assert not len(goals)or np.all(val['rewards'][goals]==0)
    query_index=np.r_[index,goals]
    training_index=np.random.RandomState(TRAIN_SEED).randint(len(train['actions']),size=(32,256))
    template=policy_core.create(conf['identity']['seed'],jnp.asarray(train['observations'][:1]),
        jnp.asarray(train['actions'][:1]),cfg,'ifql')
    state=flax.serialization.from_state_dict(template,saved['state']);net=state.agent.network
    unit=cfg.get('return_representation')=='unit_interval_sigmoid_of_raw_logits'
    scale=(1-cfg['discount'])/REWARD_COMPONENTS[environment]
    if unit:assert cfg['return_affine_scale']==scale

    @jax.jit
    def q_values(observations,actions):
        q=net.select('critic')(observations,actions=actions)
        tq=net.select('target_critic')(observations,actions=actions)
        fit=tq.min(0)
        if unit:return jax.nn.sigmoid(q),jax.nn.sigmoid(tq),jax.nn.sigmoid(fit)
        return q,tq,fit

    @jax.jit
    def v_values(vp,observations):
        values=net.select('value')(observations,params={**net.params,'modules_value':vp})
        return jax.nn.sigmoid(values)if unit else values

    def batched_q(observations,actions):
        all_q=[];all_tq=[];all_fit=[]
        for offset in range(0,len(observations),256):
            ob=observations[offset:offset+256];act=actions[offset:offset+256];n=len(ob)
            if n<256:ob=np.pad(ob,((0,256-n),(0,0)),mode='edge');act=np.pad(act,((0,256-n),(0,0)),mode='edge')
            q,tq,fit=jax.device_get(q_values(ob,act))
            all_q.append(q[:,:n]);all_tq.append(tq[:,:n]);all_fit.append(fit[:n])
        return np.concatenate(all_q,axis=1),np.concatenate(all_tq,axis=1),np.concatenate(all_fit)

    def batched_v(vp,observations):
        result=[]
        for offset in range(0,len(observations),256):
            ob=observations[offset:offset+256];n=len(ob)
            if n<256:ob=np.pad(ob,((0,256-n),(0,0)),mode='edge')
            result.append(np.asarray(v_values(vp,ob))[:n])
        return np.concatenate(result)

    decode=lambda value:(np.asarray(value,np.float64)-1)/scale if unit else np.asarray(value,np.float64)
    obs=val['observations'][query_index];actions=val['actions'][query_index]
    q,tq,tfit=batched_q(obs,actions)
    arrays=dict(validation_uniform_indices=index,validation_goal_indices=goals,training_indices=training_index,
        rewards=val['rewards'][query_index],masks=val['masks'][query_index],
        online_Q_cost=decode(q),target_Q_heads_cost=decode(tq),target_fit_Q_cost=decode(tfit))
    arrays['initial_V_cost']=decode(batched_v(net.params['modules_value'],obs))
    if unit:
        sampled_ob=train['observations'][training_index.reshape(-1)]
        sampled_ac=train['actions'][training_index.reshape(-1)]
        _,_,targets=batched_q(sampled_ob,sampled_ac)
        sampled_ob=sampled_ob.reshape((32,256,-1));targets=targets.reshape((32,256))
        for geometry in ('mse','bernoulli_kl'):
            vp=net.params['modules_value'];tx=optax.adam(cfg['lr']);vo=tx.init(vp)
            def loss(parameters,observations,target):
                logit=net.select('value')(observations,params={**net.params,'modules_value':parameters})
                return scoring.expectile_loss(logit,target,cfg['expectile'],geometry)
            @jax.jit
            def fit_step(parameters,optimizer,observations,target):
                gradients=jax.grad(loss)(parameters,observations,target)
                increments,optimizer=tx.update(gradients,optimizer,parameters)
                return optax.apply_updates(parameters,increments),optimizer
            for iteration in range(1,33):
                vp,vo=fit_step(vp,vo,sampled_ob[iteration-1],targets[iteration-1])
                if iteration in COUNTERFIT_STEPS:
                    arrays[f'counterfit_{geometry}_{iteration}_V_cost']=decode(batched_v(vp,obs))
                    assert int(vo[0].count)==iteration
    for name,array in arrays.items():assert np.isfinite(array).all(),name
    strata={'uniform_validation':slice(0,QUERY_ROWS)}
    if len(goals):strata['all_validation_success_boundary']=slice(QUERY_ROWS,None)
    report={}
    for stratum,sl in strata.items():
        target=arrays['target_fit_Q_cost'][sl];q=arrays['online_Q_cost'][:,sl]
        reward=arrays['rewards'][sl];mask=arrays['masks'][sl]
        report[stratum]={name:metrics(q,target,value[sl],reward,mask,cfg['expectile'])
            for name,value in arrays.items()if name.endswith('_V_cost')}
    assert helper.digest(checkpoint)==checkpoint_sha,'The source checkpoint changed'
    output.parent.mkdir(parents=True,exist_ok=True);npz=output.with_suffix('.npz');assert not npz.exists()
    with npz.open('wb')as stream:np.savez_compressed(stream,**arrays)
    data_metadata=dict(train_rows=len(train['actions']),validation_rows=len(val['actions']),
        full_validation_terminal_rows=len(goals),validation_field_sha256={k:array_sha(val[k])for k in fields},
        uniform_query_index_sha256=array_sha(index),training_index_sha256=array_sha(training_index),
        declared_reward_components=REWARD_COMPONENTS[environment],reward_min=float(val['rewards'].min()),
        reward_max=float(val['rewards'].max()))
    result=dict(time=time.time(),status='complete',job=job,unit_return_model=unit,
        checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,config_sha256=helper.digest(Path(registration['root'])/'runs'/job['run_id']/'config.json'),
        Q_Adam=job['step'],V_Adam=2*job['step'],data=data_metadata,statistics=report,
        raw_arrays=str(npz),raw_arrays_sha256=helper.digest(npz),source_sha256=helper.digest(__file__),
        new_GPU_training=0,new_policy_evaluations=0,goal_complete=False,
        caveats=['Posthoc diagnostic; it cannot promote methods based on validation fitting or boundary error.',
            'Only the success boundary has a known zero-value label; general nonterminal Q errors remain unknown.',
            'Fitted sample expectile losses and score moments are not conditional residual or B_k estimates.',
            'Counterfits freeze each source target critic, reset a PRIVATE V Adam, use common fresh training batches, and affect no live model.',
            'Compare objectives within one source Q. Different source models have different teachers, so their fitting losses do not prove true-Q ranking.',
            'Uniformvalidation and allterminalvalidation are separate strata; do not pool the oversampled boundary into population loss.',
            'Raw baselines receive no Bernoulli counterfit because their unbounded heads/targets have different semantics.',
            'Currentvalidation field hashes are not retrospective evidence of every historical in-memory training array.'])
    helper.atomic(output,result);env.close()
    print(json.dumps(dict(job=job['id'],terminal_rows=len(goals),unit_model=unit,complete=True)),flush=True)


def watch(plan,registration):
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    lock=(output/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    running={};valid={};failed={};metadata={}
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        MUJOCO_GL='egl',PYTHONUNBUFFERED='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
    while True:
        helper.check_sources(plan['source_sha256'])
        for key,row in list(running.items()):
            code=row['process'].poll()
            if code is not None:
                row['log'].close();del running[key]
                if code:failed[key]=dict(exit_code=code,log=str(row['path']))
        ready=[]
        for job in plan['jobs']:
            target=Path(job['output'])
            if target.exists():
                result=json.loads(target.read_text())
                assert result['job']==job and result['source_sha256']==plan['source_sha256'][__file__]
                assert helper.digest(result['raw_arrays'])==result['raw_arrays_sha256']
                assert result['Q_Adam']==job['step'] and result['V_Adam']==2*job['step']
                env=plan['expected_identities'][job['run_id']]['environment']
                if env in metadata:assert metadata[env]==result['data'],'Validation queries/data changed across models'
                metadata[env]=result['data'];valid[job['id']]=dict(output=str(target),sha256=helper.digest(target))
            elif job['id']not in running and job['id']not in failed:
                checkpoint=Path(plan['root'])/'runs'/job['run_id']/f"checkpoint_{job['step']}.pkl"
                archive=Path(plan['root'])/'analysis/checkpoint_archive'/job['run_id']/checkpoint.name
                if checkpoint.exists()or archive.exists():ready.append(job)
        occupied={r['slot']for r in running.values()}
        for slot,cpus in enumerate(plan['CPUslots']):
            if slot in occupied or not ready:continue
            job=ready.pop(0);logpath=output/(job['id']+'.log');log=logpath.open('a')
            command=['nice','-n','19','taskset','-c',','.join(map(str,cpus)),sys.executable,'-u','-m',
                'utils.flow_iql_research.unit_value_fitting_diagnostic_v1',str(registration),'--job',job['id']]
            proc=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            running[job['id']]=dict(process=proc,log=log,path=logpath,slot=slot,
                process_identity=Path(f'/proc/{proc.pid}/stat').read_text().split(') ',1)[1].split()[19])
        helper.atomic(output/'status.json',dict(time=time.time(),pid=os.getpid(),
            status='complete'if len(valid)==len(plan['jobs'])else'failures_require_inspection'if failed else'running',
            valid_jobs=len(valid),planned_jobs=len(plan['jobs']),results=valid,failures=failed,
            running={k:dict(pid=v['process'].pid,slot=v['slot'],process_identity=v['process_identity'])for k,v in running.items()},
            validation_metadata=metadata,registration_sha256=helper.digest(registration),goal_complete=False))
        if len(valid)==len(plan['jobs']):return
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path);parser.add_argument('--job')
    args=parser.parse_args();plan=json.loads(args.registration.read_text())
    if args.job:
        if 'source_sha256'in plan:helper.check_sources(plan['source_sha256'])
        job=next(j for j in plan['jobs']if j['id']==args.job)
        run_job(plan,job,Path(job['output']))
    else:watch(plan,args.registration)


if __name__=='__main__':main()
