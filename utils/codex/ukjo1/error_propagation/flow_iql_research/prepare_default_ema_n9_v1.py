"""Register the two-rate default-task experiment and audited existing controls."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from utils.codex.ukjo1.error_propagation.flow_iql_research import default_ema_n9_core_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research.prepare_nondefault_n9_v1 import resources, sha


HERE=Path(__file__).resolve().parent
PREFIX='utils.codex.ukjo1.error_propagation.flow_iql_research.'


def audit_reused(run, directory, method, evidence):
    conf=json.loads((run/'config.json').read_text())
    identity=conf['identity']; cfg=identity['config']
    env=identity['environment']; seed=identity['seed']
    assert env in core.ENVIRONMENTS and seed in (0,1)
    expected=core.base.config_for(env,'n9')
    rate=.005 if method=='control' else core.RATES[method]
    expected['tau']=rate
    for key,value in expected.items():
        if key=='onestep_specification':continue
        assert json.loads(json.dumps(cfg[key]))==json.loads(json.dumps(value)),(run,key)
    for key in ('target','input','objective','effective_alpha','teacher_states','q_aggregation','q_scale',
                'teacher_q_aggregation','mixture_components','std_floor','likelihood_normalization',
                'deployment_decoder'):
        assert cfg['onestep_specification'][key]==expected['onestep_specification'][key],(run,key)
    assert cfg.get('critic_batch_size',256)==256 and cfg.get('dr3_coefficient',0)==0
    archived={sha(p) for p in (run/'source').rglob('*') if p.is_file()}
    for source,digest in conf['source_sha256'].items():
        assert digest in archived,(run,source)
    done=json.loads((run/'completed.json').read_text())
    assert done['step']==done['q_updates']==done['actor_updates']==1000000 and done['v_updates']==0
    for name in ('config.json','dataset.json','completed.json'):
        path=run/name;evidence[str(path)]=sha(path)
    for step in (800000,900000,1000000):
        checkpoint=run/f'checkpoint_{step}.pkl'; actor=run/f'actor_{step}.pkl'
        evidence[str(checkpoint)]=sha(checkpoint);evidence[str(actor)]=sha(actor)
        for mode in ('lean','actor'):
            path=directory/f'{step}_{mode}.json'; result=json.loads(path.read_text())
            assert result['training_seed']==seed and result['environment']==env
            assert result['step']==step and result['policy_mode']==mode
            assert result['n_episodes']==len(result['episode_successes'])==50 and result['first_episode_seed']==42
            assert result['config_sha256']==evidence[str(run/'config.json')]
            assert result['checkpoint_sha256']==evidence[str(checkpoint)]
            assert result['actor_artifact_sha256']==evidence[str(actor)]
            evidence[str(path)]=sha(path)
    return dict(id=run.name,environment=env,family=env,task_id=core.task_info(env)['task_id'],
                code=core.task_info(env)['code'],seed=seed,recipe=identity['recipe'],method=method,
                target_EMA=rate,output=str(run),evaluation_directory=str(directory),reused=True,
                head_aggregation=cfg['backup_Q_aggregation'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1])
    parser.add_argument('--per-gpu',type=int,default=8)
    parser.add_argument('--eval-workers',type=int,default=8)
    parser.add_argument('--dataset-dir',type=Path,default=Path('~/.ogbench/data'))
    args=parser.parse_args();root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    registration=root/'registration_v1.json'
    if registration.exists():
        plan=json.loads(registration.read_text())
        for path,expected in plan['source_sha256'].items():assert sha(path)==expected,path
        print('Existing registration verified');return
    workspace=next(p for p in HERE.parents if (p/'.git').exists())
    logs=workspace/'logs/codex/ukjo1/error_propagation/diagnosis'
    canonical=logs/'finite_mean_target_ablation_20261105'
    controls=json.loads((canonical/'registration_v1.json').read_text())
    reused=[];evidence={}
    for job in controls['jobs']:
        expected='ant_min_clip' if job['environment'].startswith('antmaze-') else 'mean_clip'
        if job['recipe']!=expected or job['seed'] not in (0,1):continue
        reused.append(audit_reused(canonical/'runs'/job['id'],canonical/'evaluations'/job['id'],'control',evidence))
    assert len(reused)==20
    historical=logs/'finite_mean_stability_4task_20261120_v1'
    for run in sorted((historical/'runs').glob('*slow_target_s[01]')):
        if (run/'completed.json').exists():
            reused.append(audit_reused(run,historical/'evaluations'/run.name,'ema0p001',evidence))
    existing={(j['environment'],j['seed'],j['method']) for j in reused}
    jobs=[]
    for seed in (0,1):
        for method,rate in core.RATES.items():
            for code,env,default in core.FAMILIES:
                if (env,seed,method) in existing:continue
                token=f'em1_{code}_{method}_s{seed}'
                jobs.append(dict(id=token,code=code,environment=env,family=env,task_id=default,
                    recipe=method,method=method,target_EMA=rate,seed=seed,reused=False,
                    head_aggregation='min' if env.startswith('antmaze-') else 'mean',
                    output=str(root/'runs'/token),evaluation_directory=str(root/'evaluations'/token),
                    train_module=PREFIX+'train_default_ema_n9_v1',eval_module=PREFIX+'evaluate_default_ema_n9_v1'))
    assert len(jobs)+len(reused)==60
    sources=core.source_paths()+[HERE/name for name in (
        'prepare_default_ema_n9_v1.py','train_default_ema_n9_v1.py','evaluate_default_ema_n9_v1.py',
        'validate_default_ema_n9_v1.py','validate_default_ema_controller_v1.py',
        'run_default_ema_n9_v1.py','supervise_default_ema_n9_v1.py','analyze_default_ema_n9_v1.py',
        'prepare_nondefault_n9_v1.py','stability_study_core_v1.py','train_mean_empirical_backup_v1.py')]
    sources.append(HERE.parent/'propagation_research/seeded_reset.py')
    plan=dict(created=time.time(),purpose='Default 10 tasks, gamma=.999 N9, target EMA coefficient sweep',
        environments=list(core.ENVIRONMENTS),families=core.FAMILIES,training_seeds=[0,1],
        methods=['control','ema0p001','ema0p0005'],rates={'control':.005,**core.RATES},
        discount=.999,expectile=.9,sample_counts=[9],steps=1000000,batch_size=256,lr=.0003,
        jobs=jobs,reused_jobs=reused,total_cells=60,requested_intervention_cells=40,
        dataset_dir=str(args.dataset_dir.expanduser().resolve()),
        resources=resources(args.gpus,args.per_gpu,args.eval_workers),
        evaluation_steps=[800000,900000,1000000],deployment_modes=['lean','actor'],episodes_per_point=50,
        actor_label_samples=9,lean_deployment_samples=9,critic_heads=2,target_clipping=True,
        head_rule='AntMaze min; other families mean',learned_V_updates=0,
        actor_objective='best9 MDN likelihood distillation; no Q maximization',
        evaluation_reset_seeds=list(range(42,92)),evaluation_CPU_cores=4,
        target_update_timing='EMA of pre-update online Q, preserving the original implementation',
        intervention_scope='Only tau and descriptive recipe metadata change; all schedules, RNG and losses preserved',
        nondefault_study_temporarily_paused=True,prior_batch_and_DR3_studies_remain_cancelled=True,
        source_sha256={str(p.resolve()):sha(p) for p in sources},reused_evidence_sha256=evidence,
        packages={k:importlib.metadata.version(k) for k in ('ogbench','jax','jaxlib','flax','optax','numpy','scipy','mujoco')})
    registration.write_text(json.dumps(plan,indent=2)+'\n')
    (root/'analysis').mkdir(exist_ok=True);(root/'validation').mkdir(exist_ok=True)
    print(json.dumps(dict(new_training=len(jobs),reused_controls=20,reused_completed_EMA=len(reused)-20,
                          requested_settings=40,root=str(root)),indent=2))


if __name__=='__main__':main()
