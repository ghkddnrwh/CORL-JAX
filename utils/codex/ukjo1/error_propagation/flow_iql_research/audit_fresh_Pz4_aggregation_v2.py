"""Posthoc paired-source audit of the completed first fresh Puzzle4x4 cohort."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
from utils.flow_iql_research import core,policy_core
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
from utils.flow_iql_research.watch_independent_critic_confirmation_v1 import ready_job
from utils.flow_iql_research.audit_independent_critic_policy_v2 import audit_job
from utils.flow_iql_research.audit_independent_critic_policy_v1 import audit_job as audit_median_job
from utils.flow_iql_research.evaluate_cross_seed_temporal_v2 import read
from utils.flow_iql_research.independent_critic_policy_v2 import make_selector
from utils.flow_iql_research.train_policy_v2 import make_policy
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    root=args.root.resolve();verifier=Verifier()
    registration_path=root/'independent_critic_rank_mean_confirmation_registration_v1.json'
    registration=json.loads(registration_path.read_text())
    for path,digest in registration['source_sha256'].items():assert verifier.digest(path)==digest,path
    assert verifier.digest(registration['training_plan'])==registration['training_plan_sha256']
    plan=json.loads(Path(registration['training_plan']).read_text())
    environment='puzzle-4x4-play-singletask-v0'
    actor=root/'runs/ieconfirm_s3_pz4_ifql_bounded_v2_g999_t90_N9'
    primary={r['step']:r for r in map(json.loads,(actor/'evaluations.jsonl').read_text().splitlines())}
    individual_path=root/'analysis/fresh_Pz4_individual_last3_audit_v1.json'
    individual=json.loads(individual_path.read_text())
    assert individual['status']=='complete' and individual['actual_unique_late_checkpoints_verified']==9
    for row in individual['rows']:
        assert row['environment']==environment and row['learner_seed']in(3,4,5)
        assert verifier.digest(root/'runs'/row['run']/'evaluations.jsonl')==row['evaluation_sha256']
    rules=[];raw_results={}
    for rule,directory in (('mean_rank','independent_critic_rank_mean_confirmation_v1'),
                           ('median_rank','independent_critic_fresh_confirmation_v1'),
                           ('mean','independent_critic_raw_mean_control_v1')):
        script_plan=copy.deepcopy(plan);script_plan['output_directory']=str(root/'analysis'/directory)
        rows=[];counts=dict(ensemble_only=0,own_only=0,both=0,neither=0)
        for step in (800000,900000,1000000):
            job=ready_job(root,script_plan,0,environment,step,rule,verifier);assert job is not None
            auditor=audit_median_job if rule=='median_rank' else audit_job
            audited=auditor(registration,job,verifier);assert audited['parent_learners_complete']
            result=json.loads(Path(job['output']).read_text());raw_results[job['id']]=verifier.digest(job['output'])
            aa=np.asarray(result['episode_successes']);bb=np.asarray(primary[step]['episode_successes'])
            counts['ensemble_only']+=int(np.sum((aa==1)&(bb==0)))
            counts['own_only']+=int(np.sum((aa==0)&(bb==1)))
            counts['both']+=int(np.sum((aa==1)&(bb==1)))
            counts['neither']+=int(np.sum((aa==0)&(bb==0)))
            rows.append(dict(step=step,ensemble=audited['success'],own_single=primary[step]['success_rate'],
                actor_checkpoint_sha256=result['actor_checkpoint_sha256'],result_sha256=audited['result_sha256']))
        rules.append(dict(rule=rule,rows=rows,last3_mean=statistics.mean(row['ensemble']for row in rows),
            own_single_last3_mean=statistics.mean(row['own_single']for row in rows),paired_reset_counts=counts))
    env,raw,val=ogbench.make_env_and_datasets(environment);env.close()
    indices=np.random.RandomState(91349).randint(len(val['actions']),size=64)
    observations=np.asarray(val['observations'][indices],np.float32)
    engine_checks=[]
    for step in (800000,900000,1000000):
        config,saved,path=read(actor,step);cfg=dict(config['identity']['config'])
        assert cfg['num_samples']==9
        for name in ('actor_hidden_dims','value_hidden_dims'):cfg[name]=tuple(cfg[name])
        template=policy_core.create(3,jnp.zeros((1,raw['observations'].shape[-1])),jnp.zeros((1,raw['actions'].shape[-1])),cfg,'ifql')
        state=jax.device_put(flax.serialization.from_state_dict(template,saved['state']),jax.devices('cpu')[0])
        native=make_policy(state.agent);single=make_selector(state.agent,'mean_rank',9)
        params=state.agent.network.params;critic=(params['modules_critic'],)
        maxima=[];different=0
        for index,observation in enumerate(observations):
            key=jax.random.split(jax.random.PRNGKey(42+index))[1]
            a=np.asarray(native(params,observation,key));b=np.asarray(single(params,critic,observation,key)[0])
            maxima.append(float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))
            different+=int(not np.array_equal(a,b))
        engine_checks.append(dict(step=step,query_rows=64,max_action_abs_difference=max(maxima),
            query_rows_with_different_action=different,checkpoint_sha256=verifier.digest(path)))
    result=dict(time=time.time(),status='complete',scope='One fresh task/cohort; no nine-task macro or extra training replication.',
        environment=environment,learner_group=[3,4,5],actor_seed=3,individual_last3={str(r['learner_seed']):r['last3_mean']for r in individual['rows']},
        fixed_rules=rules,raw_ensemble_result_sha256=raw_results,actual_unique_models_verified=len(verifier.checkpoints),
        native_vs_single_selector_CPU_query_checks=engine_checks,
        native_single_selector_actions_exact=all(r['query_rows_with_different_action']==0 for r in engine_checks),
        validation_query_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        individual_audit_sha256=verifier.digest(individual_path),registration_sha256=verifier.digest(registration_path),
        source_sha256=verifier.digest(__file__),new_benchmark_evaluations=0,new_GPU_jobs=0,goal_complete=False,
        limits=['Same64fixedvalidationobservationsareusedat3checkpoints;enginechecksarenotfullrolloutequivalence.',
            '150episodeoutcomesperrule reuse50resetseedsand3coupledtrainingcheckpoints,not150independenttrainingreplicas.',
            'Rankmeanversusraw/median differences are descriptive within this task; do not choose a rule by task.',
            'Nonzero ensemble success despite zero first-BC own-policy successes does not establish oracle Q accuracy or a universal error-cancellation guarantee.'])
    assert len(verifier.checkpoints)==9
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(individual=result['individual_last3'],rules={r['rule']:r['last3_mean']for r in rules},native_engine_exact=result['native_single_selector_actions_exact'],actual_models=9)),flush=True)


if __name__=='__main__':main()
