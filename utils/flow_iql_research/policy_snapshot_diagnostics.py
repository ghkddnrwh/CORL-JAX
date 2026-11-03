"""Heldout policy/teacher/critic diagnostics on an immutable checkpoint.

Conditional references use64 generated actions, not true behavior actions at the
same state. Their finite-proposal expectile is exact; behavior-model and Monte
Carlo errors remain. A large sampled Q gain is not evidence of real improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core,policy_core,mixture_core,selection_v2 as selection
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--queries',type=int,default=256);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing diagnostic evidence')
    conf=json.loads((args.run/'config.json').read_text());ident=conf['identity'];kind=ident['kind']
    for source,digest in conf['source_sha256'].items():assert sha(source)==digest,source
    cfg=dict(ident['config'])
    for name in ('actor_hidden_dims','value_hidden_dims'):cfg[name]=tuple(cfg[name])
    _,_,val=ogbench.make_env_and_datasets(ident['environment'])
    rng=np.random.RandomState(77231);idx=rng.randint(len(val['actions']),size=args.queries)
    obs=np.asarray(val['observations'][idx],np.float32);actions=np.clip(np.asarray(val['actions'][idx],np.float32),-1+1e-5,1-1e-5)
    module=mixture_core if kind in ('gaussian_bc','mixture_bc') else policy_core
    template=module.create(ident['seed'],jnp.zeros_like(obs[:1]),jnp.zeros_like(actions[:1]),cfg,kind)
    with args.checkpoint.open('rb') as f:saved=pickle.load(f)
    state=flax.serialization.from_state_dict(template,saved['state']);agent=state.agent;net=agent.network
    @jax.jit
    def analyze(observations,data_actions):
        key=jax.random.PRNGKey(77128)
        proposal_actions,q=selection.proposals(agent,observations,key,64,'behavior')
        ob=jnp.broadcast_to(observations,(64,*observations.shape))
        tq=net.select('target_critic')(ob,actions=proposal_actions).min(0)
        v=net.select('value')(observations)
        qdata=net.select('critic')(observations,actions=data_actions)
        expectile=cfg['expectile']
        roots={str(n):selection.empirical_expectile(tq[:n],expectile) for n in (16,32,64)}
        adv=tq-v;w=jnp.where(adv>=0,expectile,1-expectile)
        score=(w*adv).mean(0);kappa=w.mean(0)
        records=dict(q_data=qdata,v=v,proposal_q=q,proposal_target_q=tq,
                     empirical_target_expectiles=roots,conditional_score=score,conditional_kappa=kappa,
                     score_over_kappa=score/kappa,
                     weighted_target_mean_minus_v=(w*tq).sum(0)/w.sum(0)-v,
                     proposal_action_saturation=jnp.mean((jnp.abs(proposal_actions)>=.98).astype(jnp.float32)))
        if 'modules_actor_onestep_flow' in net.params:
            action_key,_=jax.random.split(key)
            noises=jax.random.normal(action_key,data_actions.shape)
            student_raw=net.select('actor_onestep_flow')(observations,noises)
            student=jnp.clip(student_raw,-1,1)
            teacher=noises
            for i in range(cfg['flow_steps']):
                t=jnp.full((*observations.shape[:-1],1),i/cfg['flow_steps'])
                teacher=teacher+net.select('actor_flow')(observations,teacher,t)/cfg['flow_steps']
            teacher=jnp.clip(teacher,-1,1)
            qs=net.select('critic')(observations,actions=student)
            qt=net.select('critic')(observations,actions=teacher)
            records.update(student_actions=student,student_raw=student_raw,paired_teacher_actions=teacher,
                           student_q=qs,paired_teacher_q=qt,
                           paired_distillation_squared_error=jnp.mean((student_raw-teacher)**2,axis=-1))
        if 'modules_actor_flow' in net.params:
            _,xkey,tkey=jax.random.split(jax.random.PRNGKey(77129),3)
            x0=jax.random.normal(xkey,data_actions.shape);t=jax.random.uniform(tkey,(len(data_actions),1))
            xt=(1-t)*x0+t*data_actions
            pred=net.select('actor_flow')(observations,xt,t)
            records['heldout_flow_matching_squared_error']=jnp.mean((pred-(data_actions-x0))**2,axis=-1)
        return records
    result=jax.device_get(analyze(jnp.asarray(obs),jnp.asarray(actions)))
    rms=lambda x:float(np.sqrt(np.mean(np.asarray(x,np.float64)**2)))
    q=np.asarray(result['proposal_q'],np.float64);v=np.asarray(result['v'],np.float64)
    qdata=np.asarray(result['q_data'],np.float64);roots=result['empirical_target_expectiles']
    summary=dict(data_q_mean=float(qdata.min(0).mean()),data_q_disagreement_rms=rms(qdata[0]-qdata[1]),
        data_q_positive_fraction=float(np.mean(qdata.min(0)>0)),v_mean=float(v.mean()),
        behavior_proposal_q_mean=float(q.mean()),behavior_best32_q_mean=float(q[:32].max(0).mean()),
        behavior_proposal_q_positive_fraction=float(np.mean(q>0)),
        behavior_best32_q_positive_fraction=float(np.mean(q[:32].max(0)>0)),
        behavior_proposal_action_saturation=float(result['proposal_action_saturation']),
        v_minus_empirical_target_expectile64_rms=rms(v-roots['64']),
        empirical_target_expectile32_vs64_rms=rms(roots['32']-roots['64']),
        empirical_target_expectile16_vs64_rms=rms(roots['16']-roots['64']),
        conditional_model_score_rms=rms(result['conditional_score']),
        conditional_model_score_over_kappa_rms=rms(result['score_over_kappa']),
        finite_proposal_alignment_identity_max_error=float(np.max(np.abs(result['score_over_kappa']-result['weighted_target_mean_minus_v']))))
    if 'student_q' in result:
        qs=np.asarray(result['student_q'],np.float64);qt=np.asarray(result['paired_teacher_q'],np.float64)
        summary.update(student_q_mean=float(qs.min(0).mean()),paired_teacher_q_mean=float(qt.min(0).mean()),
            student_minus_paired_teacher_q=float((qs.min(0)-qt.min(0)).mean()),
            student_minus_data_q=float((qs.min(0)-qdata.min(0)).mean()),
            student_positive_q_fraction=float(np.mean(qs.min(0)>0)),
            student_q_disagreement_rms=rms(qs[0]-qs[1]),
            student_action_saturation=float(np.mean(np.abs(result['student_actions'])>=.98)),
            paired_teacher_action_saturation=float(np.mean(np.abs(result['paired_teacher_actions'])>=.98)),
            student_teacher_action_rms=rms(result['student_actions']-result['paired_teacher_actions']),
            heldout_paired_distill_MSE=float(np.mean(result['paired_distillation_squared_error'])))
    if 'heldout_flow_matching_squared_error' in result:
        summary['heldout_flow_matching_MSE']=float(np.mean(result['heldout_flow_matching_squared_error']))
    output=dict(time=time.time(),run=args.run.name,step=saved['step'],queries=args.queries,summary=summary,
        checkpoint_sha256=sha(args.checkpoint),config_sha256=sha(args.run/'config.json'),
        diagnostic_source_sha256=sha(__file__),proposal_source_sha256=sha(selection.__file__),
        query_indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest(),
        caveats=['Generated64action references include behavior-model and MonteCarlo error; they are not oracle conditional residuals.',
                 'A predicted Q gain or action/data distance does not certify actual action quality or out-of-distribution status.',
                 'Same-latent teacher/student distances measure the actual distillation pairing; real policy success must come from environment evaluations.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(output,indent=2,allow_nan=False));tmp.replace(args.output)
    print(json.dumps(dict(run=args.run.name,step=saved['step'],summary=summary)),flush=True)


if __name__=='__main__':main()
