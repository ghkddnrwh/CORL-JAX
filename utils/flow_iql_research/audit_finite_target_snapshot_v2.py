"""Correct legacy learned-V TD labels using frozen actual finite backup targets.

Training code is immutable. This external diagnostic reconstructs its fixed
train probe exactly and evaluates the finite min/mean target on a fixed new
proposal draw. This is a snapshot residual, not an actual replay-batch loss.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import policy_core,selection_v2
from utils.flow_iql_research.evaluate_cross_seed_temporal_v2 import read
from utils.propagation_research.reward_bounds import return_bounds
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--steps',default='800000,900000,1000000');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();run=args.run.resolve()
    conf=json.loads((run/'config.json').read_text());identity=conf['identity'];cfg=dict(identity['config'])
    assert identity['kind']=='ifql'and identity['profile']in('bounded_v2_empirical_backup','bounded_v2_mean_empirical_backup')
    for path,digest in conf['source_sha256'].items():assert sha(path)==digest,path
    aggregation=cfg.get('backup_Q_aggregation','min');assert aggregation in('min','mean')and cfg['backup_samples']==9
    env,train,_=ogbench.make_env_and_datasets(identity['environment'])
    fields=('observations','actions','next_observations','rewards','masks')
    data={key:np.asarray(train[key],np.float32)for key in fields};data['actions']=np.clip(data['actions'],-1+1e-5,1-1e-5)
    indices=np.random.RandomState(identity['seed']+19401).randint(len(data['actions']),size=1024)
    fixed={key:jnp.asarray(value[indices])for key,value in data.items()}
    bounds=return_bounds(identity['environment'],cfg['discount'],float(np.min(data['rewards'])),float(np.max(data['rewards'])),'ogbench')
    for key in('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    template=policy_core.create(identity['seed'],fixed['observations'][:1],fixed['actions'][:1],cfg,'ifql')
    @jax.jit
    def predict(params):
        agent=template.agent.replace(network=template.agent.network.replace(params=params));net=agent.network
        q=net.select('critic')(fixed['observations'],actions=fixed['actions'])
        target_data=net.select('target_critic')(fixed['observations'],actions=fixed['actions']).min(0)
        learned_v=net.select('value')(fixed['observations']);next_v=net.select('value')(fixed['next_observations'])
        candidates,_=selection_v2.proposals(agent,fixed['next_observations'],jax.random.PRNGKey(19402),9,'behavior')
        repeated=jnp.broadcast_to(fixed['next_observations'],(9,*fixed['next_observations'].shape))
        heads=net.select('target_critic')(repeated,actions=candidates)
        minimum=heads.min(0);average=heads.mean(0)
        root_min=selection_v2.empirical_expectile(minimum,cfg['expectile'])
        root_mean=selection_v2.empirical_expectile(average,cfg['expectile'])
        return q,target_data,learned_v,next_v,minimum,average,root_min,root_mean
    diagnostics=[json.loads(line)for line in(run/'diagnostics.jsonl').read_text().splitlines()];rows=[];raw={}
    for step in map(int,args.steps.split(',')):
        config,saved,cp=read(run,step);assert config==conf
        state=flax.serialization.from_state_dict(template,saved['state'])
        assert int(state.q_updates)==int(state.agent.network.opt_state[0].count)==step
        assert int(state.v_updates)==int(state.v_opt[0].count)==2*step
        assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(state))
        values=jax.device_get(predict(state.agent.network.params))
        q,tq,v,nv,minimum,average,rmin,rmean=[np.asarray(x,np.float64)for x in values]
        root=rmin if aggregation=='min'else rmean;proposals=minimum if aggregation=='min'else average
        rewards=np.asarray(fixed['rewards'],np.float64);masks=np.asarray(fixed['masks'],np.float64)
        actual=rewards+cfg['discount']*masks*root;projected=np.clip(actual,*bounds)
        counterfactual=rewards+cfg['discount']*masks*nv
        residual=proposals-root;finite_score=np.mean(np.where(residual>=0,cfg['expectile'],1-cfg['expectile'])*residual,axis=0)
        aux_difference=tq-v;aux_score=np.where(aux_difference>=0,cfg['expectile'],1-cfg['expectile'])*aux_difference
        row=next(row for row in diagnostics if row['step']==step);rms=lambda x:float(np.sqrt(np.mean(x*x)))
        reproduced=rms(q-counterfactual[None]);matches=bool(np.isclose(reproduced,row['td_rms'],rtol=1e-3,atol=5e-4))
        legacy_cpu=jax.device_get(policy_core.probe(state,fixed))
        legacy_q=np.asarray(legacy_cpu[0],np.float64);legacy_y=np.asarray(legacy_cpu[3],np.float64)
        same_function_cpu_td=rms(legacy_q-legacy_y[None])
        marginal_control={
            'q_mean':bool(np.isclose(q.mean(),row['q_mean'],rtol=1e-3,atol=5e-4)),
            'q_rms':bool(np.isclose(rms(q),row['q_rms'],rtol=1e-3,atol=5e-4)),
            'v_mean':bool(np.isclose(v.mean(),row['v_mean'],rtol=1e-3,atol=5e-4)),
            'v_rms':bool(np.isclose(rms(v),row['v_rms'],rtol=1e-3,atol=5e-4))}
        assert all(marginal_control.values()),'Underlyingpredictionmagnitudesmismatch'
        q_graph_rms=rms(q-legacy_q);y_graph_rms=rms(counterfactual-legacy_y)
        numerical_bound=q_graph_rms+y_graph_rms
        assert abs(reproduced-same_function_cpu_td)<=numerical_bound+1e-9

        root_gap=rmean-rmin;max_head_gap=np.max(average-minimum,axis=0)
        assert np.min(root_gap)>=-1e-3 and np.max(root_gap-max_head_gap)<=1e-3
        rows.append(dict(step=step,aggregation=aggregation,
            actual_unprojected_finite_backup_snapshot_TD_RMS=rms(q-actual[None]),
            actual_projected_finite_backup_snapshot_TD_RMS=rms(q-projected[None]),
            actual_projected_finite_backup_snapshot_critic_MSE=float(np.mean(np.sum((q-projected[None])**2,axis=0))),
            legacy_logged_td_rms=row['td_rms'],reconstructed_learned_V_counterfactual_TD_RMS=reproduced,
            legacy_probe_reproduction_with_CPU_GPU_tolerance=matches,
            same_legacy_function_CPU_TD_RMS=same_function_cpu_td,
            fused_formula_vs_legacy_CPU_Q_RMS_difference=q_graph_rms,
            fused_formula_vs_legacy_CPU_target_RMS_difference=y_graph_rms,
            CPU_graph_TD_RMS_difference_triangle_bound=numerical_bound,
            actual_CPU_graph_TD_RMS_difference=abs(reproduced-same_function_cpu_td),
            legacy_scalar_logged_GPU_vs_CPU_absolute_difference=abs(same_function_cpu_td-row['td_rms']),
            marginal_prediction_magnitude_controls=marginal_control,
            auxiliary_learned_V_score_RMS=rms(aux_score),finite_proposal_root_score_abs_mean=float(np.mean(np.abs(finite_score))),
            mean_minus_min_root_mean=float(root_gap.mean()),mean_minus_min_root_RMS=rms(root_gap),
            root_mean=float(root.mean()),within_state_proposal_SD_mean=float(proposals.std(0).mean()),
            lower_projection_fraction=float(np.mean(actual<bounds[0])),upper_projection_fraction=float(np.mean(actual>bounds[1])),
            logged_window_actual_critic_loss=row['critic/critic_loss'],
            checkpoint_sha256=sha(cp),actual_Q_Adam_count=step,actual_V_Adam_count=2*step))
        for name,value in [('Q',q),('actual_y',actual),('projected_y',projected),('legacy_V_y',counterfactual),('proposal_Q',proposals),('root',root)]:raw[f'{step}_{name}']=value
    env.close();args.output.parent.mkdir(parents=True,exist_ok=True)
    arrays=args.output.with_suffix('.npz');np.savez_compressed(arrays,indices=indices,**raw)
    result=dict(time=time.time(),kind='actual_finite_backup_frozen_snapshot_TD_audit',run=run.name,environment=identity['environment'],rows=rows,
        fixed_probe_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),fixed_probe_count=1024,
        config_sha256=sha(run/'config.json'),diagnostics_sha256=sha(run/'diagnostics.jsonl'),array_archive_sha256=sha(arrays),source_sha256=sha(__file__),
        reconstruction_backend='cpu',formula_dtype='float64_from_float32_network_predictions',
        numerical_control_note='V1strictrelativeTDcontrolfailedforcollapsedminmodels;V2retainsandreportsthemismatch,andrequiresmarginalpredictioncontrolsplusanexacttriangleboundbetweentwoCPUgraphresiduals. LoggedGPUarrayswere notsaved,so noGPUkernel/precisionfaultisattributed.',
        correction='Legacytd_rmsforthesetwofinite-backupfamiliesusesr+gamma*mask*learnedV(s_next). ItisacounterfactualwhenVnotusedintheQtarget. Trainingcritic_lossandtrainingtargetcodeareunchanged/correctfortheregisteredmethod.',
        limits=['SnapshotcurrentQ/targetQ/BCandonefixedproposalset;notactualsampledtrainingbatchesorapopulationBellmanerror.',
            'Finiteproposalscoreiszeroforthatsetonly;notoracleconditionalresidualzeta.',
            'Smallfixed-dataRMSdoesnotcontrolmaxerrororheldoutgoalvalueerror.'],primary_scores_unchanged=True,goal_complete=False)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(run=run.name,rows=rows)),flush=True)


if __name__=='__main__':main()
