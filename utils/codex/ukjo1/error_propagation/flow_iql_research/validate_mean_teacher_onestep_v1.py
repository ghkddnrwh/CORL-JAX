"""Prototype logic/isolation and frozen-label equivalence, not return tests."""
import argparse
import json
from pathlib import Path
import pickle
import random
import time
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.flow_iql_research import mean_teacher_onestep_core_v1 as core
from utils.flow_iql_research.evaluate_teacher_aggregation_v1 import make_policy
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.validate_onestep_deploy_v1 import difference
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert not args.output.exists()
    assert all(device.platform=='cpu'for device in jax.devices())
    actions=jnp.asarray([[[-.8]],[[.4]],[[.9]]]);heads=jnp.asarray([[[20.],[5.],[1.]],[[-2.],[4.],[1.]]])
    target,index=core.choose_teacher(actions,heads)
    assert int(index[0])==0 and abs(float(target[0,0])+.8)<1e-6
    assert int(heads.min(0).argmax(0)[0])==1
    obs=jnp.asarray([[.1,.2,-.1],[.3,-.2,.4]],jnp.float32);act=jnp.asarray([[.2,-.1],[-.4,.3]],jnp.float32)
    batch=dict(observations=obs,actions=act,next_observations=obs*.9,rewards=jnp.full((2,),-1.),masks=jnp.ones((2,)))
    cases=[]
    for recipe in core.RECIPES:
        cfg=core.config_for(core.ENVIRONMENTS[0],recipe,width=8);state=core.create(6,obs,act,cfg,recipe)
        update=jax.jit(core.build_update(recipe,(-1000.,0.)))
        first,_=update(state,batch)
        reference=jax.jit(core.lean.build_update('lean',(-1000.,0.)))(state.base,batch)[0]
        diff=difference(flax.serialization.to_state_dict(first.base),flax.serialization.to_state_dict(reference));assert diff<1e-5
        poison=state.replace(actor=state.actor.replace(params=jax.tree_util.tree_map(lambda value:value+.25,state.actor.params)))
        other,_=update(poison,batch);assert difference(flax.serialization.to_state_dict(first.base),flax.serialization.to_state_dict(other.base))==0
        current=first
        for _ in range(7):current,_=update(current,batch)
        assert int(current.base.q_updates)==int(current.actor_updates)==int(current.actor.opt_state[0].count)==int(current.base.agent.network.opt_state[0].count)==8
        assert int(current.base.v_updates)==0
        metadata=core.actor_metadata(cfg,3,2);artifact=core.export_actor(current,metadata)
        action=np.asarray(core.make_actor_policy(metadata)(artifact['actor_params'],obs[0],jax.random.PRNGKey(42)))
        assert action.shape==(2,)and np.isfinite(action).all()and np.max(np.abs(action))<=1
        summary=core.probe_summary(jax.device_get(core.make_probe(recipe,(-1000.,0.))(current,batch)),(-1000.,0.))
        assert summary['finite']and all(np.isfinite(summary[key])for key in ['mean_teacher_raw_actor_MSE','mean_teacher_predicted_Q_mean_regret'])
        assert metadata['actor_forward_passes']==1 and not metadata['requires_critic']and metadata['generated_candidate_actions']==metadata['flow_integration_steps']==0
        cases.append(dict(recipe=recipe,Q_Adam=8,actor_Adam=8,V_Adam=0,base_difference_vs_lean=diff,actor_perturbation_base_effect=0.,mean_teacher_probe_finite=True,single_actor_only=True))
        print(json.dumps(dict(recipe=recipe,status='pass')),flush=True)
    run=args.root/'runs/os1_cubed_fqln_a1_s0';cp=run/'checkpoint_900000.pkl';cp_hash=sha(cp)
    conf=json.loads((run/'config.json').read_text());cfg=dict(conf['identity']['config'])
    for key in ('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    recipe='bank_bc__meanTeacher';cfg['onestep_specification']=core.specification(recipe);cfg['onestep_specification']['effective_alpha']=1.
    metadata=json.loads((run/'dataset.json').read_text());template=core.create(6,jnp.zeros((1,metadata['observation_dim'])),jnp.zeros((1,metadata['action_dim'])),cfg,recipe)
    with cp.open('rb')as stream:saved=pickle.load(stream)
    # Restore only the frozen base into the test template; no actor training,
    # artifact export, resume or performance claim is made from this hybrid.
    state=template.replace(base=flax.serialization.from_state_dict(template.base,saved['state']['base']))
    mean_policy=make_policy(state,'mean');payload=core.lean.policy_parameters(state.base)
    env=ogbench.make_env_and_datasets(conf['identity']['environment'],env_only=True);prepare_seeded_action_space(env,42)
    differences=[]
    for i in range(4):
        seed=42+i;np.random.seed(seed);random.seed(seed);env.action_space.seed(seed);ob,_=env.reset(seed=seed)
        ob=jnp.asarray(ob,jnp.float32);rng=jax.random.PRNGKey(28421+i)
        test=dict(observations=ob[None],next_observations=ob[None])
        label=core.teacher_batch(state,test,rng,cfg['onestep_specification'])[2][0]
        deployed=mean_policy(payload,ob,jax.random.fold_in(rng,61723))
        differences.append(float(np.abs(np.asarray(label)-np.asarray(deployed)).max()))
    env.close();assert max(differences)<1e-5 and sha(cp)==cp_hash
    result=dict(time=time.time(),status='pass',cases=cases,disagreement_fixture_mean_index=0,min_index=1,
        frozen_real_model_label_alignment=dict(checkpoint=str(cp),sha256=cp_hash,cases=4,max_differences=differences,optimizer_updates_performed=0),
        source_sha256={str(path.resolve()):sha(path)for path in [Path(__file__),*core.source_paths(),Path(make_policy.__code__.co_filename)]},
        no_performance_claim=True,no_GPU_or_production_queue_change=True,
        scope='Four synthetic8update prototypes and4 frozen CubeD label/deployment comparisons. Not a trained9task method or an efficacy result.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status='pass',output=str(args.output))),flush=True)


if __name__=='__main__':main()
