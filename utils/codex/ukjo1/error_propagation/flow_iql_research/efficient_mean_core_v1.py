"""Speed ablations of the frozen finite-mean method, with explicit BC students.

BC distillation never maximizes Q. A distinct guided policy, when enabled,
maximizes Q for deployment only and never supplies bootstrap proposals.
"""
from pathlib import Path
from utils.flow_iql_research import core as base_core
from utils.flow_iql_research import benchmark_core
from utils.flow_iql_research import mean_empirical_backup_core_v1 as reference
from utils.flow_iql_research import selection_v2
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.vendor.networks import ActorVectorField
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax


ENVIRONMENTS=base_core.ENVIRONMENTS
VARIANTS=('reference','fql','lean','euler5','euler2','bc_student','anchor_student','guided_actor','student_guided')
BC_STUDENTS=('bc_student','anchor_student','student_guided')
GUIDED_POLICIES=('guided_actor','student_guided')
WARMUP=50000


@flax.struct.dataclass
class State:
    agent: object
    v_opt: object
    v_updates: object
    q_updates: object
    student: object
    student_updates: object
    policy_student: object
    policy_updates: object


def config_for(environment,variant,width=None):
    assert variant in VARIANTS and environment in ENVIRONMENTS
    if variant=='fql':
        return benchmark_core.config_for(environment,.999,width=width,kind='fql')
    cfg=reference.config_for(environment,.999,.9,width)
    cfg.update(num_samples=9,flow_steps=10,agent_name='ifql',policy_type='ifql',
        flow_temperature=3.,distill_coefficient=30.)
    return cfg


def specification(variant):
    return dict(variant=variant,learned_V_fits=2 if variant=='reference' else 0,
        flow_steps=5 if variant=='euler5' else 2 if variant=='euler2' else 10,
        backup_samples=1 if variant=='fql' else 9,
        BC_student=variant in BC_STUDENTS,guided_deployment_policy=variant in GUIDED_POLICIES,
        BC_student_warmup=WARMUP if variant in BC_STUDENTS else 0,
        teacher_backup_anchor=variant=='anchor_student',
        deployment='one_step_guided' if variant in GUIDED_POLICIES or variant=='fql' else 'student_best9' if variant in BC_STUDENTS else 'flow_best9')


def create(seed,observations,actions,cfg,variant):
    module,kind=(benchmark_core,'fql') if variant=='fql' else (reference,'ifql')
    old=module.create(seed,observations,actions,cfg,kind)
    def student(tag):
        definition=ActorVectorField(hidden_dims=tuple(cfg['actor_hidden_dims']),action_dim=actions.shape[-1],
            layer_norm=cfg['actor_layer_norm'])
        key=jax.random.fold_in(jax.random.PRNGKey(seed),tag)
        params=definition.init(key,observations,jnp.zeros_like(actions))['params']
        return TrainState.create(definition,params,optax.adam(cfg['lr']))
    return State(old.agent,old.v_opt,old.v_updates,old.q_updates,
        student(8819) if variant in BC_STUDENTS else None,jnp.asarray(0),
        student(8820) if variant in GUIDED_POLICIES else None,jnp.asarray(0))


def flow(agent,observations,noises,steps=10):
    name='actor_flow' if 'modules_actor_flow' in agent.network.params else 'actor_bc_flow'
    actions=noises
    for index in range(steps):
        times=jnp.full((*actions.shape[:-1],1),index/steps)
        actions=actions+agent.network.select(name)(observations,actions,times,is_encoded=True)/steps
    return jnp.clip(actions,-1,1)


def bank(agent,observations,key,n=9):
    action_key,_=jax.random.split(key)
    shape=(n,*observations.shape[:-1],agent.config['action_dim'])
    noises=jax.random.normal(action_key,shape)
    obs=jnp.broadcast_to(observations,(n,*observations.shape))
    return obs,noises


def active_actions(state,observations,key,variant):
    spec=specification(variant)
    obs,noises=bank(state.agent,observations,key)
    if variant not in BC_STUDENTS:
        return flow(state.agent,obs,noises,spec['flow_steps'])
    def compressed(_):
        actions=jnp.clip(state.student(obs,noises),-1,1)
        if variant=='anchor_student':
            anchor=flow(state.agent,observations,noises[0],10)
            actions=actions.at[0].set(anchor)
        return actions
    return jax.lax.cond(state.q_updates<WARMUP,lambda _:flow(state.agent,obs,noises,10),compressed,operand=None)


def build_update(variant,interval):
    if variant=='reference':
        return reference.build_update('bounded_v2_mean_empirical_backup',interval,'ifql')
    if variant=='fql':
        return benchmark_core.build_update('official',interval,'fql')
    def update(state,batch):
        agent,net=state.agent,state.agent.network
        cfg=agent.config
        new_rng,rng=jax.random.split(agent.rng)
        proposal_key=jax.random.fold_in(rng,61723)
        actions=active_actions(state,batch['next_observations'],proposal_key,variant)
        obs=jnp.broadcast_to(batch['next_observations'],(9,*batch['next_observations'].shape))
        proposal_q=net.select('target_critic')(obs,actions=actions).mean(0)
        root=selection_v2.empirical_expectile(proposal_q,cfg['expectile'])
        unbounded=batch['rewards']+cfg['discount']*batch['masks']*root
        target=jnp.clip(unbounded,*interval)
        _,actor_key=jax.random.split(rng)
        def loss(parameters):
            q=net.select('critic')(batch['observations'],actions=batch['actions'],params=parameters)
            critic_loss=jnp.mean(jnp.sum((q-target)**2,axis=0))
            actor_loss,actor_info=agent.actor_loss(batch,parameters,actor_key)
            return critic_loss+actor_loss,dict(critic_loss=critic_loss,BC_loss=actor_loss,
                **{'actor/'+k:v for k,v in actor_info.items()})
        new_net,logs=net.apply_loss_fn(loss)
        student,student_updates=state.student,state.student_updates
        policy_student,policy_updates=state.policy_student,state.policy_updates
        if variant in BC_STUDENTS:
            _,noises=bank(agent,batch['next_observations'],proposal_key)
            teacher=jax.lax.stop_gradient(flow(agent,batch['next_observations'],noises[0],10))
            def distill(parameters):
                predicted=student(batch['next_observations'],noises[0],params=parameters)
                error=jnp.mean((predicted-teacher)**2)
                return error,dict(BC_distill_MSE=error)
            student,student_logs=student.apply_loss_fn(distill)
            student_updates=student_updates+1
            logs.update({'student/'+k:v for k,v in student_logs.items()})
            logs['student/used_for_backup']=(state.q_updates>=WARMUP).astype(jnp.float32)
        if variant in GUIDED_POLICIES:
            noises=jax.random.normal(jax.random.fold_in(rng,73819),batch['actions'].shape)
            teacher=jax.lax.stop_gradient(flow(agent,batch['observations'],noises,10))
            # Public FQL coefficients: Ant10, Humanoid30, manipulation300/1000.
            coefficient=cfg['efficient_guidance_alpha']
            def guidance(parameters):
                raw=policy_student(batch['observations'],noises,params=parameters)
                distill=jnp.mean((raw-teacher)**2)
                q=net.select('critic')(batch['observations'],actions=jnp.clip(raw,-1,1)).mean(0)
                q_loss=-q.mean()
                return coefficient*distill+q_loss,dict(distill_MSE=distill,Q_loss=q_loss)
            policy_student,policy_logs=policy_student.apply_loss_fn(guidance)
            policy_updates=policy_updates+1
            logs.update({'policy/'+k:v for k,v in policy_logs.items()})
        agent.target_update(new_net,'critic')
        agent=agent.replace(network=new_net,rng=new_rng)
        score=jnp.mean(jnp.where(proposal_q-root>=0,cfg['expectile'],1-cfg['expectile'])*(proposal_q-root),axis=0)
        logs.update({'backup/clip_fraction':jnp.mean((unbounded!=target).astype(jnp.float32)),
            'backup/lower_clip_fraction':jnp.mean((unbounded<interval[0]).astype(jnp.float32)),
            'backup/upper_clip_fraction':jnp.mean((unbounded>interval[1]).astype(jnp.float32)),
            'backup/finite_proposal_score_abs_mean':jnp.mean(jnp.abs(score)),
            'value/steps':jnp.asarray(0.,jnp.float32)})
        return state.replace(agent=agent,q_updates=state.q_updates+1,student=student,student_updates=student_updates,
            policy_student=policy_student,policy_updates=policy_updates),logs
    return update


def set_guidance_config(cfg,environment,variant):
    if variant in GUIDED_POLICIES:
        cfg['efficient_guidance_alpha']=benchmark_core.config_for(environment,.999,kind='fql')['alpha']
    return cfg


make_scan=base_core.make_scan


def policy_parameters(state):
    return dict(base=state.agent.network.params,bc_student=None if state.student is None else state.student.params,
        guided=None if state.policy_student is None else state.policy_student.params,q_updates=state.q_updates)


def make_policy(template,variant,mode='primary',backend='cpu'):
    assert mode in ('primary','teacher')
    def policy(parameters,observations,key):
        agent=template.agent.replace(network=template.agent.network.replace(params=parameters['base']))
        if variant=='fql' or (mode=='primary' and variant not in BC_STUDENTS and variant not in GUIDED_POLICIES and variant not in ('euler5','euler2')):
            return agent.sample_actions(observations=observations,seed=key,temperature=0.)
        if mode=='primary' and variant in GUIDED_POLICIES:
            noise_key,_=jax.random.split(key)
            noise=jax.random.normal(noise_key,(agent.config['action_dim'],))
            return jnp.clip(template.policy_student(observations,noise,params=parameters['guided']),-1,1)
        obs,noises=bank(agent,observations,key)
        if mode=='primary' and variant in BC_STUDENTS:
            actions=jax.lax.cond(parameters['q_updates']<WARMUP,
                lambda _:flow(agent,obs,noises,10),
                lambda _:jnp.clip(template.student(obs,noises,params=parameters['bc_student']),-1,1),operand=None)
        else:
            steps=10 if mode=='teacher' else specification(variant)['flow_steps']
            actions=flow(agent,obs,noises,steps)
        q=agent.network.select('critic')(obs,actions=actions).min(0)
        return actions[jnp.argmax(q)]
    return jax.jit(policy,backend=backend)


def make_probe(variant,interval):
    @jax.jit
    def probe(state,batch):
        if variant=='fql':
            values=benchmark_core.probe(state,batch)
            return dict(q=values[0],target_q=values[1],y=values[2],finite=values[3])
        agent=state.agent
        q=agent.network.select('critic')(batch['observations'],actions=batch['actions'])
        target_q=agent.network.select('target_critic')(batch['observations'],actions=batch['actions'])
        key=jax.random.PRNGKey(41019)
        obs,noises=bank(agent,batch['next_observations'],key)
        teacher_actions=flow(agent,obs,noises,10)
        active=active_actions(state,batch['next_observations'],key,variant)
        teacher_values=agent.network.select('target_critic')(obs,actions=teacher_actions).mean(0)
        values=agent.network.select('target_critic')(obs,actions=active).mean(0)
        root=selection_v2.empirical_expectile(values,agent.config['expectile'])
        teacher_root=selection_v2.empirical_expectile(teacher_values,agent.config['expectile'])
        y=jnp.clip(batch['rewards']+agent.config['discount']*batch['masks']*root,*interval)
        score=jnp.mean(jnp.where(values-root>=0,agent.config['expectile'],1-agent.config['expectile'])*(values-root),axis=0)
        finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(state)]))
        return dict(q=q,target_q=target_q,y=y,finite=finite,paired_action_MSE=jnp.mean((active-teacher_actions)**2),
            paired_root_error=root-teacher_root,paired_proposal_Q_error=values-teacher_values,
            finite_proposal_score=score)
    return probe


def probe_summary(values,interval):
    rms=lambda x:float(np.sqrt(np.mean(np.asarray(x,np.float64)**2)))
    q=np.asarray(values['q'],np.float64)
    result=dict(finite=bool(np.asarray(values['finite'])),q_mean=float(q.mean()),q_rms=rms(q),
        q_centered_rms=rms(q-q.mean()),q_disagreement_rms=rms(q[0]-q[1]),
        q_rms_over_return_bound=rms(q)/abs(interval[0]),td_rms=rms(q-np.asarray(values['y'])[None]),
        q_target_lag_rms=rms(q-np.asarray(values['target_q'])))
    if 'paired_action_MSE' in values:
        result.update(paired_action_MSE=float(np.asarray(values['paired_action_MSE'])),
            paired_root_error_RMS=rms(values['paired_root_error']),
            paired_root_error_abs_max=float(np.abs(np.asarray(values['paired_root_error'])).max()),
            paired_proposal_Q_error_RMS=rms(values['paired_proposal_Q_error']),
            finite_proposal_score_abs_mean=float(np.abs(np.asarray(values['finite_proposal_score'])).mean()))
    return result


def source_paths():
    folder=Path(__file__).resolve().parent
    names=['efficient_mean_core_v1.py','core.py','benchmark_core.py','policy_core.py',
        'mean_empirical_backup_core_v1.py','selection_v2.py','train_mean_empirical_backup_v1.py']
    paths=[folder/name for name in names]+sorted((folder/'vendor').glob('*.py'))
    paths += [folder.parent/'propagation_research'/name for name in ('reward_bounds.py','seeded_reset.py')]
    return paths
