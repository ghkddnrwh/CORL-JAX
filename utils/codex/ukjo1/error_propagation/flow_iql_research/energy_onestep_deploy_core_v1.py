"""Experimental best9 distribution student over an unchanged finite-mean lean.

Energy-score minimization uses independent student/teacher draws given s.
It does not try to reproduce the teacher's particular nine-latent argmax map.
The population scoring property is not a finite-data RL performance guarantee.
Reference: Gneiting & Raftery2007, Section4.3Eq22 (minimization sign).
https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
"""
from pathlib import Path

import jax
import jax.numpy as jnp

from utils.flow_iql_research import onestep_deploy_core_v1 as base


ENVIRONMENTS = base.ENVIRONMENTS
RECIPES = ('energy_bc',)
State = base.State
lean = base.lean


@jax.custom_jvp
def norm(diff):
    return jnp.sqrt(jnp.mean(diff * diff,axis=-1))


@norm.defjvp
def norm_jvp(primals,tangents):
    (diff,), (tangent,) = primals,tangents
    value = norm(diff)
    denominator = jnp.where(value > 0,value,1.)
    derivative = jnp.where(value > 0,jnp.mean(diff * tangent,axis=-1) / denominator,0.)
    return value,derivative


def specification(recipe):
    assert recipe in RECIPES
    spec = base.specification('bank_bc')
    spec.update(name=recipe,input='single',objective='energy_distribution',
                distillation_objective_key='distill_objective',
                teacher_q_aggregation='min',student_noise_draws=2,
                student_noise_independent_of_teacher=True,
                pair_repulsion_coefficient=.5,
                norm='Euclidean_divided_by_sqrt_action_dim',
                effective_alpha=1.)
    return spec


def config_for(environment,recipe,width=None):
    cfg = base.config_for(environment,'bank_bc',width)
    cfg['onestep_specification'] = specification(recipe)
    return cfg


actor_metadata = base.actor_metadata
actor_definition = base.actor_definition
create = base.create
make_scan = base.make_scan
make_actor_policy = base.make_actor_policy
make_teacher_policy = base.make_teacher_policy
export_actor = base.export_actor


def teacher_batch(state,batch,rng,spec):
    observations,_,target,heads,actions,index = base.teacher_batch(state,batch,rng,spec)
    # Different PRNG branch from both original best9 bank and learner updates.
    # The two student draws are iid, never antithetic or teacher-coupled.
    key = jax.random.fold_in(rng,74321)
    noises = jax.random.normal(key,(2,*observations.shape[:-1],target.shape[-1]))
    return observations,noises,target,heads,actions,index


def actor_loss(state,observations,noises,target,proposal_heads,spec,params):
    assert spec['objective'] == 'energy_distribution' and noises.shape[0] == 2
    first = state.actor(observations,noises[0],params=params)
    second = state.actor(observations,noises[1],params=params)
    cross = .5*jnp.mean(norm(first-target)+norm(second-target))
    pair = jnp.mean(norm(first-second))
    energy = cross-.5*pair
    # MSE is logged only as a diagnostic, never used as the energy objective.
    mse = .5*jnp.mean((first-target)**2+(second-target)**2)
    saturation = .5*jnp.mean((jnp.abs(first)>1).astype(jnp.float32)+(jnp.abs(second)>1).astype(jnp.float32))
    return energy,dict(loss=energy,distill_objective=energy,energy_score=energy,
        energy_cross_mean_distance=cross,energy_student_pair_mean_distance=pair,
        distill_MSE=mse,distill_MSE_is_training_objective=jnp.asarray(0.),
        Q_loss=jnp.asarray(0.),raw_saturation_fraction=saturation)


def build_update(recipe,bounds):
    assert recipe in RECIPES
    lean_update = base.lean.build_update('lean',bounds)
    def update(state,batch):
        learner,logs = lean_update(state.base,batch)
        _,rng = jax.random.split(state.base.agent.rng)
        spec = state.base.agent.config['onestep_specification']
        observations,noises,target,heads,_,_ = teacher_batch(state,batch,rng,spec)
        actor,info = state.actor.apply_loss_fn(
            lambda params:actor_loss(state,observations,noises,target,heads,spec,params))
        logs.update({'deployment/'+k:v for k,v in info.items()})
        return state.replace(base=learner,actor=actor,actor_updates=state.actor_updates+1),logs
    return update


def make_probe(recipe,bounds):
    previous = base.make_probe(recipe,bounds)
    @jax.jit
    def probe(state,batch):
        values = previous(state,batch)
        spec = state.base.agent.config['onestep_specification']
        obs,noise,target,heads,_,_ = teacher_batch(state,batch,jax.random.PRNGKey(74913),spec)
        _,info = actor_loss(state,obs,noise,target,heads,spec,state.actor.params)
        return dict(**values,energy_distribution_score=info['energy_score'])
    return probe


def probe_summary(values,bounds):
    result = base.probe_summary(values,bounds)
    result.update(energy_distribution_score=float(values['energy_distribution_score']),
                  actor_probe_MSE_is_training_objective=0.)
    return result


def source_paths():
    return [Path(__file__).resolve()]+base.source_paths()
