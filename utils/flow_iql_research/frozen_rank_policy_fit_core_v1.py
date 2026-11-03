"""Fit only the policy block against one frozen Q/BC teacher, on CPU.

Extract actual policy Adam moments. Running the full joint Adam with zero Q/BC
gradients would still move those blocks through momentum, so never do that.
"""
import jax
import jax.numpy as jnp
import optax


POLICY='modules_actor_policy_flow'


def extract(state,key):
    joint=state.agent.network.opt_state
    assert len(joint)==2 and hasattr(joint[0],'mu')and hasattr(joint[0],'nu')
    adam=joint[0]
    policy_adam=type(adam)(count=adam.count,mu=adam.mu[POLICY],nu=adam.nu[POLICY])
    return state.agent.network.params[POLICY],(policy_adam,*joint[1:]),key


def objective(agent,policy_params,batch,rng):
    selected,_,_=agent.selected_teacher_targets(batch['observations'],jax.random.fold_in(rng,71927))
    shape=batch['actions'].shape
    initial=jax.random.normal(jax.random.fold_in(rng,71928),shape)
    time=jax.random.uniform(jax.random.fold_in(rng,71929),(*shape[:-1],1))
    position=(1-time)*initial+time*selected
    params={**agent.network.params,POLICY:policy_params}
    prediction=agent.network.select('actor_policy_flow')(batch['observations'],position,time,params=params)
    return jnp.mean((prediction-(selected-initial))**2)


def make_update(agent):
    assert agent.config['policy_objective']=='FM_of_best9_BC_action_distribution'
    optimizer=optax.adam(agent.config['lr'])
    def update(carry,batch):
        params,opt_state,key=carry
        new_key,rng=jax.random.split(key)
        loss,gradient=jax.value_and_grad(lambda value:objective(agent,value,batch,rng))(params)
        increments,opt_state=optimizer.update(gradient,opt_state,params)
        params=optax.apply_updates(params,increments)
        return (params,opt_state,new_key),loss
    return update


def policy_only_inference_state(base,policy_params):
    """Inference-only view; not a joint-optimizer resume state. Never save as one."""
    params={**base.agent.network.params,POLICY:policy_params}
    return base.replace(agent=base.agent.replace(network=base.agent.network.replace(params=params)))
