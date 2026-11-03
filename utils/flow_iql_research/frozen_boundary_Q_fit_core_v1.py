"""Isolated Q Adam updates on known success boundaries; never train other blocks."""
import jax
import jax.numpy as jnp
import optax

Q='modules_critic'


def extract(state,key):
    joint=state.agent.network.opt_state
    assert len(joint)==2 and hasattr(joint[0],'mu')
    adam=joint[0]
    isolated=type(adam)(count=adam.count,mu=adam.mu[Q],nu=adam.nu[Q])
    return state.agent.network.params[Q],(isolated,*joint[1:]),key


def objective(agent,q_params,batch,rng):
    uniform=jax.random.uniform(jax.random.fold_in(rng,67101),batch['actions'].shape,minval=-1.,maxval=1.)
    observations=jnp.concatenate((batch['observations'],batch['observations']))
    actions=jnp.concatenate((batch['actions'],uniform))
    values=agent.network.select('critic')(observations,actions=actions,
        params={**agent.network.params,Q:q_params})
    return jnp.mean(jnp.sum(values**2,axis=0))


def make_update(agent):
    optimizer=optax.adam(agent.config['lr'])
    def update(carry,batch):
        params,opt_state,key=carry
        new_key,rng=jax.random.split(key)
        loss,gradient=jax.value_and_grad(lambda value:objective(agent,value,batch,rng))(params)
        increments,opt_state=optimizer.update(gradient,opt_state,params)
        return (optax.apply_updates(params,increments),opt_state,new_key),loss
    return update


def inference_view(base,q_params):
    """This view is for evaluation only; it is not a joint training resume state."""
    params={**base.agent.network.params,Q:q_params}
    return base.replace(agent=base.agent.replace(network=base.agent.network.replace(params=params)))
