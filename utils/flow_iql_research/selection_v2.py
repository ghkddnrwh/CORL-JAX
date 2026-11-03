"""Fixed-proposal policy extraction and empirical expectile diagnostics.

The sign weights follow IDQL Eq9 (arxiv2304.10573); they are not a novelty
claim. The sample-expectile variant replaces the learned threshold with the
exact finite-proposal root. Its residual is zero only on that proposal set, not
under the unknown true behavior distribution. Neither choice changes training.
"""
from utils.flow_iql_research import core  # SciPy compatibility before JAX deps.
import jax
import jax.numpy as jnp

RULES=('max','implicit_learned','implicit_sample')


def empirical_expectile(q,expectile):
    """Exact piecewise-linear root, centered for numerical conditioning; axis0."""
    n=q.shape[0]
    center=q.mean(axis=0)
    ordered=jnp.sort(q-center,axis=0)
    prefix=jnp.concatenate([jnp.zeros_like(ordered[:1]),jnp.cumsum(ordered,axis=0)],axis=0)
    count=jnp.arange(n+1,dtype=q.dtype).reshape((n+1,)+(1,)*(q.ndim-1))
    denominator=(1-expectile)*count+expectile*(n-count)
    roots=((1-expectile)*prefix+expectile*(prefix[-1]-prefix))/denominator
    lower=jnp.concatenate([jnp.full_like(ordered[:1],-jnp.inf),ordered],axis=0)
    upper=jnp.concatenate([ordered,jnp.full_like(ordered[:1],jnp.inf)],axis=0)
    violation=jnp.maximum(lower-roots,0)+jnp.maximum(roots-upper,0)
    index=jnp.argmin(violation,axis=0)
    return jnp.take_along_axis(roots,index[None,...],axis=0)[0]+center


def proposals(agent,observations,seed,n,proposal_source="auto"):
    """Same noises and proposal forward pass for every selection rule."""
    cfg=agent.config;net=agent.network
    action_key,noise_key=jax.random.split(seed)
    leading=observations.shape[:-1]
    ob=jnp.broadcast_to(observations,(n,*observations.shape))
    if cfg.get('policy_type') in ('gaussian_bc','mixture_bc'):
        logits,means,logstd=net.select('actor')(observations)
        component=jax.random.categorical(action_key,logits,shape=(n,*leading))
        index=jnp.broadcast_to(component[...,None,None],(n,*leading,1,cfg['action_dim']))
        mu=jnp.take_along_axis(jnp.broadcast_to(means,(n,*means.shape)),index,axis=-2)[...,0,:]
        sigma=jnp.take_along_axis(jnp.broadcast_to(logstd,(n,*logstd.shape)),index,axis=-2)[...,0,:]
        actions=mu+jnp.exp(sigma)*jax.random.normal(noise_key,mu.shape)
    elif 'modules_actor_onestep_flow' in net.params and proposal_source!='behavior':
        noises=jax.random.normal(action_key,(n,*leading,cfg['action_dim']))
        actions=net.select('actor_onestep_flow')(ob,noises)
    elif 'modules_actor_flow' in net.params or 'modules_actor_bc_flow' in net.params:
        flow_name='actor_flow' if 'modules_actor_flow' in net.params else 'actor_bc_flow'
        actions=jax.random.normal(action_key,(n,*leading,cfg['action_dim']))
        for i in range(cfg['flow_steps']):
            t=jnp.full((n,*leading,1),i/cfg['flow_steps'])
            velocities=net.select(flow_name)(ob,actions,t,is_encoded=True)
            actions=actions+velocities/cfg['flow_steps']
    else:
        raise ValueError('This selector requires a flow or behavioral mixture proposal')
    actions=jnp.clip(actions,-1,1)
    q=net.select('critic')(ob,actions=actions).min(0)
    return actions,q


def select_index(q,v,expectile,seed,rule):
    if rule=='max':return jnp.argmax(q,axis=0)
    threshold=v if rule=='implicit_learned' else empirical_expectile(q,expectile)
    weights=jnp.where(q>=threshold,expectile,1-expectile)
    return jax.random.categorical(seed,jnp.moveaxis(jnp.log(weights),0,-1))


def make_policy(agent,rule='max',samples=None,proposal_source='auto'):
    assert rule in RULES
    n=samples or agent.config.get('num_samples',32)
    expectile=agent.config.get('expectile')
    if rule!='max':
        assert expectile is not None and 0<expectile<1
        assert 'modules_value' in agent.network.params
    def policy(params,observations,key):
        a=agent.replace(network=agent.network.replace(params=params))
        actions,q=proposals(a,observations,key,n,proposal_source)
        if rule=='max':
            index=jnp.argmax(q,axis=0)
        else:
            v=a.network.select('value')(observations)
            index=select_index(q,v,expectile,jax.random.fold_in(key,28403),rule)
        selected=jnp.broadcast_to(index[None,...,None],(1,*actions.shape[1:]))
        return jnp.take_along_axis(actions,selected,axis=0)[0]
    return jax.jit(policy,backend='cpu')
