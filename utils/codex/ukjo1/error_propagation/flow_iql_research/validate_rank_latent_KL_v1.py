"""Analytic prior KL, score-offset invariance, parameter stops and real Adam."""
import argparse
import json
from pathlib import Path
import time
from utils.flow_iql_research import onestep_deploy_core_v1 as legacy
from utils.flow_iql_research import affine_latent_actor_v1 as affine
from utils.flow_iql_research import rank_latent_KL_loss_v1 as loss_core
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import jax
import jax.numpy as jnp
import numpy as np
import optax


def norm(tree):return float(jnp.sqrt(sum(jnp.sum(v*v)for v in jax.tree_util.tree_leaves(tree))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()and all(d.platform=='cpu'for d in jax.devices())
    obs=jnp.array([[.1,.2],[.3,-.2]],jnp.float32);epsilon=jnp.array([[.2,-.3],[-.4,.1]],jnp.float32)
    selected=epsilon+.2;bank=jnp.broadcast_to(jnp.linspace(-4,4,9),(2,9))
    router=affine.ConditionalAffineRouter((8,8),2,4,1.)
    rp=router.init(jax.random.PRNGKey(1),obs,epsilon)['params']
    decoder=affine._Actor((8,)*4,2,False);dp=decoder.init(jax.random.PRNGKey(2),obs,epsilon)['params']
    def critic(offset=0.):
        def call(x,actions):
            q=actions[...,0]+.3*actions[...,1]+offset
            return jnp.stack([q,q+1.])
        return call
    def f(r,d,off=0.):return loss_core.objective(router,r,obs,epsilon,decoder,d,critic(off),bank+off,selected,.3)
    value,info=f(rp,dp);other=f(rp,dp,100.)[0];assert abs(float(value-other))<2e-5
    rg,dg=jax.grad(lambda r,d:f(r,d)[0],argnums=(0,1))(rp,dp)
    assert norm(dg)==0 and norm(rg)>0
    assert 0<float(info['rank_CDF_min'])<=float(info['rank_CDF_mean'])<=1
    # KL of N(mu,sigma²I) to N(0,I), integrated by points with exact first two moments.
    dimension=2;sigma=1.2;mu=jnp.array([.3,-.1])
    quadrature=jnp.sqrt(2.)*jnp.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    z=mu+sigma*quadrature
    estimated=jnp.mean(.5*jnp.sum(z*z-quadrature*quadrature,-1)-dimension*jnp.log(sigma))
    exact=.5*jnp.sum(mu*mu)+dimension*(.5*(sigma*sigma-1)-jnp.log(sigma))
    assert abs(float(estimated-exact))<1e-6
    # Cross-entropy anchor reconstruction is in nats, without a dimension-dependent coefficient.
    assert abs(float(value-info['reverse_rank_KL_surrogate']-.3*info['forward_relative_cross_entropy']))<1e-6
    rows=[]
    for coefficient in [.3,3.]:
        state=TrainState.create(router,rp,optax.adam(.0003))
        for _ in range(8):
            def loss(params):return loss_core.objective(router,params,obs,epsilon,decoder,dp,critic(),bank,selected,coefficient)
            state,_=state.apply_loss_fn(loss)
        assert int(state.opt_state[0].count)==8
        assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(state))
        rows.append(dict(coefficient=coefficient,actual_router_Adam=8,full_optimizer_finite=True))
    result=dict(time=time.time(),status='pass',rows=rows,decoder_Q_gradient_norm=norm(dg),router_gradient_norm=norm(rg),
        Q_offset_invariance_error=abs(float(value-other)),analytic_prior_KL_error=abs(float(estimated-exact)),
        finite_bank_smooth_CDF_is_surrogate_not_exact_KL=True,no_performance_or_GPU_claim=True,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,loss_core.__file__,affine.__file__]})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result))


if __name__=='__main__':main()
