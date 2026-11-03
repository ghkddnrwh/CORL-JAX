"""Test inversion, Jacobians, likelihood, real Adam and actor graph structure."""
import argparse
import json
from pathlib import Path
import time
from utils.flow_iql_research import onestep_deploy_core_v1 as legacy
from utils.flow_iql_research import affine_latent_actor_v1 as affine
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()and all(d.platform=='cpu'for d in jax.devices())
    rows=[]
    for d in [2,5,21]:
        x=jax.random.normal(jax.random.PRNGKey(10),(4,3));eps=jax.random.normal(jax.random.PRNGKey(11),(4,d))
        model=affine.ConditionalAffineRouter((8,8),d,4,1.)
        params=model.init(jax.random.PRNGKey(12),x,eps)['params']
        initial=model.apply({'params':params},x,eps)
        np.testing.assert_array_equal(initial['latent'],eps);np.testing.assert_array_equal(initial['logdet'],0)
        # Perturb every output head, avoiding a vacuous identity-only inverse check.
        varied=jax.tree_util.tree_map(lambda v:v+.03*jnp.sin(jnp.arange(v.size).reshape(v.shape)+1),params)
        transformed=model.apply({'params':varied},x,eps)
        reverse=model.apply({'params':varied},x,transformed['latent'],inverse=True)
        err=float(jnp.max(jnp.abs(eps-reverse['latent'])));assert err<1e-5
        assert float(jnp.max(jnp.abs(transformed['logdet']+reverse['logdet'])))<1e-5
        jac=jax.jacrev(lambda e:model.apply({'params':varied},x[0],e)['latent'])(eps[0])
        sign,logdet=jnp.linalg.slogdet(jac);assert sign>0
        jerr=float(jnp.abs(logdet-transformed['logdet'][0]));assert jerr<1e-5
        density=affine.log_probability(model,varied,x,transformed['latent'])
        expected=-.5*jnp.sum(eps*eps,-1)-.5*d*jnp.log(2*jnp.pi)-transformed['logdet']
        assert np.max(np.abs(np.asarray(density-expected)))<2e-5
        state=TrainState.create(model,params,optax.adam(.0003));target=eps+.3*x[:,:1]
        def update(s):
            def loss(p):
                value=-affine.log_probability(model,p,x,target).mean()/d
                return value,dict(loss=value)
            return s.apply_loss_fn(loss)[0]
        update=jax.jit(update);full=state
        for _ in range(8):full=update(full)
        resumed=state
        for _ in range(4):resumed=update(resumed)
        resumed=flax.serialization.from_state_dict(state,flax.serialization.to_state_dict(resumed))
        for _ in range(4):resumed=update(resumed)
        for u,v in zip(jax.tree_util.tree_leaves(full),jax.tree_util.tree_leaves(resumed)):np.testing.assert_array_equal(u,v)
        assert int(full.opt_state[0].count)==8 and all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(full))
        assert any(not np.array_equal(u,v)for u,v in zip(jax.tree_util.tree_leaves(full.params),jax.tree_util.tree_leaves(params)))
        rows.append(dict(dimension=d,identity_init_exact=True,nonidentity_inverse_max_error=err,
            analytic_vs_autodiff_logdet_error=jerr,actual_router_Adam=8,resume4plus4_bitwise=True,finite=True))
    m=dict(router_hidden_dims=[8,8],decoder_hidden_dims=[8]*4,action_dim=5,coupling_layers=4,
        log_scale_bound=1.,latent_draws=1,generated_candidate_actions=0,flow_integration_steps=0,
        actor_forward_passes=1,requires_critic=False,requires_BC_flow=False)
    model=affine.AffineLatentActor((8,8),(8,)*4,5,4,1.)
    params=model.init(jax.random.PRNGKey(31),jnp.zeros((3,)),jnp.zeros((5,)))['params']
    policy=affine.make_policy(m);out=policy(params,jnp.zeros((3,)),jax.random.PRNGKey(42))
    assert out.shape==(5,)and np.isfinite(np.asarray(out)).all()
    hlo=jax.xla_computation(policy)(params,jnp.zeros((3,)),jax.random.PRNGKey(42)).as_hlo_text()
    dots=sum(' dot('in line for line in hlo.splitlines());assert dots==17,dots
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(time=time.time(),status='pass',rows=rows,MLP_blocks=5,Dense_dots=dots,latent_draws=1,
        output_actions=1,candidate_bank=False,flow_ODE_steps=0,critic_required=False,
        no_performance_or_latency_claim=True,prior_art='https://arxiv.org/abs/1605.08803',
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,affine.__file__]})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result))


if __name__=='__main__':main()
