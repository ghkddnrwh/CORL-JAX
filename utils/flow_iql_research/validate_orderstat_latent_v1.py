"""Check normalization, sampler Jacobian, density bound, and CPU Adam."""
import json
import math
from pathlib import Path
import sys
import time
from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import orderstat_latent_actor_v1 as order
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.vendor.flax_utils import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax


def main():
    output=Path(sys.argv[1]);assert not output.exists()
    nodes,weights=np.polynomial.hermite.hermgauss(128)
    cdf=np.array([.5*(1+math.erf(float(x)))for x in nodes])
    integral=float(np.dot(weights,9*cdf**8)/np.sqrt(np.pi))
    assert abs(integral-1)<1e-10
    v=jnp.array([1.,2.,-1.])/jnp.sqrt(6.)
    pars=dict(logits=jnp.array([-1e9,0.]),directions=v[None])
    jacobian_errors=[]
    for a in (-4.,0.,4.):
        epsilon=jnp.array([a,.2,-.4])
        def transform(e):return order.sample_latent(pars,jnp.concatenate([jnp.zeros((1,)),e]))
        z=transform(epsilon);jac=jax.jacfwd(transform)(epsilon)
        sign,logdet=jnp.linalg.slogdet(jac);assert float(sign)>0
        expected=-.5*jnp.sum(epsilon**2)-1.5*jnp.log(2*jnp.pi)-logdet
        actual=order.log_probability(pars,z)
        error=abs(float(expected-actual));assert error<=2e-5
        jacobian_errors.append(error)
    tails=order.maximum_normal_quantile(jnp.array([-100.,-10.,-6.,0.,6.,10.,100.]))
    assert np.isfinite(np.asarray(tails)).all()
    x=jax.random.normal(jax.random.PRNGKey(761),(32,55))
    z=jax.random.normal(jax.random.PRNGKey(762),(32,5))
    model=order.OrderStatisticRouter((32,32),5,9,9)
    params=model.init(jax.random.PRNGKey(34210),x)['params']
    for observations in (x,jnp.zeros_like(x)):
        pp=model.apply({'params':params},observations)
        np.testing.assert_allclose(np.asarray(jnp.sum(pp['directions']**2,-1)),1.,rtol=1e-6,atol=1e-6)
        lp=order.log_probability(pp,z)
        base=-.5*jnp.sum(z*z,-1)-2.5*jnp.log(2*jnp.pi)
        assert float(jnp.max(lp-base))<=math.log(9)+1e-5
        grads=jax.grad(lambda p:-order.log_probability(model.apply({'params':p},observations),z).mean())(params)
        assert all(np.isfinite(np.asarray(g)).all()for g in jax.tree_util.tree_leaves(grads))
    state=TrainState.create(model,params,optax.adam(.0003))
    for _ in range(8):
        def objective(p):
            loss=-order.log_probability(model.apply({'params':p},x),z).mean()/5
            return loss,dict(loss=loss)
        state,_=state.apply_loss_fn(objective)
    assert int(state.opt_state[0].count)==8
    assert all(np.isfinite(np.asarray(a)).all()for a in jax.tree_util.tree_leaves(state))
    decoder=order._Actor((32,32),5,False)
    dparams=decoder.init(jax.random.PRNGKey(34220),x,z)['params']
    noise=jax.random.normal(jax.random.PRNGKey(763),(32,6))
    composed=order.OrderStatisticSelectedActor((32,32),5,9,9)
    actual=composed.apply({'params':dict(router=state.params,decoder=dparams)},x,noise)
    pp=model.apply({'params':state.params},x)
    expected=jnp.clip(decoder.apply({'params':dparams},x,order.sample_latent(pp,noise)),-1,1)
    np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=1e-6,atol=1e-6)
    output.write_text(json.dumps(dict(time=time.time(),status='pass',actual_Adam=8,
        normalization_quadrature=integral,sampler_density_log_Jacobian_errors=jacobian_errors,
        unit_directions_and_zero_state_fallback_checked=True,ideal_density_ratio_bound_checked=True,
        tail_guard_finite=True,composite_manual_match=True,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,order.__file__]},
        no_performance_claim=True,tail_guard_ideal_Gaussian_probability_upper=math.erfc(10/math.sqrt(2))),indent=2,allow_nan=False))
    print('passed')


if __name__=='__main__':main()
