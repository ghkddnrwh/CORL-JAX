"""Mathematical and optimizer readiness, not an environment experiment."""
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import teacher_gradient_metric_v1 as metric
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.vendor.flax_utils import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax


def main():
    output=Path(sys.argv[1]);assert not output.exists()
    key=jax.random.PRNGKey(921)
    g=jax.random.normal(key,(32,2,5));t=jnp.zeros((32,5))
    x=jax.random.normal(jax.random.fold_in(key,1),(32,5))*.1
    scale=jnp.mean(g*g);normalized=metric.normalized_gradients(g,scale)
    for coefficient in (0.,1.,10.):
        value,info=metric.loss(x,t,normalized,coefficient)
        assert float(value)>=float(info['raw_MSE'])
        zero,_=metric.loss(t,t,normalized,coefficient);assert float(zero)==0
        assert np.all(np.asarray(jax.grad(lambda gg:metric.loss(x,t,gg,coefficient)[0])(normalized))==0)
    pure=jax.grad(lambda xx:jnp.mean((xx-t)**2))(x)
    control=jax.grad(lambda xx:metric.loss(xx,t,normalized,0.)[0])(x)
    assert np.array_equal(np.asarray(pure),np.asarray(control))
    rescaled=metric.normalized_gradients(g*7,scale*49)
    np.testing.assert_allclose(rescaled,normalized,rtol=2e-6,atol=2e-6)
    expected=jnp.mean(jnp.sum(normalized*x[:,None],axis=-1)**2)/5
    assert abs(float(expected)-float(metric.loss(x,t,normalized,1.)[1]['teacher_linearized_penalty']))<1e-7
    large=metric.normalized_gradients(g*100,scale)
    assert float(jnp.max(jnp.sum(large**2,-1)))<=50.00002
    obs=jax.random.normal(jax.random.fold_in(key,2),(32,55))
    noise=jax.random.normal(jax.random.fold_in(key,3),(32,45))
    definition=flat.actor_definition(dict(hidden_dims=[32,32],action_dim=5,layer_norm=False))
    initial=definition.init(key,obs,noise)['params']
    actor=TrainState.create(definition,initial,optax.adam(.0003))
    for _ in range(8):
        def objective(p):
            raw=definition.apply({'params':p},obs,noise)
            return metric.loss(raw,t,normalized,1.)
        actor,_=actor.apply_loss_fn(objective)
    assert int(actor.opt_state[0].count)==8
    assert all(np.isfinite(np.asarray(x)).all()for x in jax.tree_util.tree_leaves(actor))
    output.write_text(json.dumps(dict(time=time.time(),status='pass',actual_Adam=8,
        zero_coefficient_gradient_exact=True,target_zero_all_coefficients=True,
        gradient_geometry_stopped=True,common_scale_invariance=True,
        linear_critic_identity_checked=True,gradient_norm_cap_checked=True,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,metric.__file__]},
        no_performance_claim=True),indent=2,allow_nan=False))
    print('passed')


if __name__=='__main__':main()
