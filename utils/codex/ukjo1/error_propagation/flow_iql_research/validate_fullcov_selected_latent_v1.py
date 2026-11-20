"""Density/covariance/sampling checks and eight CPU Adam steps."""
import json
from pathlib import Path
import sys
import time
from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import fullcov_selected_latent_actor_v1 as full
from utils.flow_iql_research import latent_selected_actor_v1 as diagonal
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
from utils.flow_iql_research.vendor.flax_utils import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax


def main():
    output=Path(sys.argv[1]);assert not output.exists()
    l=np.array([[1.,0.,0.],[.4,.8,0.],[-.2,.3,1.1]],np.float32)
    values=np.array([[0.,1.,3.],[-2.,0.,4.]],np.float32)
    pars=dict(logits=jnp.zeros((2,1)),means=jnp.zeros((2,1,3)),cholesky=jnp.broadcast_to(l,(2,1,3,3)))
    expected=-.5*np.sum(np.linalg.solve(l,values.T).T**2,axis=-1)-np.log(np.diag(l)).sum()-1.5*np.log(2*np.pi)
    np.testing.assert_allclose(np.asarray(full.log_probability(pars,jnp.asarray(values))),expected,rtol=1e-6,atol=2e-6)
    actual=full.sample_latent(pars,jnp.concatenate([jnp.zeros((2,1)),jnp.asarray(values)],-1))
    np.testing.assert_allclose(np.asarray(actual),values@l.T,rtol=1e-6,atol=1e-6)
    assert np.abs(np.asarray(actual)).max()>1
    x=jax.random.normal(jax.random.PRNGKey(421),(16,55))
    z=jax.random.normal(jax.random.PRNGKey(422),(16,5))
    model=full.FullCovarianceRouter((32,32),5,9,.1)
    params=model.init(jax.random.PRNGKey(34210),x)['params']
    diag_model=diagonal.MixtureActor((32,32),5,9,.1)
    diag_params=diag_model.init(jax.random.PRNGKey(34210),x,jnp.zeros((16,6)),return_parameters=True)['params']
    for i in range(2):
        for field in ('kernel','bias'):assert np.array_equal(np.asarray(params['Dense_'+str(i)][field]),np.asarray(diag_params['Dense_'+str(i)][field]))
    state=TrainState.create(model,params,optax.adam(.0003))
    for _ in range(8):
        def objective(p):
            v=-full.log_probability(model.apply({'params':p},x),z).mean()/5
            return v,dict(loss=v)
        state,_=state.apply_loss_fn(objective)
    assert int(state.opt_state[0].count)==8
    assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(state))
    decoder=full._Actor((32,32),5,False)
    dparams=decoder.init(jax.random.PRNGKey(34220),x,z)['params']
    noise=jax.random.normal(jax.random.PRNGKey(423),(16,6))
    composed=full.FullCovSelectedActor((32,32),5,9,.1)
    p=model.apply({'params':state.params},x)
    action=composed.apply({'params':dict(router=state.params,decoder=dparams)},x,noise)
    manual=jnp.clip(decoder.apply({'params':dparams},x,full.sample_latent(p,noise)),-1,1)
    np.testing.assert_allclose(np.asarray(action),np.asarray(manual),rtol=1e-6,atol=1e-6)
    output.write_text(json.dumps(dict(time=time.time(),status='pass',actual_Adam=8,
        correlated_normal_density_matches_numpy=True,unclipped_sampling_matches_linear_transform=True,
        initial_hidden_trunk_matches_diagonal_router_exact=True,composite_manual_match=True,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,full.__file__]},performance_claim=False),indent=2,allow_nan=False))
    print('passed')


if __name__=='__main__':main()
