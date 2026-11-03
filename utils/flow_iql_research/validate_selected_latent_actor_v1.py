"""Small mathematical and optimizer readiness checks, no performance claims."""
import json
from pathlib import Path
import sys
import time

from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import latent_selected_actor_v1 as latent
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import jax
import jax.numpy as jnp
import numpy as np
import optax


def main():
    output=Path(sys.argv[1]);assert not output.exists()
    assert all(d.platform=='cpu'for d in jax.devices())
    pars=dict(logits=jnp.zeros((2,1)),means=jnp.zeros((2,1,3)),scales=jnp.ones((2,1,3)))
    values=jnp.array([[0.,1.,3.],[-2.,0.,4.]])
    observed=latent.latent_log_probability(pars,values)
    expected=-.5*jnp.sum(values**2,axis=-1)-1.5*jnp.log(2*jnp.pi)
    np.testing.assert_allclose(observed,expected,rtol=1e-6,atol=1e-6)
    draws=latent.sample_latent(pars,jnp.concatenate([jnp.zeros((2,1)),values],-1))
    assert np.array_equal(np.asarray(draws),np.asarray(values))
    assert np.abs(np.asarray(draws)).max()>1
    cases=[]
    for od,ad in ((28,5),(55,5),(29,8)):
        x=jax.random.normal(jax.random.PRNGKey(10),(16,od))
        z=jax.random.normal(jax.random.PRNGKey(11),(16,ad))
        noise=jax.random.normal(jax.random.PRNGKey(12),(16,ad+1))
        target=jnp.tanh(x[:,:ad]+z)
        rdef=latent.MixtureActor((32,32),ad,9,.1)
        ddef=latent._Actor((32,32),ad,False)
        rparams=rdef.init(jax.random.PRNGKey(13),x,noise,return_parameters=True)['params']
        dparams=ddef.init(jax.random.PRNGKey(14),x,z)['params']
        router=TrainState.create(rdef,rparams,optax.adam(.0003))
        decoder=TrainState.create(ddef,dparams,optax.adam(.0003))
        for _ in range(8):
            def rl(p):
                d=rdef.apply({'params':p},x,noise,return_parameters=True)
                loss=-latent.latent_log_probability(d,z).mean()/ad
                return loss,dict(loss=loss)
            def dl(p):
                loss=((ddef.apply({'params':p},x,z)-target)**2).mean()
                return loss,dict(loss=loss)
            router,_=router.apply_loss_fn(rl);decoder,_=decoder.apply_loss_fn(dl)
        assert int(router.opt_state[0].count)==int(decoder.opt_state[0].count)==8
        params=dict(router=router.params,decoder=decoder.params)
        actor=latent.LatentSelectedActor((32,32),ad,9,.1)
        p=rdef.apply({'params':router.params},x,noise,return_parameters=True)
        manual=jnp.clip(ddef.apply({'params':decoder.params},x,latent.sample_latent(p,noise)),-1,1)
        prediction=jax.jit(lambda pp:actor.apply({'params':pp},x,noise))(params)
        difference=float(np.max(np.abs(np.asarray(prediction-manual))))
        assert difference<=1e-6
        assert np.isfinite(np.asarray(prediction)).all()and np.abs(np.asarray(prediction)).max()<=1
        cases.append(dict(observation_dim=od,action_dim=ad,actual_router_Adam=8,
            actual_decoder_Adam=8,composite_manual_max_difference=difference))
    output.write_text(json.dumps(dict(time=time.time(),status='pass',cases=cases,
        ordinary_density_matches_standard_normal=True,latent_sampling_not_clipped=True,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,latent.__file__]},
        performance_claim=False),indent=2,allow_nan=False))
    print(json.dumps(dict(status='pass',cases=cases)))


if __name__=='__main__':main()
