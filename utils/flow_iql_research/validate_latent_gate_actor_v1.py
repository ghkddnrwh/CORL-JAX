"""Eight actual classifier updates, permutation behavior, decoder and HLO proof."""
import argparse
import json
from pathlib import Path
import time
from utils.flow_iql_research import onestep_deploy_core_v1 as legacy
from utils.flow_iql_research import latent_gate_actor_v1 as gate
from utils.flow_iql_research.vendor.flax_utils import TrainState
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import jax
import jax.numpy as jnp
import numpy as np
import optax


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()and all(d.platform=='cpu'for d in jax.devices())
    x=jax.random.normal(jax.random.PRNGKey(1),(8,3));z=jax.random.normal(jax.random.PRNGKey(2),(8,9,2))
    definition=gate.SharedLatentGate((8,)*4);params=definition.init(jax.random.PRNGKey(3),x,z)['params']
    logits=definition.apply({'params':params},x,z)
    permutation=jnp.array([8,0,3,5,7,1,6,2,4]);reordered=definition.apply({'params':params},x,z[:,permutation])
    assert np.array_equal(np.asarray(reordered),np.asarray(logits[:,permutation]))
    np.testing.assert_array_equal(gate.select_latent(z,logits),gate.select_latent(z[:,permutation],reordered))
    labels=jnp.argmax(z[...,0]+.3*x[:,0,None],-1)
    state=TrainState.create(definition,params,optax.adam(.0003))
    for _ in range(8):
        def loss(p):
            l=optax.softmax_cross_entropy_with_integer_labels(definition.apply({'params':p},x,z),labels).mean()
            return l,dict(loss=l)
        state,_=state.apply_loss_fn(loss)
    assert int(state.opt_state[0].count)==8
    assert all(np.isfinite(np.asarray(v)).all()for v in jax.tree_util.tree_leaves(state))
    actor=gate.LatentGateActor((8,)*4,2,9);ap=actor.init(jax.random.PRNGKey(4),x,z.reshape(8,18))['params']
    metadata=dict(MLP_blocks=2,realized_action_samples=1,latent_draws=9,generated_candidate_actions=0,
        requires_critic=False,requires_BC_flow=False,hidden_dims=[8]*4,action_dim=2)
    policy=gate.make_policy(metadata);key=jax.random.PRNGKey(42)
    actual=policy(ap,x,key);noise=jax.random.normal(jax.random.split(key)[0],(8,18))
    expected=actor.apply({'params':ap},x,noise);assert np.max(np.abs(np.asarray(actual-expected)))<1e-6
    hlo=jax.xla_computation(policy)(ap,x[0],key).as_hlo_text();dots=sum(' dot('in line for line in hlo.splitlines());assert dots==10
    # For any gate, selected Z belongs to A only if at least one proposal does.
    # Exhaustively verify domination for a discrete prior, including score ties.
    import itertools
    proposals=np.array(list(itertools.product(range(3),repeat=3)))
    for indices in (np.zeros(len(proposals),int),proposals.argmax(1),proposals.argmin(1)):
        selected=proposals[np.arange(len(proposals)),indices]
        for subset in itertools.product([False,True],repeat=3):
            indicator=np.array(subset)
            assert indicator[selected].mean()<=3*indicator.mean()+1e-12
    result=dict(time=time.time(),status='pass',actual_router_Adam=8,permutation_equivariance_passed=True,
        latent_selection_permutation_invariant_without_ties=True,finite_full_optimizer=True,
        Dense_dots=dots,MLP_blocks=2,action_candidates_generated=0,latent_proposals=9,
        scalar_gating_logits=9,realized_action_samples=1,one_actor_forward=True,no_critic_or_Flow_in_graph=True,
        discrete_density_domination_check_passed=True,no_performance_or_latency_claim=True,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,gate.__file__]})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
