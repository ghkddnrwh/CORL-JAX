"""Small checks for source fidelity, optimizer semantics and checkpoint resume."""
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core, policy_core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def main():
    root=Path('logs/diagnosis/flow_iql_research_20261026/analysis')
    source=Path('algorithms/uk_offline/basic/propagation/fql')
    vendor=Path(core.__file__).parent/'vendor'
    for name in json.loads((vendor/'provenance.json').read_text()):
        expected=(source/name).read_text().replace('from utils.','from utils.flow_iql_research.vendor.')
        if name.startswith('agents/'):
            expected=expected.replace("network_params = network_def.init(init_rng, **network_args)['params']", "network_params = flax.core.unfreeze(network_def.init(init_rng, **network_args)['params'])")
        assert expected==(vendor/Path(name).name).read_text(),name
    cfg=core.config_for(core.ENVIRONMENTS[4],width=8)
    obs=jnp.zeros((1,4));act=jnp.zeros((1,2))
    initial=core.create(0,obs,act,cfg)
    rng=np.random.RandomState(128)
    batch={k:jnp.asarray(v) for k,v in dict(observations=rng.randn(5,4).astype('float32'),
        actions=rng.uniform(-1,1,(5,2)).astype('float32'),next_observations=rng.randn(5,4).astype('float32'),
        rewards=-np.ones(5,'float32'),masks=np.ones(5,'float32')).items()}
    baseline=jax.jit(core.build_update('official',(-np.inf,np.inf)))
    inactive=jax.jit(core.build_update('bound',(-np.inf,np.inf)))
    a=b=initial;errors=[]
    for _ in range(5):
        old=a;a,_=baseline(a,batch);b,_=inactive(b,batch)
        for x,y in zip(jax.tree_util.tree_leaves(a.agent.network.params),jax.tree_util.tree_leaves(b.agent.network.params)):
            np.testing.assert_allclose(x,y,atol=2e-6,rtol=2e-5)
            errors.append(float(np.max(np.abs(x-y))))
        for oldq,oldt,newt in zip(jax.tree_util.tree_leaves(old.agent.network.params['modules_critic']),
            jax.tree_util.tree_leaves(old.agent.network.params['modules_target_critic']),
            jax.tree_util.tree_leaves(a.agent.network.params['modules_target_critic'])):
            np.testing.assert_allclose(newt,.005*oldq+.995*oldt,atol=1e-7,rtol=2e-6)
    corrected=jax.jit(core.build_update('bounded_v2',(-100,0)))
    uninterrupted=initial
    for _ in range(4): uninterrupted,_=corrected(uninterrupted,batch)
    first=initial
    for _ in range(2): first,_=corrected(first,batch)
    serialized=pickle.loads(pickle.dumps(flax.serialization.to_state_dict(jax.device_get(first))))
    restored=flax.serialization.from_state_dict(initial,serialized)
    for _ in range(2): restored,_=corrected(restored,batch)
    for x,y in zip(jax.tree_util.tree_leaves(restored),jax.tree_util.tree_leaves(uninterrupted)):
        np.testing.assert_array_equal(x,y)
    assert int(restored.q_updates)==4 and int(restored.v_updates)==8
    policy_checks={}
    for kind in ('ifql','awrflow','distilled','fql_policy'):
        c=policy_core.config_for(core.ENVIRONMENTS[4],width=8)
        c.update(num_samples=2,policy_type=kind)
        s=policy_core.create(0,obs,act,c,kind)
        for name in ('modules_value','modules_critic','modules_target_critic'):
            for x,y in zip(jax.tree_util.tree_leaves(initial.agent.network.params[name]),jax.tree_util.tree_leaves(s.agent.network.params[name])):
                np.testing.assert_array_equal(x,y)
        s,_=jax.jit(policy_core.build_update('bounded_v2',(-100,0),kind))(s,batch)
        actions=s.agent.sample_actions(jnp.ones(4),seed=jax.random.PRNGKey(1),temperature=0.)
        assert actions.shape==(2,) and np.isfinite(actions).all()
        policy_checks[kind]=dict(initial_QV_bitwise_equal=True,q_updates=int(s.q_updates),v_updates=int(s.v_updates),finite_action=True)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(core.__file__),Path(policy_core.__file__),Path(__file__)]}
    result=dict(time=time.time(),passed=True,source_translation_exact=True,
        inactive_projection_separate_adam_v1_max_param_difference=max(errors),
        reference_target_EMA_uses_old_Q_verified=True,resume_full_state_bitwise_equal=True,
        corrected_q_updates=4,corrected_v_updates=8,policies=policy_checks,source_sha256=hashes,
        scope='Tiny CPU optimizer/initialization/source checks; real GPU/OGBench smoke is recorded separately. Not a learning-performance test.')
    (root/'implementation_validation_v1.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
