"""CPU inference from a single mean-backup learner's actor and online critic.

No source checkpoint, V, target critic, optimizer, replay or training RNG is
needed. Repository code and the recorded framework versions remain required.
"""
import hashlib
from importlib.metadata import version
from pathlib import Path
import pickle
from types import SimpleNamespace
from utils.flow_iql_research import core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


RECIPE=dict(training_backup='finite_expectile_of_mean_target_twins',deployment='argmax_min_online_twins',
    samples=9,flow_steps=10,learners=1,expectile=.9,discount=.999)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frameworks():return {name:version(name)for name in('jax','jaxlib','flax','optax')}


def inference_sources():
    root=Path(__file__).resolve().parents[2];folder=root/'utils/flow_iql_research'
    files=[Path(__file__).resolve(),folder/'core.py',root/'utils/propagation_research/reward_bounds.py']
    files+=sorted((folder/'vendor').glob('*.py'))
    return {str(path.relative_to(root)):sha(path)for path in files}


def load(path):
    with Path(path).open('rb')as stream:bundle=pickle.load(stream)
    assert bundle['kind']=='mean_finite_expectile_IFQL_inference_v1'and bundle['resumable_training_checkpoint']is False
    assert bundle['step']==1000000 and bundle['recipe']==RECIPE
    assert set(bundle['network_params'])=={'modules_actor_flow','modules_critic'}
    assert not {'state','opt_state','replay_rng','v_opt','training_rng'}&bundle.keys()
    root=Path(__file__).resolve().parents[2]
    for relative,digest in bundle['inference_source_sha256'].items():
        assert not Path(relative).is_absolute()and '..'not in Path(relative).parts
        assert sha(root/relative)==digest,relative
    assert bundle['framework_versions']==frameworks(),'Use the recorded validated framework versions for this bundle.'
    identity=bundle['identity'];cfg=dict(identity['config'])
    assert identity['environment']==bundle['environment']and identity['seed']==bundle['training_seed']
    assert identity['kind']=='ifql'and identity['profile']=='bounded_v2_mean_empirical_backup'
    assert cfg['expectile']==.9 and cfg['discount']==.999 and cfg['num_samples']==cfg['backup_samples']==9 and cfg['flow_steps']==10
    assert cfg['backup_Q_aggregation']=='mean'and cfg['encoder']is None
    for key in('actor_hidden_dims','value_hidden_dims'):cfg[key]=tuple(cfg[key])
    cpu=jax.devices('cpu')[0]
    with jax.default_device(cpu):
        template=core.create(identity['seed'],jnp.zeros((1,bundle['observation_dim'])),jnp.zeros((1,bundle['action_dim'])),cfg,'ifql')
    params={key:flax.serialization.from_state_dict(template.agent.network.params[key],value)for key,value in bundle['network_params'].items()}
    params=jax.device_put(params,cpu)
    assert all(np.isfinite(np.asarray(a)).all()for a in jax.tree_util.tree_leaves(params))
    # The official forward pass selects only these two modules. Empty optimizer
    # state and step0 are inference scaffolding, never claimed as trained clocks.
    network=template.agent.network.replace(params=params,tx=None,opt_state=None,step=0)
    agent=template.agent.replace(network=network)
    def policy(parameters,observation,key):
        current=agent.replace(network=agent.network.replace(params=parameters))
        return current.sample_actions(observations=observation,seed=key,temperature=0.)
    return bundle,SimpleNamespace(agent=agent),jax.jit(policy,backend='cpu')
