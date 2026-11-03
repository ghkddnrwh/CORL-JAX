"""Check raw and explicitly normalized actor-only clients without RL imports."""
import hashlib
import json
from pathlib import Path
import sys
import time

from utils.flow_iql_research import standalone_mixture_actor_v1 as old
from utils.flow_iql_research import standalone_mixture_actor_v2 as new

import jax
import jax.numpy as jnp
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=Path(sys.argv[1]).resolve();out=root/'analysis/mixture_capacity_CPU_v1/portable_V2_validation_v1.json';assert not out.exists()
    paths=[root/'analysis/mixture_onestep_pipeline_v1/actual_shapes_v1/cube/full/actor_8.pkl',
           root/'analysis/mixture_onestep_pipeline_v1/actual_shapes_v1/hmmed/full/actor_8.pkl',
           root/'analysis/frozen_mixture_multilabel_v1/puzzle3_v2/NLL4_input_standardized_private_actor_20000.pkl',
           root/'analysis/frozen_mixture_multilabel_v1/cube_single_v2/NLL4_input_standardized_private_actor_20000.pkl']
    assert all(x.platform=='cpu'for x in jax.devices())
    rows=[]
    for path in paths:
        actor=new.load_actor(path,allow_diagnostic=True);m=actor.metadata
        normalized='observation_normalization'in m
        if normalized:
            case='puzzle3_v1'if' puzzle'.strip()in str(path)else'cube_single_v1'
            with np.load(root/'analysis/broad_frozen_teacher_feasibility_v1'/case/'frozen_replay_pairs_v1.npz')as cache:
                obs=cache['observations'][65536:65600].copy()
        else:obs=np.asarray(jax.random.normal(jax.random.PRNGKey(49351),(64,m['observation_dim'])))
        key=jax.random.PRNGKey(49352);got=actor.policy(actor.params,jnp.asarray(obs),key)
        if normalized:
            mean=jnp.asarray(m['observation_normalization']['mean'],jnp.float32);scale=jnp.asarray(m['observation_normalization']['scale'],jnp.float32)
            definition=new.definition(m)
            def reference(params,observations,key):
                action_key,_=jax.random.split(key);latent=jax.random.normal(action_key,(*observations.shape[:-1],m['noise_dim']))
                return definition.apply({'params':params},(observations-mean)/scale,latent)
            want=jax.jit(reference,backend='cpu')(actor.params,jnp.asarray(obs),key)
        else:
            prior=old.load_actor(path);want=prior.policy(prior.params,jnp.asarray(obs),key)
        difference=float(jnp.max(jnp.abs(got-want)));assert difference<1e-6,difference
        rows.append(dict(path=str(path),sha256=sha(path),normalized=normalized,max_action_difference=difference,
            actor_forward_passes=m['actor_forward_passes'],generated_candidates=m['generated_candidate_actions']))
    forbidden=[name for name in sys.modules if name=='ogbench'or name.startswith(('utils.flow_iql_research.vendor','utils.flow_iql_research.onestep_deploy_core'))]
    assert not forbidden,forbidden
    out.write_text(json.dumps(dict(time=time.time(),status='pass',rows=rows,only_actor_trees_loaded=True,no_RL_imports=True,no_optimizer_or_GPU_updates=True,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,old.__file__,new.__file__]},
        scope='Compatibility/portable normalized diagnostic exports only; no change to frozenV1 used by active learners or goal performance claim.'),indent=2))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
