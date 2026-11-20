"""Read-only-to-teacher CPU best9 likelihood-distillation prototype.

Compare1and9component mixture actors on the identical frozen replay cache.
No environment rollout, joint learner, GPUqueue mutation or export.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# Apply the existing legacyJAX/SciPy shim before optax imports jax.scipy.linalg.
from utils.flow_iql_research import onestep_deploy_core_v1 as flat

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.integrate import quad
from scipy.stats import norm as scipy_normal
from scipy.special import logsumexp as scipy_logsumexp

from utils.flow_iql_research import mixture_onestep_actor_v1 as mixture
from utils.flow_iql_research.energy_onestep_deploy_core_v1 import norm
from utils.flow_iql_research.vendor.flax_utils import TrainState


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def validate_math():
    p=dict(logits=jnp.asarray([.3,-.2]),means=jnp.asarray([[-.3],[.7]]),scales=jnp.asarray([[.2],[.4]]))
    weights=np.asarray(jax.nn.softmax(p['logits']),np.float64)
    mu,sigma=np.asarray(p['means'])[:,0],np.asarray(p['scales'])[:,0]
    errors=[]
    for action in [-1.,-.5,0.,.8,1.]:
        terms=(scipy_normal.logcdf((-1-mu)/sigma) if action==-1 else
               scipy_normal.logcdf((mu-1)/sigma) if action==1 else scipy_normal.logpdf(action,mu,sigma))
        reference=float(scipy_logsumexp(np.log(weights)+terms))
        computed=float(mixture.log_probability(p,jnp.asarray([action])))
        errors.append(abs(reference-computed))
    assert max(errors)<2e-6,errors
    density=lambda a:float(np.sum(weights*scipy_normal.pdf(a,mu,sigma)))
    mass=float(np.sum(weights*(scipy_normal.cdf((-1-mu)/sigma)+scipy_normal.cdf((mu-1)/sigma))))
    total=mass+quad(density,-1,1)[0]
    assert abs(total-1)<1e-7,total
    # The deploy sampler generates one action, with the intended censoring.
    z=jax.random.normal(jax.random.PRNGKey(34101),(100000,2))
    batch={k:jnp.broadcast_to(v,(100000,*v.shape)) for k,v in p.items()}
    actions=np.asarray(mixture.sample_one(batch,z))[:,0]
    lower=float(np.mean(actions==-1));upper=float(np.mean(actions==1))
    expected_lower=float(np.sum(weights*scipy_normal.cdf((-1-mu)/sigma)))
    expected_upper=float(np.sum(weights*scipy_normal.cdf((mu-1)/sigma)))
    assert abs(lower-expected_lower)<.005 and abs(upper-expected_upper)<.005
    grad=jax.grad(lambda mu:-mixture.log_probability(dict(p,means=mu),jnp.asarray([1.])))(p['means'])
    assert np.isfinite(np.asarray(grad)).all()
    return dict(status='pass',normalization_total=total,max_log_probability_error_vs_SciPy=max(errors),
        empirical_lower_atom=lower,expected_lower_atom=expected_lower,
        empirical_upper_atom=upper,expected_upper_atom=expected_upper,
        action_samples_generated=100000,model_fit_updates=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache-directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists() and len(os.sched_getaffinity(0))==6
    assert all(d.platform=='cpu' for d in jax.devices())
    args.output.mkdir(parents=True);started=time.time()
    parent_path=args.cache_directory/'result.json';parent=json.loads(parent_path.read_text())
    for p,d in parent['source_sha256'].items():assert sha(p)==d,p
    cache_path=args.cache_directory/'fixed_pairs.npz';assert sha(cache_path)==parent['pair_cache_sha256']
    cache=np.load(cache_path);count=int(cache['train_size']);assert count==4096
    observations=jnp.asarray(cache['observations']);targets=jnp.asarray(cache['targets'])
    d=targets.shape[-1];eval_obs=observations[count::4]
    eval_y=jnp.moveaxis(targets[count:].reshape((256,4,d)),1,0)
    cfg=flat.config_for(parent['environment'],'bank_bc')
    core_path=Path(flat.__file__).with_name('energy_onestep_deploy_core_v1.py')
    result=dict(schema='clipped_mixture_best9_CPU_prototype_v1',started=started,
        environment=parent['environment'],parent_result=str(parent_path),parent_result_sha256=sha(parent_path),
        pair_cache_sha256=sha(cache_path),source_sha256={str(Path(p).resolve()):sha(p) for p in
            [__file__,mixture.__file__,str(core_path)]+[str(p) for p in flat.source_paths()]},
        mathematical_checks=validate_math(),components=[1,9],std_floor=.001,Adam_updates_per_fit=5000,
        learning_rate=.0003,batch_size=256,index_seed=34102,heldout_noise_seed=34103,
        likelihood='Exact clipped diagonal Gaussian mixture: endpoint probability masses, interior densities; loss is NLL/action_dim.',
        teacher='Frozen original minQ-best9 labels; no newteacher draws in this probe.',
        teacher_updates=0,registered_GPU_updates=0,RL_rollouts=0,exports=0,
        scope='Fixed-label CPU conditional-density fit, not joint learning or9task/speed result. OneMLP→mixture parameters→one categorical/normal action, no Q/candidates atdeploy.',
        prior_art='Bishop1994 Mixture Density Networks; no novelty claim for MDNs or ordinary likelihood fitting.',actors={})
    write(args.output/'status.json',result)
    for k in (1,9):
        definition=mixture.MixtureActor(tuple(cfg['actor_hidden_dims']),d,k)
        params=definition.init(jax.random.PRNGKey(34100),observations[:1],jnp.zeros((1,d+1)))['params']
        actor=TrainState.create(definition,params,optax.adam(.0003))
        record=dict(parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(params)),snapshots=[])
        result['actors'][str(k)]=record
        def fit(actor,key,steps):
            def update(carry,_):
                actor,key=carry;key,index_key=jax.random.split(key)
                indices=jax.random.randint(index_key,(256,),0,count)
                obs,y=observations[indices],targets[indices]
                def loss(params):
                    p=actor(obs,jnp.zeros((256,d+1)),params=params,return_parameters=True)
                    value=-jnp.mean(mixture.log_probability(p,y))/d
                    return value,dict(NLL=value)
                actor,_=actor.apply_loss_fn(loss)
                return (actor,key),None
            return jax.lax.scan(update,(actor,key),None,length=steps)[0]
        fit=jax.jit(fit,static_argnums=2)
        @jax.jit
        def metrics(actor):
            p=actor(observations[count:],jnp.zeros((1024,d+1)),return_parameters=True)
            train_p=actor(observations[:count],jnp.zeros((4096,d+1)),return_parameters=True)
            z=jax.random.normal(jax.random.PRNGKey(34103),(32,256,d+1))
            raw=actor(jnp.broadcast_to(eval_obs,(32,*eval_obs.shape)),z)
            cross=norm(raw[:,None]-eval_y[None,:]).mean()
            xx=norm(raw[:,None]-raw[None,:]).sum()/(32*31*256)
            yy=norm(eval_y[:,None]-eval_y[None,:]).sum()/(4*3*256)
            probabilities=jax.nn.softmax(p['logits'],axis=-1)
            entropy=-jnp.sum(probabilities*jax.nn.log_softmax(p['logits'],axis=-1),axis=-1)
            return dict(heldout_NLL_per_dimension=-mixture.log_probability(p,targets[count:]).mean()/d,
                train_NLL_per_dimension=-mixture.log_probability(train_p,targets[:count]).mean()/d,
                heldout_energy_distance_Ustat=2*cross-xx-yy,
                student_to_teacher_variance_ratio=jnp.var(raw,axis=0,ddof=1).mean()/jnp.maximum(jnp.var(eval_y,axis=0,ddof=1).mean(),1e-12),
                average_scale=p['scales'].mean(),minimum_scale=p['scales'].min(),
                component_entropy=entropy.mean(),effective_components=jnp.exp(entropy).mean(),
                action_endpoint_fraction=jnp.mean((jnp.abs(raw)>=1).astype(jnp.float32)))
        previous,key=0,jax.random.PRNGKey(34102)
        for updates in (0,100,1000,5000):
            if updates:actor,key=fit(actor,key,updates-previous)
            assert int(actor.opt_state[0].count)==updates
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(actor.params))
            snapshot=dict(Adam_updates=updates,elapsed_seconds=time.time()-started,**{k:float(v) for k,v in jax.device_get(metrics(actor)).items()})
            record['snapshots'].append(snapshot);write(args.output/'status.json',result)
            print(json.dumps(dict(components=k,**snapshot)),flush=True);previous=updates
    assert sha(cache_path)==parent['pair_cache_sha256']
    result.update(status='complete',completed=time.time(),scratch_Adam_updates=10000)
    write(args.output/'result.json',result);write(args.output/'status.json',result)


if __name__=='__main__':main()
