"""Independently check cached critic-action derivatives by finite differences."""
import argparse
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import onestep_deploy_core_v1 as flat
from utils.flow_iql_research import teacher_gradient_metric_v1 as metric
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('study',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    proposal=json.loads((a.study/'proposal_v1.json').read_text());rows=[]
    for name,case in proposal['cases'].items():
        status=json.loads((a.study/name/'status.json').read_text())
        gp=a.study/name/'teacher_action_gradients_v1.npz'
        assert sha(gp)==status['gradient_cache_sha256']
        assert sha(case['selected_latent_cache'])==case['cache_sha256']
        with np.load(gp)as f:all_g=f['gradients'].copy();scale=float(f['training_mean_square']);train=int(f['train_size'])
        recomputed=float(np.mean(all_g[:train].astype(np.float64)**2));assert recomputed==scale
        idx=np.r_[np.arange(8),np.arange(train,train+8)]
        with np.load(case['selected_latent_cache'])as f:
            x=f['observations'][idx].copy();index=f['best_index'][idx]
            target=f['actions'][idx][np.arange(16),index].copy()
        dim=target.shape[-1];g=all_g[idx]
        conf=json.loads((Path(case['teacher_run'])/'config.json').read_text());cfg=dict(conf['identity']['config'])
        for k in ('actor_hidden_dims','value_hidden_dims'):cfg[k]=tuple(cfg[k])
        assert sha(case['teacher_checkpoint'])==case['teacher_checkpoint_sha256']
        template=flat.create(0,jnp.asarray(x[:1]),jnp.zeros((1,dim)),cfg,conf['identity']['recipe'])
        with Path(case['teacher_checkpoint']).open('rb')as f:saved=pickle.load(f)
        teacher=flax.serialization.from_state_dict(template,saved['state'])
        agent=teacher.base.agent
        q=jax.jit(lambda obs,act:agent.network.select('critic')(obs,actions=act))
        errors=[]
        for epsilon in (.001,.0003):
            derivatives=[]
            for d in range(dim):
                plus=target.copy();minus=target.copy();plus[:,d]+=epsilon;minus[:,d]-=epsilon
                derivatives.append((np.asarray(q(jnp.asarray(x),jnp.asarray(plus)))-
                                    np.asarray(q(jnp.asarray(x),jnp.asarray(minus))))/(2*epsilon))
            fd=np.stack(derivatives,axis=-1).transpose(1,0,2)
            relative=float(np.linalg.norm(fd-g)/(np.linalg.norm(g)+1))
            assert relative<=.05,(name,epsilon,relative)
            errors.append(dict(epsilon=epsilon,relative_Frobenius_error=relative,
                max_absolute_error=float(np.max(np.abs(fd-g)))))
        normalized=np.asarray(metric.normalized_gradients(jnp.asarray(g),jnp.asarray(scale,jnp.float32)))
        assert np.isfinite(normalized).all()and np.max(np.sum(normalized**2,-1))<=10*dim+1e-4
        rows.append(dict(case=name,states=16,heads=2,finite_difference_checks=errors,
            training_only_scale_recomputed_exact=True,gradient_cache_sha256=sha(gp)))
    with a.output.open('x')as f:json.dump(dict(time=time.time(),status='pass',rows=rows,
        source_sha256=sha(__file__),proposal_sha256=sha(a.study/'proposal_v1.json'),
        no_model_updates=True,no_performance_claim=True,
        caveat='Checks derivatives of the learned critic, including its mathematical extension outside action bounds for central differences; not a true-value derivative oracle.'),f,indent=2,allow_nan=False)
    print(json.dumps(rows))


if __name__=='__main__':main()
