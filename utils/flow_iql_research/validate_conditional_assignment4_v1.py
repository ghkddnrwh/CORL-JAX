"""CPU mathematical checks before any OT student fitting."""
from itertools import product
import json
from pathlib import Path
import sys
import time

from utils.flow_iql_research import prototype_statewise_OT4_v1 as proto
from utils.flow_iql_research import conditional_assignment4_v1 as assignment
from utils.flow_iql_research.evaluate_onestep_deploy_v1 import sha

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment


def main():
    root=Path(sys.argv[1]);area=root/'analysis/one_step_OT_CPU_v1';area.mkdir(exist_ok=False)
    assert all(d.platform=='cpu'for d in jax.devices())
    x=jax.random.normal(jax.random.PRNGKey(0),(4,5,3));y=jax.random.normal(jax.random.PRNGKey(1),(4,5,3))
    matched,index,cost=assignment.match(x,y)
    assert np.array_equal(np.sort(np.asarray(index),axis=0),np.tile(np.arange(4)[:,None],(1,5)))
    assert np.allclose(np.asarray(matched).sum(0),np.asarray(y).sum(0),atol=1e-6)
    errors=[]
    for b in range(5):
        c=np.mean((np.asarray(x[:,None,b])-np.asarray(y[None,:,b]))**2,axis=-1)
        rows,cols=linear_sum_assignment(c);errors.append(abs(float(cost[b])-float(c[rows,cols].mean())))
    assert max(errors)<1e-6
    value,grad=jax.value_and_grad(lambda p:proto.objective('action_OT4',p,x,y))(x)
    expected=2*(x-matched)/x.size
    grad_error=float(jnp.max(jnp.abs(grad-expected)));assert grad_error<1e-6
    assert abs(float(value)-float(cost.mean()))<1e-6
    # Exact expectation over4iidbinary draws, target law±1withprobability.5.
    values=jnp.asarray(list(product([-1.,1.],repeat=4)))
    total=.0
    for row in values:
        for target in values:
            total+=float(proto.objective('energy4',row[:,None,None],row[:,None,None],target[:,None,None]))/256
    assert abs(total-.5)<1e-6,total
    zero=jnp.zeros((4,1,1));target=jnp.asarray([-1.,-1.,1.,1.])[:,None,None]
    dirac=float(proto.objective('energy4',zero,zero,target));assert dirac==1
    result=dict(time=time.time(),status='pass',per_state_assignment_matches_SciPy_max_error=max(errors),
        target_multiset_and_all4noise_samples_preserved_per_state=True,action_OT_envelope_gradient_max_error=grad_error,
        energy4_binary_target_population_score=total,energy4_Dirac0_score=dirac,
        no_optimizer_GPUTeacher_or_actor_updates=True,no_performance_claim=True,
        source_sha256={str(Path(p).resolve()):sha(p)for p in [__file__,assignment.__file__,proto.__file__]})
    (area/'math_preflight_v1.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
