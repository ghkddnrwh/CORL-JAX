"""Rescore all Q snapshots on one common bank and the same CPU backend."""
import argparse
import json
from pathlib import Path

from utils.codex.ukjo1.error_propagation.flow_iql_research import sample_budget_core_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research.critic_stabilization_experiment_v1 import load_plan
from utils.codex.ukjo1.error_propagation.flow_iql_research.diagnose_sample_budget_components_v1 import atomic,read_source,sha
import jax
import jax.numpy as jnp
import numpy as np


def run(path,require_complete):
    plan=load_plan(path)
    assert all(d.platform=='cpu' for d in jax.devices())
    root=path.parent/'analysis/common_cpu_rankings'
    done,pending=[],[]
    for case in plan['cases']:
        anchor_job=next(j for j in plan['jobs'] if j['case']==case['id'])
        bankpath=Path(anchor_job['output'])/'ranking_bank.npz'
        if not bankpath.exists():
            pending.append(str(bankpath));continue
        with np.load(bankpath) as bank:
            obs,actions=jnp.asarray(bank['observations']),jnp.asarray(bank['actions'])
        for method in ('baseline','q_lr_0p1','q_cosine'):
            job=next((j for j in plan['jobs'] if (j['case'],j['method'])==(case['id'],method)),None)
            for step in (800000,900000,1000000):
                target=root/case['id']/f'{method}_{step}.npz'
                if target.exists():
                    done.append(str(target));continue
                run=Path(case['parent_run']) if method=='baseline' or step==800000 else Path(job['output'])
                cp=run/f'checkpoint_{step}.pkl'
                if not cp.exists():
                    pending.append(str(cp));continue
                if method!='baseline' and step==800000:
                    original=root/case['id']/f'baseline_{step}.npz'
                    assert original.exists()
                    target.write_bytes(original.read_bytes());done.append(str(target));continue
                source=dict(run=str(run),step=step,checkpoint=str(cp),checkpoint_sha256=sha(cp),config_sha256=sha(run/'config.json'))
                if method=='baseline':assert source['checkpoint_sha256']==case['parent_checkpoints'][str(step)]
                _,state=read_source(source)
                net=state.base.agent.network
                heads=jax.jit(lambda p:net.replace(params=p).select('critic')(obs,actions=actions),backend='cpu')(net.params)
                target.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,heads=np.asarray(heads),step=step,bank_sha256=sha(bankpath),
                    checkpoint_sha256=source['checkpoint_sha256'],scoring_backend='cpu')
                atomic(target.with_suffix('.json'),dict(case=case['id'],method=method,step=step,
                    checkpoint_sha256=source['checkpoint_sha256'],bank_sha256=sha(bankpath),
                    array_sha256=sha(target),scoring_backend='cpu',source_sha256=sha(__file__)))
                done.append(str(target));print(case['id'],method,step,flush=True)
    if require_complete:assert not pending,pending
    atomic(path.parent/'analysis/common_cpu_rankings_status_v1.json',dict(
        phase='complete' if not pending else 'partial',files=len(done),pending=pending,
        source_sha256=sha(__file__),scope='Same common 800K Flow action bank for all methods; all scores on CPU. '
            'Avoid comparing GPU training probes against CPU baseline probes. No rollout or method changes.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();run(a.plan.resolve(),a.require_complete)
