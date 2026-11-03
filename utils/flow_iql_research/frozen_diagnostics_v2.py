"""Independent validation and frozen-Q V fitting; never changes a training run.

Heldout expectile objectives contain one observed action per state. They are not
oracle conditional expectile residuals. Frozen fitting is an auxiliary CPU
counterfactual, not an additional benchmark update or a simulated GPU trajectory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core,policy_core
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
import optax


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--step',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args();run=args.run
    if args.output.exists():return
    conf=json.loads((run/'config.json').read_text());ident=conf['identity'];kind=ident['kind']
    assert kind not in ('fql','rebrac')
    for source,expected in conf['source_sha256'].items():assert sha(source)==expected,source
    cfg=dict(ident['config'])
    for name in ('actor_hidden_dims','value_hidden_dims'):cfg[name]=tuple(cfg[name])
    module=core if kind=='iql' and 'policy_type' not in cfg else policy_core
    _,raw,val=ogbench.make_env_and_datasets(ident['environment'])
    device=jax.devices('cpu')[0]
    template=module.create(ident['seed'],jnp.zeros((1,raw['observations'].shape[-1])),
                           jnp.zeros((1,raw['actions'].shape[-1])),cfg,kind)
    cp=args.checkpoint or run/f'checkpoint_{args.step}.pkl'
    with cp.open('rb') as f:saved=pickle.load(f)
    assert saved['step']==args.step
    state=jax.device_put(flax.serialization.from_state_dict(template,saved['state']),device)
    net=state.agent.network
    rng=np.random.RandomState(ident['seed']+87111)
    fields=('observations','actions','next_observations','rewards','masks')
    def sample(data,size):
        indices=rng.randint(len(data['actions']),size=size)
        b={k:np.asarray(data[k][indices],np.float32) for k in fields}
        b['actions']=np.clip(b['actions'],-1+1e-5,1-1e-5)
        return {k:jax.device_put(v,device) for k,v in b.items()}
    fitting=sample(raw,256);heldtrain=sample(raw,4096);heldval=sample(val,4096)
    boundsdata=json.loads((run/'dataset.json').read_text())
    bounds=(boundsdata['return_lower'],boundsdata['return_upper'])
    @jax.jit
    def targetq(b):return net.select('target_critic')(b['observations'],actions=b['actions']).min(0)
    sets={'same_minibatch':fitting,'independent_train':heldtrain,'validation':heldval}
    targetqs={k:targetq(b) for k,b in sets.items()}
    @jax.jit
    def losses(vp,b,q):
        v=net.select('value')(b['observations'],params={**net.params,'modules_value':vp})
        diff=q-v;w=jnp.where(diff>=0,cfg['expectile'],1-cfg['expectile'])
        return w*diff**2,w*diff,v
    tx=optax.adam(cfg['lr']);vp=net.params['modules_value'];vo=state.v_opt
    if ident['profile']=='official':
        # Reconstruct V's actual joint-Adam moments and clock, not a fresh Adam.
        adam=net.opt_state[0]
        vadam=type(adam)(count=adam.count,mu=adam.mu['modules_value'],nu=adam.nu['modules_value'])
        vo=(vadam,*net.opt_state[1:])
    initial_clock=int(vo[0].count)
    expected_clock=args.step*(2 if 'v2' in ident['profile'] else 1)
    assert initial_clock==expected_clock,(initial_clock,expected_clock)
    @jax.jit
    def fit(vp,vo):
        def objective(p):return losses(p,fitting,targetqs['same_minibatch'])[0].mean()
        gradients=jax.grad(objective)(vp);updates,vo=tx.update(gradients,vo,vp)
        return optax.apply_updates(vp,updates),vo
    rms=lambda x:float(np.sqrt(np.mean(np.asarray(x,np.float64)**2)))
    @jax.jit
    def backup(vp,b):
        vn=net.select('value')(b['next_observations'],params={**net.params,'modules_value':vp})
        y=b['rewards']+cfg['discount']*b['masks']*vn
        return y,jnp.clip(y,*bounds)
    baseline_losses={};baseline_backup={};trace=[]
    for n in range(9):
        if n in (0,1,2,4,8):
            entry={'extra_frozen_v_steps':n,'sets':{}}
            for name,b in sets.items():
                loss,score,values=[np.asarray(x,np.float64) for x in jax.device_get(losses(vp,b,targetqs[name]))]
                y,by=[np.asarray(x,np.float64) for x in jax.device_get(backup(vp,b))]
                if n==0:baseline_losses[name]=loss;baseline_backup[name]=y
                delta=loss-baseline_losses[name]
                entry['sets'][name]=dict(expectile_objective=float(loss.mean()),
                    objective_change_from_snapshot=float(delta.mean()),
                    fraction_rows_objective_increased=float(np.mean(delta>0)),
                    score_mean=float(score.mean()),score_rms=rms(score),v_rms=rms(values),
                    backup_rms=rms(y),backup_shift_rms=rms(y-baseline_backup[name]),
                    backup_clip_fraction=float(np.mean(y!=by)),backup_clip_rms=rms(y-by))
            trace.append(entry)
        if n<8:vp,vo=fit(vp,vo)
    probe_alpha=cfg['flow_temperature'] if kind=='awrflow' else cfg['alpha'] if kind=='iql' else None
    probes={name:module.probe_summary(jax.device_get(module.probe(state,b)),cfg['expectile'],bounds,probe_alpha) for name,b in sets.items() if name!='same_minibatch'}
    result=dict(run=run.name,step=args.step,kind='auxiliary_frozen_Q_V_fit_CPU_diagnostic',
                checkpoint_sha256=sha(cp),config_sha256=sha(run/'config.json'),
                diagnostic_source_sha256=sha(__file__),time=time.time(),initial_v_Adam_count=initial_clock,
                final_auxiliary_v_Adam_count=int(vo[0].count),trace=trace,snapshot_probes=probes,
                caveats=['Single-pair heldout objectives are not true conditional expectile residuals.',
                         'Repeated fitting keeps Q frozen and uses one minibatch; it does not reproduce coupled training.',
                         'This uses CPU floating point and is not an exact GPU update replay.',
                         'No checkpoint, benchmark counters, replay RNG or primary evaluation is modified.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(args.output)
    print(json.dumps(dict(run=run.name,step=args.step,initial_clock=initial_clock,
                         validation_loss_changes={r['extra_frozen_v_steps']:r['sets']['validation']['objective_change_from_snapshot'] for r in trace})),flush=True)


if __name__=='__main__':main()
