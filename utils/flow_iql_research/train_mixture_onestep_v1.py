"""Joint one-step actor training over unchanged lean, with actor-only exports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import mixture_onestep_deploy_core_v1 as core
from utils.propagation_research.reward_bounds import return_bounds
import flax.serialization
import jax
import numpy as np
import ogbench


STEPS=(800000,900000,1000000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    os.replace(temporary,path)


def append(path,value):
    with path.open('a') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')


def sync(tree):
    for value in jax.tree_util.tree_leaves(tree):
        if hasattr(value,'block_until_ready'):
            value.block_until_ready()


def configuration(environment,recipe,seed,steps=1000000,chunk=100,probe_every=1000,train_device='gpu'):
    cfg=core.config_for(environment,recipe)
    return json.loads(json.dumps(dict(environment=environment,recipe=recipe,config=cfg,
        specification=cfg['onestep_specification'],seed=seed,steps=steps,chunk=chunk,probe_every=probe_every,
        training_device=train_device,evaluation_steps=list(STEPS),episodes=50,
        evaluation='last3_cpu50_independent_seeded_episodes_v1',evaluation_execution='asynchronous_CPU_fixed_checkpoints',
        online_metrics='all_updates_in_each_probe_window_v2')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--env',choices=core.ENVIRONMENTS,required=True)
    parser.add_argument('--recipe',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1000000)
    parser.add_argument('--stop-at',type=int)
    parser.add_argument('--chunk',type=int,default=100)
    parser.add_argument('--probe-every',type=int,default=1000)
    parser.add_argument('--train-device',choices=('cpu','gpu'),default='gpu')
    args=parser.parse_args()
    began=time.time()
    start=time.monotonic()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    identity=configuration(args.env,args.recipe,args.seed,args.steps,args.chunk,args.probe_every,args.train_device)
    cfg=core.config_for(args.env,args.recipe)
    sources={str(p):sha(p) for p in core.source_paths()+[Path(__file__).resolve()]}
    if (out/'config.json').exists():
        previous=json.loads((out/'config.json').read_text())
        assert previous['identity']==identity and previous['source_sha256']==sources
    else:
        atomic(out/'config.json',dict(identity=identity,source_sha256=sources,
            arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}))
        folder=out/'source'
        folder.mkdir()
        for name in sources:
            p=Path(name)
            (folder/('vendor_'+p.name if p.parent.name=='vendor' else p.name)).write_bytes(p.read_bytes())
    if (out/'completed.json').exists():
        return
    device=jax.devices(args.train_device)[0]
    env,raw,validation=ogbench.make_env_and_datasets(args.env)
    data={key:np.asarray(raw[key],np.float32) for key in ('observations','actions','next_observations','rewards','masks')}
    data['actions']=np.clip(data['actions'],-1+1e-5,1-1e-5)
    interval=return_bounds(args.env,.999,float(data['rewards'].min()),float(data['rewards'].max()),'ogbench')
    atomic(out/'dataset.json',dict(size=len(data['actions']),reward_min=float(data['rewards'].min()),reward_max=float(data['rewards'].max()),
        return_lower=interval[0],return_upper=interval[1],normalize_states=False,action_clip_eps=1e-5,
        observation_dim=data['observations'].shape[-1],action_dim=data['actions'].shape[-1]))
    env.close()
    del raw,validation
    state=core.create(args.seed,jax.device_put(data['observations'][:1],device),
        jax.device_put(data['actions'][:1],device),cfg,args.recipe)
    metadata=core.actor_metadata(cfg,data['observations'].shape[-1],data['actions'].shape[-1])
    rng=np.random.RandomState(args.seed)
    latest=out/'latest.pkl'
    step=0
    timings=dict(batch_prepare_seconds=0.,scan_seconds=0.,first_scan_seconds=0.,probe_seconds=0.,checkpoint_seconds=0.,
        updates=0,session_start=began)
    if latest.exists():
        with latest.open('rb') as stream:
            saved=pickle.load(stream)
        assert saved['identity']==identity and saved['source_sha256']==sources
        state=jax.device_put(flax.serialization.from_state_dict(state,saved['state']),device)
        rng.set_state(saved['replay_rng'])
        step=saved['step']
        timings=saved.get('timings',timings)
        path=out/'diagnostics.jsonl'
        if path.exists():
            records=[json.loads(line) for line in path.read_text().splitlines()]
            tail=[r for r in records if r['step']>step]
            if tail:
                atomic(out/f'resume_tail_{time.time_ns()}.json',tail)
                path.write_text(''.join(json.dumps(r)+'\n' for r in records if r['step']<=step))
        print('RESUME',step,flush=True)
    scan=core.make_scan(core.build_update(args.recipe,interval))
    probe=core.make_probe(args.recipe,interval)
    indices=np.random.RandomState(args.seed+19401).randint(len(data['actions']),size=1024)
    fixed={key:jax.device_put(value[indices],device) for key,value in data.items()}
    endpoint=args.stop_at or args.steps
    sums,count=None,0
    first_call=True
    window_scan,window_prepare=0.,0.

    @jax.jit
    def add_logs(total,logs):
        return {key:total[key]+value.sum() for key,value in logs.items()}

    def save():
        before=time.perf_counter()
        payload=dict(step=step,state=flax.serialization.to_state_dict(jax.device_get(state)),replay_rng=rng.get_state(),
            identity=identity,source_sha256=sources,timings=dict(timings))
        temp=latest.with_suffix('.tmp')
        with temp.open('wb') as stream:
            pickle.dump(payload,stream,protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp,latest)
        if step in (100000,300000,*STEPS) or step==endpoint:
            target=out/f'checkpoint_{step}.pkl'
            assert not target.exists() or sha(target)==sha(latest)
            if not target.exists():
                os.link(latest,target)
            actor_path=out/f'actor_{step}.pkl'
            assert not actor_path.exists()
            artifact=core.export_actor(state,metadata)
            artifact.update(identity=identity,source_sha256=sources,checkpoint_sha256=sha(target),
                config_sha256=sha(out/'config.json'),step=step)
            actor_temp=actor_path.with_suffix('.tmp')
            with actor_temp.open('wb') as stream:
                pickle.dump(artifact,stream,protocol=pickle.HIGHEST_PROTOCOL)
            actor_temp.replace(actor_path)
        elapsed=time.perf_counter()-before
        timings['checkpoint_seconds']+=elapsed
        append(out/'runtime_events.jsonl',dict(step=step,kind='checkpoint',seconds=elapsed,time=time.time()))

    while step<endpoint:
        n=min(args.chunk,endpoint-step)
        before=time.perf_counter()
        indices=rng.randint(len(data['actions']),size=(n,cfg['batch_size']))
        batches={key:jax.device_put(value[indices],device) for key,value in data.items()}
        sync(batches)
        prepared=time.perf_counter()-before
        before=time.perf_counter()
        state,logs=scan(state,batches)
        sync((state,logs))
        elapsed=time.perf_counter()-before
        if first_call:
            append(out/'runtime_events.jsonl',dict(step=step+n,kind='compile_inclusive_first_scan',seconds=elapsed,updates=n,time=time.time()))
            timings['first_scan_seconds']+=elapsed
            first_call=False
        timings['scan_seconds']+=elapsed
        timings['batch_prepare_seconds']+=prepared
        timings['updates']+=n
        window_scan+=elapsed
        window_prepare+=prepared
        step+=n
        if sums is None:
            sums={key:np.float32(0) for key in logs}
        sums=add_logs(sums,logs)
        count+=n
        if step%args.probe_every==0 or step==endpoint:
            before=time.perf_counter()
            values=jax.device_get(probe(state,fixed))
            stats=core.probe_summary(values,interval)
            stats.update({key:float(value)/count for key,value in jax.device_get(sums).items()})
            assert int(state.base.q_updates)==step
            assert int(state.base.v_updates)==0 and int(state.actor_updates)==step
            stats.update(step=step,q_updates=int(state.base.q_updates),v_updates=int(state.base.v_updates),
                actor_updates=int(state.actor_updates),
                elapsed_seconds=time.monotonic()-start,metric_window_updates=count,
                window_scan_seconds=window_scan,window_batch_prepare_seconds=window_prepare,
                shared_GPU_scan_ms_per_update=window_scan*1000/count)
            timings['probe_seconds']+=time.perf_counter()-before
            if not stats['finite']:
                save()
                atomic(out/'stopped.json',dict(reason='nonfinite_state',step=step,time=time.time()))
                return
            append(out/'diagnostics.jsonl',stats)
            atomic(out/'runtime.json',dict(time=time.time(),step=step,timings=timings,
                session_wall_seconds=time.monotonic()-start,phase='training',CPU_evaluation='asynchronous_from_actor_only_exports_and_saved_teacher_models'))
            if step%10000==0:
                print('TRAIN',step,'Q_RMS',stats['q_rms'],'SHARED_GPU_MS',stats['shared_GPU_scan_ms_per_update'],flush=True)
            sums,count={key:np.float32(0) for key in logs},0
            window_scan,window_prepare=0.,0.
        if step%100000==0 or step==endpoint:
            save()
    complete=step==args.steps
    atomic(out/('completed.json' if complete else 'screened.json'),dict(kind='complete' if complete else 'screened',
        step=step,q_updates=int(state.base.q_updates),v_updates=int(state.base.v_updates),actor_updates=int(state.actor_updates),
        time=time.time(),timings=timings,
        evaluation_status='separate_CPU_controller_required',training_process_wall_seconds=time.monotonic()-start))


if __name__=='__main__':main()
