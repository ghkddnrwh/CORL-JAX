"""Evaluate a retained model with the primary policy/reset code, without editing a run.

Outputs are auxiliary evidence. They never overwrite or replace primary training
evaluations. Existing primary records can be compared episode by episode.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from utils.flow_iql_research import core,policy_core,benchmark_core,mixture_core
from utils.flow_iql_research import train as primary
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.propagation_research.seeded_reset import prepare_seeded_action_space


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=50)
    args=parser.parse_args();run=args.run
    if args.output.exists():raise FileExistsError('Preserve the existing frozen result')
    conf=json.loads((run/'config.json').read_text());ident=conf['identity'];kind=ident['kind']
    for source,expected in conf['source_sha256'].items():assert sha(source)==expected,source
    cfg=dict(ident['config'])
    for name in ('actor_hidden_dims','value_hidden_dims'):cfg[name]=tuple(cfg[name])
    module=benchmark_core if kind in ('fql','rebrac') else mixture_core if kind in ('gaussian_bc','mixture_bc') else core if kind=='iql' and 'policy_type' not in cfg else policy_core
    env=ogbench.make_env_and_datasets(ident['environment'],env_only=True)
    reset=prepare_seeded_action_space(env,42)
    device=jax.devices('cpu')[0]
    template=module.create(ident['seed'],jnp.zeros((1,env.observation_space.shape[-1])),
                           jnp.zeros((1,env.action_space.shape[-1])),cfg,kind)
    with args.checkpoint.open('rb') as f:saved=pickle.load(f)
    state=jax.device_put(flax.serialization.from_state_dict(template,saved['state']),device)
    assert int(state.q_updates)==saved['step']
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
    result=primary.evaluate(state,env,primary.make_policy(state.agent),saved['step'],args.episodes)
    es=run/'evaluations.jsonl';matches=[]
    if es.exists():
        matches=[json.loads(line) for line in es.read_text().splitlines() if json.loads(line)['step']==saved['step']]
    assert len(matches)<=1
    comparison=None
    if matches:
        old=matches[0]
        comparison={name:old.get(name)==result.get(name) for name in ('success_rate','episode_successes','episode_rewards','episode_lengths')}
    result.update(kind='auxiliary_frozen_evaluation_same_primary_protocol',run=run.name,
                  environment=ident['environment'],training_seed=ident['seed'],time=time.time(),
                  checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
                  config_sha256=sha(run/'config.json'),evaluation_helper_source_sha256=sha(primary.__file__),
                  evaluator_source_sha256=sha(__file__),reset_metadata=reset,
                  comparison_with_existing_primary=comparison,
                  cpu_affinity=sorted(__import__('os').sched_getaffinity(0)),
                  primary_evaluations_modified=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(args.output)
    print(json.dumps(dict(run=run.name,step=result['step'],success=result['success_rate'],seconds=result['seconds'],primary_comparison=comparison)),flush=True)


if __name__=='__main__':main()
