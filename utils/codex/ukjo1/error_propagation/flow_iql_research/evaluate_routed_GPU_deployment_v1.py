"""Auxiliary real GPU-actor/CPU-environment rollout with a CPU reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from utils.flow_iql_research import train_mean_empirical_backup_v1 as evaluation
from utils.propagation_research.seeded_reset import prepare_seeded_action_space
from utils.flow_iql_research import standalone_mixture_location_actor_v1 as portable
import jax
import numpy as np
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--CPU-reference',type=Path,required=True);p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    assert os.environ['NVIDIA_TF32_OVERRIDE']=='0'and 1<=a.episodes<=50
    with a.actor.open('rb')as f:artifact=pickle.load(f)
    identity=artifact['training_identity'];reference=json.loads(a.CPU_reference.read_text())
    assert reference['actor_artifact_sha256']==sha(a.actor) and reference['step']==artifact['training_step']
    assert reference['first_episode_seed']==42 and reference['n_episodes']>=a.episodes
    gpu=portable.load_actor(a.actor,'gpu');cpu=portable.load_actor(a.actor,'cpu')
    assert gpu.device.platform=='gpu';digest=sha(a.actor);shadow=dict(states=0,max_abs_action_difference=0.,differences_above_1e4=0)
    def policy(_unused,obs,key):
        actual=gpu(obs,key)
        if shadow['states']<128:
            error=float(np.max(np.abs(np.asarray(actual)-np.asarray(cpu(obs,key)))))
            shadow['states']+=1;shadow['max_abs_action_difference']=max(shadow['max_abs_action_difference'],error)
            shadow['differences_above_1e4']+=int(error>1e-4)
        return actual
    state=SimpleNamespace(agent=SimpleNamespace(network=SimpleNamespace(params=gpu.params)))
    env=ogbench.make_env_and_datasets(identity['environment'],env_only=True);reset=prepare_seeded_action_space(env,42)
    result=evaluation.evaluate(state,env,policy,artifact['training_step'],a.episodes);env.close()
    comparison={key:result[key]==reference[key][:a.episodes]for key in ['episode_successes','episode_rewards','episode_lengths']}
    CPU_success=float(np.mean(reference['episode_successes'][:a.episodes]))
    result.update(time=time.time(),protocol='auxiliary_GPUactor_CPUenvironment_fixedseed_v1',
        deployment_rule='sampled_component_location',environment=identity['environment'],training_seed=identity['seed'],
        actor_artifact_sha256=digest,NVIDIA_TF32_OVERRIDE='0',policy_device='gpu',environment_device='cpu',
        GPU_kind=gpu.device.device_kind,CPU_shadow=shadow,CPU_reference=str(a.CPU_reference),
        CPU_reference_sha256=sha(a.CPU_reference),CPU_matched_episode_success_rate=CPU_success,
        GPU_minus_CPU_success_rate=result['success_rate']-CPU_success,episode_arrays_exact=comparison,
        reset_metadata=reset,primary_model_trees_loaded=['actor_params'],
        auxiliary_reference='Same actor parameters, resident CPU copy for128same-state shadow calls.',
        training_updates_performed=0,critic_or_Flow_loaded=False,isolated_latency_claim=False,
        counts_toward_nine_task_or_fresh_confirmation=False,
        scope='Auxiliary actual GPU deployment on one saved native actor while16learners continue. Fixed first episodes of the CPU50 reference; no nine-task, fresh-seed, or isolated-speed claim.')
    assert sha(a.actor)==digest
    result['source_sha256']={str(Path(path).resolve()):sha(path)for path in [__file__,portable.__file__,evaluation.__file__]}
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(a.output)
    print(json.dumps(dict(success=result['success_rate'],CPU_success=CPU_success,episodes=a.episodes,
        arrays_exact=comparison,shadow=shadow)))


if __name__=='__main__':main()
