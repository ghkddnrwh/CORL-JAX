"""Instrument exact canonical CPU50 replays without changing policy calls.

Dense Puzzle reward, discounted dense return, completion-time surrogate,
first success, remaining buttons, and action magnitudes are logged. The
canonical point's three episode arrays must exactly reproduce the sealed
reference. No optimizer, model, training reward, or primary result changes.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from utils.flow_iql_research import train_mean_empirical_backup_v1 as compatibility
import gymnasium as gym
import ogbench


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Trace(gym.Wrapper):
    def __init__(self,env,gamma):
        super().__init__(env);self.gamma=gamma;self.episodes=[]

    def reset(self,**kwargs):
        observation,info=self.env.reset(**kwargs)
        self.seed_value=kwargs.get('seed');self.rewards=[];self.successes=[];self.remaining=[]
        self.action_norm=[];self.saturation=[];self.initial_observation=np.asarray(observation).copy().tolist()
        self.first_action=None
        return observation,info

    def step(self,action):
        output=self.env.step(action);observation,reward,terminated,truncated,info=output
        success=float(info.get('success',0.));reward=float(reward)
        assert math.isfinite(reward)and math.isfinite(success)
        self.rewards.append(reward);self.successes.append(success)
        actual=self.env.unwrapped
        wrong=int(np.count_nonzero(actual._cur_button_states!=actual._target_button_states))
        self.remaining.append(wrong)
        self.action_norm.append(float(np.linalg.norm(action)))
        self.saturation.append(float(np.mean(np.abs(action)>=.999)))
        if self.first_action is None:self.first_action=np.asarray(action).copy().tolist()
        if terminated or truncated:
            weights=self.gamma**np.arange(len(self.rewards),dtype=np.float64)
            rewards=np.asarray(self.rewards,np.float64);successes=np.asarray(self.successes)
            first=next((i+1 for i,x in enumerate(self.successes)if x>0),None)
            self.episodes.append(dict(seed=self.seed_value,length=len(self.rewards),success=float(successes.max()),
                reward_sum=float(rewards.sum()),discounted_dense_return=float(weights@rewards),
                discounted_completion_time_surrogate=float(weights@np.where(successes>0,0.,-1.)),
                first_success_step=first,last_reward=self.rewards[-1],last_incorrect_buttons=wrong,
                terminated=bool(terminated),truncated=bool(truncated),
                reward_equals_negative_incorrect_all_steps=all(r==-w for r,w in zip(self.rewards,self.remaining)),
                rewards=self.rewards,instantaneous_successes=self.successes,incorrect_buttons=self.remaining,
                action_L2=self.action_norm,action_saturation_fraction=self.saturation,
                initial_observation=self.initial_observation,first_action=self.first_action,
                reward_histogram=dict(Counter(map(str,self.rewards)))))
        return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--case',choices=['q_actor','teacher','center'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();root=a.root.resolve();area=a.output.parent;area.mkdir(parents=True,exist_ok=True)
    rid='osm1_pz3_mdn9_bc_s0'if a.case=='center'else'osbq_pz3_bank_q_a1_s0'
    run=root/'runs'/rid;conf=json.loads((run/'config.json').read_text());gamma=conf['identity']['config']['discount']
    point=area/f'{a.case}_canonical_point_v1.json';assert not point.exists()
    if a.case=='center':
        module=importlib.import_module('utils.flow_iql_research.evaluate_portable_mixture_locations_v1')
        actor=root/f'analysis/evaluations_v1/routed_actors_v1/{rid}_1000000_component_location.pkl'
        args=['--run',str(run),'--step','1000000','--actor-output',str(actor),'--output',str(point)]
        reference=root/f'analysis/evaluations_v1/{rid}_1000000_component_location_n50.json'
    else:
        module=importlib.import_module('utils.flow_iql_research.evaluate_onestep_deploy_v1')
        mode='teacher'if a.case=='teacher'else'primary'
        args=['--run',str(run),'--step','1000000','--mode',mode,'--episodes','50','--output',str(point)]
        reference=root/f'analysis/evaluations_v1/{rid}_1000000_{mode}_n50.json'
    expected=json.loads(reference.read_text());reference_digest=sha(reference)
    original=ogbench.make_env_and_datasets;wrappers=[]
    def factory(*factory_args,**kwargs):
        assert kwargs.get('env_only')is True
        env=original(*factory_args,**kwargs);wrapper=Trace(env,gamma);wrappers.append(wrapper);return wrapper
    ogbench.make_env_and_datasets=factory
    sys.argv=[module.__file__,*args]
    module.main()
    assert len(wrappers)==1 and len(wrappers[0].episodes)==50
    point_data=json.loads(point.read_text());traces=wrappers[0].episodes
    exact={key:point_data[key]==expected[key]for key in ['episode_successes','episode_rewards','episode_lengths']}
    assert all(exact.values()),exact
    assert [ep['success']for ep in traces]==expected['episode_successes']
    assert [ep['reward_sum']for ep in traces]==expected['episode_rewards']
    assert [ep['length']for ep in traces]==expected['episode_lengths']
    assert all(ep['seed']==42+i for i,ep in enumerate(traces))
    assert sha(reference)==reference_digest
    result=dict(time=time.time(),protocol='read_only_Puzzle_dense_vs_completion_CPU50_diagnosis_v1',case=a.case,
        run=rid,step=1000000,training_seed=0,gamma=gamma,n_episodes=50,first_episode_seed=42,
        exact_reproduction_of_sealed_episode_arrays=exact,reference=str(reference),reference_sha256=reference_digest,
        canonical_instrumented_point=str(point),canonical_point_sha256=sha(point),episodes=traces,
        success_rate=expected['success_rate'],undiscounted_dense_return_mean=float(np.mean([ep['reward_sum']for ep in traces])),
        discounted_dense_return_mean=float(np.mean([ep['discounted_dense_return']for ep in traces])),
        discounted_completion_time_surrogate_mean=float(np.mean([ep['discounted_completion_time_surrogate']for ep in traces])),
        success_first_step_mean=float(np.mean([ep['first_success_step']for ep in traces if ep['success']])),
        reward_shape_verified_all_episodes=all(ep['reward_equals_negative_incorrect_all_steps']for ep in traces),
        optimizer_updates_performed=0,training_rewards_or_original_results_changed=False,
        source_sha256={str(Path(path).resolve()):sha(path)for path in [__file__,module.__file__,compatibility.__file__]},
        scope='Instrumented replay of three fixed1M policies, not new training replicates or full-nine evidence. Discounted realized policy return is not ground-truth error of the implicit-expectile critic.')
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False));temp.replace(a.output)
    print(json.dumps(dict(case=a.case,success=result['success_rate'],discounted_dense=result['discounted_dense_return_mean'],
        discounted_completion_time=result['discounted_completion_time_surrogate_mean'],episode_arrays_exact=True)),flush=True)


if __name__=='__main__':main()
