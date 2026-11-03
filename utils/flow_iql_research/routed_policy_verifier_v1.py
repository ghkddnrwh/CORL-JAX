"""Verify routed actor artifacts and policy rows against native learner pins."""
from fractions import Fraction
import json
import math
from pathlib import Path
import pickle
import statistics
import numpy as np
from utils.flow_iql_research.routed_policy_registry_v1 import POLICY,NATIVE,MODE,DEPLOYMENT_ID


def arrays(tree,prefix=''):
    if isinstance(tree,dict):
        for key in sorted(tree):yield from arrays(tree[key],prefix+'/'+str(key))
    elif isinstance(tree,(tuple,list)):
        for key,value in enumerate(tree):yield from arrays(value,prefix+'/'+str(key))
    elif tree is not None:yield prefix,np.asarray(tree)


def expected_metadata(original):
    return dict(original,architecture='routed_component_location_MLP',
        deployment_decoder='sampled_component_location',computed_action_heads=1,
        Dense_dot_operations=6,within_component_action_noise=False,
        output='sample_one_component_by_learned_probabilities_then_clip_its_location',
        parameter_layout='Native MDN weights retained exactly; unused scale-head columns are not evaluated.')


def actor_path(root,rid,step):
    return root/'analysis/evaluations_v1/routed_actors_v1'/f'{rid}_{step}_component_location.pkl'


class PolicyVerifier:
    def __init__(self,plan,native_verifier,sha):
        self.plan=plan;self.native=native_verifier;self.sha=sha;self.pins={}

    def model(self,run,conf,step,native_pin,override_actor=None):
        path=Path(override_actor) if override_actor else actor_path(Path(self.plan['root']),run.name,step)
        key=f'{run.name}:{step}' if override_actor is None else str(path)
        if key in self.pins:return self.pins[key]
        with path.open('rb') as stream:actor=pickle.load(stream)
        with Path(native_pin['actor_path']).open('rb') as stream:original=pickle.load(stream)
        assert actor['schema']=='routed_mixture_location_actor_v1'
        assert actor['training_identity']==original['identity']==conf['identity']
        assert actor['training_step']==actor['actor_updates']==actor['Q_updates']==step
        assert actor['no_new_training_updates']is True
        assert actor['native_actor_sha256']==native_pin['actor_artifact_sha256']
        assert Path(actor['native_actor_path']).resolve()==Path(native_pin['actor_path']).resolve()
        assert actor['training_checkpoint_sha256']==native_pin['checkpoint_sha256']
        assert actor['training_source_sha256']==conf['source_sha256']
        for source,digest in actor['deployment_source_sha256'].items():
            assert self.plan['all_source_sha256'][source]==self.sha(source)==digest
        assert actor['metadata']==expected_metadata(original['metadata'])
        left,right=dict(arrays(original['actor_params'])),dict(arrays(actor['actor_params']))
        assert left.keys()==right.keys()
        assert all(left[k].dtype==right[k].dtype and left[k].shape==right[k].shape and left[k].tobytes()==right[k].tobytes() for k in left)
        assert all(np.isfinite(v).all() for v in right.values())
        pin=dict(path=native_pin['path'],checkpoint_sha256=native_pin['checkpoint_sha256'],
            actor_path=str(path),actor_artifact_sha256=self.sha(path),
            native_actor_path=native_pin['actor_path'],native_actor_sha256=native_pin['actor_artifact_sha256'],
            actual_Q_Adam=step,actual_actor_Adam=step,V_Adam=0,exported_actor_params_exact=True,
            native_training_recipe=NATIVE,deployment_candidate=POLICY,metadata=actor['metadata'],
            actor_parameter_count=sum(v.size for v in right.values()))
        self.pins[key]=pin;return pin

    def evaluation(self,path,job,step,pin):
        row=json.loads(path.read_text());args=job['args'];rid=job['id']
        assert args['recipe']==row['native_training_recipe']==NATIVE
        assert row['run']==rid and row['training_seed']==args['seed'] and row['step']==step
        assert row['environment']==args['env']
        assert row['deployment_candidate']==DEPLOYMENT_ID and row['deployment_rule']=='sampled_component_location'
        assert row['protocol']=='common_routed_mixture_location_last3_CPU50_v1'
        assert row['n_episodes']==50 and row['first_episode_seed']==42 and row['eval_device']=='cpu'
        assert row['optimizer_updates_performed']==0 and row['no_production_results_replaced']is True
        assert row['learned_model_trees_loaded']==['actor_params'] and row['deployment_metadata']==pin['metadata']
        assert row['computed_action_heads']==1 and not row['candidate_action_bank']
        assert not row['requires_critic'] and not row['requires_BC_flow']
        assert row['actor_artifact_sha256']==pin['actor_artifact_sha256']
        assert row['native_actor_artifact_sha256']==pin['native_actor_sha256']
        assert row['config_sha256']==self.native.configs[rid]['config_sha256']
        expected=self.plan['routed_deployment_extension']['evaluation_source_sha256']
        assert row['source_sha256']==expected
        for source,digest in expected.items():assert self.sha(source)==digest
        assert all(len(row[name])==50 for name in ('episode_successes','episode_rewards','episode_lengths'))
        assert all(v in (0,1) for v in row['episode_successes'])
        assert all(math.isfinite(float(v)) for name in ('episode_rewards','episode_lengths') for v in row[name])
        assert all(float(v).is_integer() and v>0 for v in row['episode_lengths'])
        successes=sum(int(v) for v in row['episode_successes'])
        assert abs(row['success_rate']-successes/50)<1e-12
        assert abs(row['reward_mean']-statistics.mean(row['episode_rewards']))<1e-8
        return dict(successes=successes,episodes=50,success_rate=successes/50,sha256=self.sha(path),
                    shared_machine_CPU_action_ms=None,deployment_candidate=POLICY)


def derived_rows(root,jobs,native_rows,state,verifier,policy_verifier,result_path,evaluate_key):
    native_by_id={r['run']:r for r in native_rows};rows=[];ready=[];evaluations={}
    for job in jobs:
        if job['args']['recipe']!=NATIVE:continue
        rid=job['id'];native=native_by_id[rid];run=root/'runs'/rid
        row=dict(native,recipe=POLICY,native_training_recipe=NATIVE,
            deployment_rule='sampled_component_location',complete=False,primary_mean=None,
            per_step=dict(primary={},teacher=native.get('per_step',{}).get('teacher',{})),
            auxiliary_cpu10={},last_probe_scope='Native MDN training diagnostic, not routed-policy rollout.')
        counts=[]
        if not(run/'config.json').exists()or rid not in verifier.configs:
            rows.append(row);continue
        conf=verifier.config(run,state['identities'][rid])
        for step in (800000,900000,1000000):
            if not(run/f'actor_{step}.pkl').exists()or not(run/f'checkpoint_{step}.pkl').exists():continue
            native_pin=verifier.model(run,conf,step)
            path=result_path(rid,step,MODE,50);key=evaluate_key(rid,step,MODE,50)
            if path.exists():
                pin=policy_verifier.model(run,conf,step,native_pin)
                score=policy_verifier.evaluation(path,job,step,pin);evaluations[key]=score
                counts.append(score['successes']);row['per_step']['primary'][str(step)]=score['success_rate']
                row['actor_parameter_count']=pin['actor_parameter_count']
            elif key not in state['evaluation_workers'] and key not in state['evaluation_failures']:
                ready.append((job,step,MODE,50))
        alias_failure=any(k.startswith(rid+':')and f':{MODE}:' in k for k in state['evaluation_failures'])
        teacher_failure=any(k.startswith(rid+':')and ':teacher:' in k for k in state['evaluation_failures'])
        native_training_failed=native['terminal']and not native['training_complete']
        if alias_failure or teacher_failure:row.update(terminal=True,status='evaluation_failed')
        elif native_training_failed:pass
        elif native['training_complete']and len(counts)==3 and len(row['per_step']['teacher'])==3:
            row.update(complete=True,terminal=True,status='complete',primary_mean=float(Fraction(sum(counts),150)),
                       teacher_mean=statistics.mean(row['per_step']['teacher'].values()))
        else:row.update(terminal=False,status='evaluating' if native['training_complete'] else native['status'])
        rows.append(row)
    return rows,ready,evaluations
