"""Audit the registered six coupled V-minibatch pilots without new learning.

Verify raw late CPU50 episodes, actual saved optimizer clocks and paired replay
streams/BC parameters. Old seed3 is a control source, not an extra replication.
"""
import argparse
import fcntl
import json
from pathlib import Path
import pickle
import statistics
import os
import time
from utils.flow_iql_research import core
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier
import numpy as np


STEPS=(800000,900000,1000000)


def atomic(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)


def identical(a,b):
    if isinstance(a,dict):
        return isinstance(b,dict) and set(a)==set(b) and all(identical(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):
        return isinstance(b,type(a)) and len(a)==len(b) and all(identical(x,y)for x,y in zip(a,b))
    return np.array_equal(np.asarray(a),np.asarray(b))


def tree_delta(a,b):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b)
        children=[tree_delta(a[k],b[k])for k in a]
    elif isinstance(a,(tuple,list)):
        assert isinstance(b,type(a)) and len(a)==len(b)
        children=[tree_delta(x,y)for x,y in zip(a,b)]
    else:
        aa,bb=np.asarray(a),np.asarray(b)
        assert aa.shape==bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
        delta=aa.astype(np.float64)-bb.astype(np.float64)
        return dict(elements=int(aa.size),different_elements=int(np.count_nonzero(delta)),
                    max_abs=float(np.max(np.abs(delta)))if delta.size else 0.,squared_delta=float(np.sum(delta*delta)))
    return dict(elements=sum(c['elements']for c in children),
                different_elements=sum(c['different_elements']for c in children),
                max_abs=max((c['max_abs']for c in children),default=0.),squared_delta=sum(c['squared_delta']for c in children))


def check_identity(config,expected,sources):
    assert config['identity']==expected,'Registered run identity differs'
    assert config['source_sha256']==sources,'Registered training sources differ'


def audit_run(run,expected,sources,verifier,require_value_replay=True):
    config_path=run/'config.json'
    if not config_path.exists():return dict(status='not_started',run=run.name,last3_mean=None)
    config=json.loads(config_path.read_text());check_identity(config,expected,sources)
    for path,digest in sources.items():assert verifier.digest(path)==digest,path
    assert expected['seed']==3 and expected['kind']=='ifql'
    assert expected['steps']==1000000 and expected['episodes']==50
    assert expected['config']['num_samples']==9 and expected['config']['discount']==.999 and expected['config']['expectile']==.9
    assert not(run/'stopped.json').exists(),'Stopped run is ineligible'
    done_path=run/'completed.json'
    if not done_path.exists():return dict(status='training_or_evaluating',run=run.name,last3_mean=None)
    done=json.loads(done_path.read_text())
    assert done['step']==done['q_updates']==1000000 and done['v_updates']==2000000
    records=[json.loads(line)for line in(run/'evaluations.jsonl').read_text().splitlines()]
    per_step={};pins={};episode_pins={}
    for step in STEPS:
        found=[row for row in records if row['step']==step];assert len(found)==1
        row=found[0]
        assert row['n_episodes']==50 and row['first_episode_seed']==42 and row['eval_device']=='cpu'
        assert row['protocol']=='last3_cpu50_independent_seeded_episodes_v1'
        for key in ('episode_successes','episode_rewards','episode_lengths'):
            assert len(row[key])==50 and np.isfinite(row[key]).all()
        assert set(row['episode_successes'])<={0.,1.}
        assert abs(statistics.mean(row['episode_successes'])-row['success_rate'])<1e-12
        per_step[str(step)]=row['success_rate']
        cp=run/f'checkpoint_{step}.pkl';pins[str(step)]=verifier.checkpoint(cp,config,step)
        if require_value_replay:
            with cp.open('rb')as stream:saved=pickle.load(stream)
            assert 'value_replay_rng'in saved
            rng=np.random.RandomState();rng.set_state(saved['value_replay_rng'])
        episode_pins[str(step)]={key:row[key]for key in ('episode_successes','episode_rewards','episode_lengths')}
    probes=[json.loads(line)for line in(run/'diagnostics.jsonl').read_text().splitlines()]
    assert probes and probes[-1]['step']==1000000 and all(p['finite']for p in probes)
    assert all(p['q_updates']==p['step'] and p['v_updates']==2*p['step']for p in probes)
    assert all(np.isfinite(p['critic/critic_loss']) and p['critic/critic_loss']>=0 for p in probes)
    late_probe={str(p['step']):dict(Q_RMS=p['q_rms'],V_RMS=p['v_rms'],
        unprojected_fixed_query_TD_RMS=p['td_rms'],samplewise_expectile_score_RMS=p['score_rms'],
        actual_online_window_projected_critic_RMS=float(np.sqrt(p['critic/critic_loss']/2)),
        post_fit_main_minibatch_expectile_objective=p['value/post_fit_loss'])for p in probes if p['step']in STEPS}
    assert len(late_probe)==3
    return dict(status='complete',run=run.name,last3_mean=statistics.mean(per_step.values()),per_step=per_step,
        environment=expected['environment'],learner_seed=3,checkpoint_sha256=pins,episodes=episode_pins,
        config_sha256=verifier.digest(config_path),evaluations_sha256=verifier.digest(run/'evaluations.jsonl'),
        diagnostics_sha256=verifier.digest(run/'diagnostics.jsonl'),probes=late_probe,
        all_observed_state_probes_finite=True,actual_Q_Adam_counts=STEPS,actual_V_Adam_counts=[2*s for s in STEPS])


def compare_models(left,right,left_row,right_row,require_value_replay):
    steps=[]
    for step in STEPS:
        with(left/f'checkpoint_{step}.pkl').open('rb')as stream:a=pickle.load(stream)
        with(right/f'checkpoint_{step}.pkl').open('rb')as stream:b=pickle.load(stream)
        assert identical(a['replay_rng'],b['replay_rng']),'Main Q/BC replay streams differ'
        if require_value_replay:
            assert identical(a['value_replay_rng'],b['value_replay_rng']),'Private V replay streams differ'
        assert identical(a['state']['agent']['rng'],b['state']['agent']['rng']),'Learner RNG differs'
        params_a=a['state']['agent']['network']['params'];params_b=b['state']['agent']['network']['params']
        bc=tree_delta(params_a['modules_actor_flow'],params_b['modules_actor_flow'])
        bc['RMS_difference']=float(np.sqrt(bc['squared_delta']/bc['elements']))
        full=tree_delta(a['state'],b['state'])
        arrays_exact=identical(left_row['episodes'][str(step)],right_row['episodes'][str(step)])
        if full['different_elements']==0:assert arrays_exact,'Identical learner states produced different seeded evaluations'
        steps.append(dict(step=step,BC_parameter_difference=bc,full_state_difference=full,
            Q_BC_replay_exact=True,private_V_replay_exact=True if require_value_replay else None,
            learner_RNG_exact=True,episode_arrays_exact=arrays_exact))
    return dict(steps=steps,all_BC_parameters_exact=all(s['BC_parameter_difference']['different_elements']==0 for s in steps),
        full_states_exact=all(s['full_state_difference']['different_elements']==0 for s in steps),
        all_episode_arrays_exact=all(s['episode_arrays_exact']for s in steps))


class Auditor:
    def __init__(self,registration):
        self.registration=registration
        self.plan=json.loads(Path(registration['training_registration']).read_text())
        self.root=Path(self.plan['root']);self.verifier=Verifier();self.complete={};self.pairs={};self.stamps={}
        assert self.verifier.digest(registration['training_registration'])==registration['training_registration_sha256']
        self.old_plan=json.loads(Path(registration['old_training_registration']).read_text())
        assert self.verifier.digest(registration['old_training_registration'])==registration['old_training_registration_sha256']
    def run(self,name,identity,sources,extra):
        run=self.root/'runs'/name
        def stamp():
            files=[run/f for f in ('config.json','completed.json','evaluations.jsonl','diagnostics.jsonl')]
            files += [run/f'checkpoint_{s}.pkl' for s in STEPS]
            return tuple((p.stat().st_dev,p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)for p in files)
        if name not in self.complete or stamp()!=self.stamps[name]:
            row=audit_run(run,identity,sources,self.verifier,extra)
            if row['status']=='complete':
                self.complete[name]=row;self.stamps[name]=stamp()
            return row
        return self.complete[name]
    def poll(self):
        for path,digest in self.registration['source_sha256'].items():assert self.verifier.digest(path)==digest,path
        rows={};tasks=[]
        for job in self.plan['jobs']:
            name=job['id'];rows[name]=self.run(name,self.plan['expected_identities'][name],self.plan['expected_training_sources'],True)
        for task in self.plan['tasks']:
            selected=[j for j in self.plan['jobs']if j['id'].startswith(f'vs1_{task}_')]
            assert len(selected)==2
            repeat=next(j for j in selected if j['args']['profile']=='bounded_v2_repeated_batches')
            fresh=next(j for j in selected if j['args']['profile']=='bounded_v2_fresh_batches')
            old=f'ieconfirm_s3_{task}_ifql_bounded_v2_g999_t90_N9'
            if all(rows[j['id']]['status']=='complete'for j in selected):
                oldrow=self.run(old,self.old_plan['expected_identities'][old],self.old_plan['training_source_sha256'],False)
                assert oldrow['status']=='complete'
                key=(repeat['id'],fresh['id'],self.stamps[repeat['id']],self.stamps[fresh['id']],self.stamps[old])
                if key not in self.pairs:
                    self.pairs[key]=dict(fresh_vs_repeated=compare_models(self.root/'runs'/fresh['id'],self.root/'runs'/repeat['id'],rows[fresh['id']],rows[repeat['id']],True),
                        repeated_vs_old_control=compare_models(self.root/'runs'/repeat['id'],self.root/'runs'/old,rows[repeat['id']],oldrow,False))
                tasks.append(dict(task=task,status='complete',repeated_last3=rows[repeat['id']]['last3_mean'],
                    fresh_last3=rows[fresh['id']]['last3_mean'],old_control_last3=oldrow['last3_mean'],
                    fresh_minus_repeated=rows[fresh['id']]['last3_mean']-rows[repeat['id']]['last3_mean'],
                    paired_parameter_and_stream_checks=self.pairs[key]))
            else:tasks.append(dict(task=task,status='incomplete',repeated_last3=None,fresh_last3=None,fresh_minus_repeated=None))
        complete=all(t['status']=='complete'for t in tasks)
        gate=None
        if complete:
            gate=bool(statistics.mean(t['fresh_minus_repeated']for t in tasks)>0 and sum(t['fresh_minus_repeated']>=0 for t in tasks)>=2)
        return dict(time=time.time(),status='complete'if complete else'incomplete',rows=rows,tasks=tasks,
            completed_runs=sum(r['status']=='complete'for r in rows.values()),
            completed_tasks=sum(t['status']=='complete'for t in tasks),
            three_pilot_mean_difference=statistics.mean(t['fresh_minus_repeated']for t in tasks)if complete else None,
            registered_performance_gate_passed=gate,nine_task_mean=None,root_review_required=True,
            clean_QV_only_control=all(t['paired_parameter_and_stream_checks']['fresh_vs_repeated']['all_BC_parameters_exact']for t in tasks)if complete else None,
            actual_unique_models_verified=len(self.verifier.checkpoints),new_GPU_jobs=0,new_CPU_evaluations=0,
            primary_candidate_unchanged=True,goal_complete=False,limits=self.registration['limits'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('registration',type=Path);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    registration=json.loads(args.registration.read_text());out=Path(registration['output']);out.mkdir(exist_ok=False)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    auditor=Auditor(registration)
    while True:
        try:
            result=auditor.poll()
        except(AssertionError,KeyError,OSError,ValueError)as error:
            atomic(out/'status.json',dict(time=time.time(),pid=os.getpid(),status='audit_failure_requires_inspection',error=repr(error),goal_complete=False))
            raise
        result.update(pid=os.getpid(),registration_sha256=auditor.verifier.digest(args.registration),
            source_sha256=auditor.verifier.digest(__file__))
        atomic(out/'status.json',result)
        if result['status']=='complete':atomic(out/'summary.json',result)
        if not args.watch or result['status']=='complete':return
        time.sleep(60)


if __name__=='__main__':main()
