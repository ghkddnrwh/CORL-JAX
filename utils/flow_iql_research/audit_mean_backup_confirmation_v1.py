"""Independent raw-episode/model audit of the locked27-run confirmation.

This verifier never trains, evaluates a policy, edits a primary result or marks
the thread goal complete. Original gates are recomputed from exact counts.
"""
import argparse
import fcntl
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import time
import numpy as np
from utils.flow_iql_research import tau_matched_extraction_v1 as helper
from utils.flow_iql_research.audit_temporal_cohort_v1 import Verifier,provided_thresholds
from utils.propagation_research.reward_bounds import REWARD_COMPONENTS


def records(path,complete=False):
    if not path.exists():return []
    text=path.read_text()
    if text and not text.endswith('\n'):
        assert not complete,f'Incomplete final JSONL line: {path}'
        text=text.rsplit('\n',1)[0]if '\n'in text else''
    return [json.loads(line)for line in text.splitlines()]


def episode_count(record,step):
    helper.check_episodes(record,step)
    assert math.isclose(statistics.mean(record['episode_rewards']),record['reward_mean'],rel_tol=1e-12,abs_tol=1e-9)
    assert all(float(n).is_integer()and n>0 for n in record['episode_lengths'])
    count=sum(int(s)for s in record['episode_successes'])
    assert abs(float(Fraction(count,50))-record['success_rate'])<1e-12
    return count


def fingerprint(tree):
    digest=hashlib.sha256()
    def visit(value):
        if isinstance(value,dict):
            for key in sorted(value):digest.update(str(key).encode()+b'\0');visit(value[key])
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        else:
            array=np.ascontiguousarray(value)
            digest.update(str(array.shape).encode()+array.dtype.str.encode()+array.tobytes())
    visit(tree);return digest.hexdigest()


class Auditor:
    def __init__(self,contract):
        self.contract=contract;self.verifier=Verifier();self.pins={};self.fingerprints={};self.stamps={}
        assert self.verifier.digest(contract['primary_registration'])==contract['primary_registration_sha256']
        self.plan=json.loads(Path(contract['primary_registration']).read_text());self.root=Path(self.plan['root'])
        assert self.plan['fresh_seeds']==[9,10,11]and len(self.plan['confirmation_jobs'])==27
        assert self.verifier.digest(contract['discovery_audit'])==contract['discovery_audit_sha256']
        self.discovery=json.loads(Path(contract['discovery_audit']).read_text());assert self.discovery['completed_tasks']==9
        thresholds,sha=provided_thresholds(Path(contract['criterion_file']))
        assert sha==contract['criterion_sha256']==self.discovery['criterion_sha256']
        self.threshold=Fraction(str(max(thresholds.values())))
        assert self.threshold==Fraction(str(self.plan['threshold']))==Fraction(47,100)
        self.thresholds=thresholds
        self.environments=[j['args']['env']for j in json.loads(Path(self.plan['discovery_registration']).read_text())['all9jobs']]
        assert len(self.environments)==len(set(self.environments))==9

    def config(self,run,expected,sources,require_snapshots=True):
        conf=json.loads((run/'config.json').read_text());assert conf['identity']==expected and conf['source_sha256']==sources
        cfg=expected['config'];assert expected['kind']=='ifql'and expected['profile']=='bounded_v2_mean_empirical_backup'
        assert cfg['discount']==.999 and cfg['expectile']==.9 and cfg['num_samples']==cfg['backup_samples']==9
        assert cfg['backup_Q_aggregation']=='mean'and cfg['backup_kind']=='finite_behavior_flow_mean_target_Q_expectile'
        width=256 if any(s in expected['environment']for s in('cube-','puzzle-','scene-'))else 512
        assert cfg['actor_hidden_dims']==cfg['value_hidden_dims']==[width]*4
        assert cfg['layer_norm']and not cfg['actor_layer_norm']and cfg['lr']==.0003 and cfg['tau']==.005 and cfg['batch_size']==256 and cfg['flow_steps']==10
        assert expected['steps']==1000000 and expected['episodes']==50
        for source,sha in sources.items():
            assert self.verifier.digest(source)==sha,source
            if require_snapshots:
                p=Path(source);name='vendor_'+p.name if p.parent.name=='vendor'else p.name
                assert self.verifier.digest(run/'source'/name)==sha,name
        dataset=json.loads((run/'dataset.json').read_text());components=REWARD_COMPONENTS[expected['environment']]
        assert math.isclose(dataset['return_lower'],-components/(1-.999),rel_tol=1e-12)and dataset['return_upper']==0
        assert -components<=dataset['reward_min']<=dataset['reward_max']<=0
        assert dataset['normalize_states']is False and dataset['action_clip_eps']==1e-5
        return conf

    def model(self,run,conf,step):
        path=run/f'checkpoint_{step}.pkl';sha=self.verifier.checkpoint(path,conf,step)
        stat=path.stat();stamp=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns);key=f'{run.name}:{step}'
        if self.stamps.get(key)!=stamp:
            with path.open('rb')as stream:saved=pickle.load(stream)
            assert self.verifier.digest(path)==sha
            network=saved['state']['agent']['network']
            self.fingerprints[key]=dict(BC_params=fingerprint(network['params']['modules_actor_flow']),
                agent_RNG=fingerprint(saved['state']['agent']['rng']),
                replay_RNG=hashlib.sha256(pickle.dumps(saved['replay_rng'],protocol=4)).hexdigest())
            if key in self.pins:assert self.pins[key]['sha256']==sha
            self.pins[key]=dict(path=str(path),sha256=sha,actual_Q_Adam=step,actual_V_Adam=2*step,actual_full_saved_state_finite=True)
            self.stamps[key]=stamp
        return key

    def discovery_count(self):
        total=0;models=0
        for row in self.discovery['rows']:
            run=self.root/'runs'/row['run'];assert self.verifier.digest(run/'config.json')==row['config_sha256']
            conf=json.loads((run/'config.json').read_text());self.config(run,conf['identity'],conf['source_sha256'])
            assert self.verifier.digest(run/'evaluations.jsonl')==row['evaluation_sha256']
            raw=records(run/'evaluations.jsonl',True)
            for step in helper.STEPS:
                selected=[e for e in raw if e['step']==step];assert len(selected)==1
                total+=episode_count(selected[0],step)
                assert self.verifier.digest(run/f'checkpoint_{step}.pkl')==row['checkpoint_sha256'][str(step)];models+=1
        assert models==27 and abs(float(Fraction(total,1350))-self.discovery['nine_task_mean'])<1e-12
        return total

    def poll(self):
        for source,sha in self.contract['source_sha256'].items():assert self.verifier.digest(source)==sha,source
        for source,sha in self.plan['source_sha256'].items():assert self.verifier.digest(source)==sha,source
        discovery_count=self.discovery_count();rows=[];contracts=0
        for job in self.plan['confirmation_jobs']:
            run=self.root/'runs'/job['id'];identity=self.plan['expected_identities'][job['id']]
            assert identity['seed']==job['args']['seed']in(9,10,11)and identity['environment']==job['args']['env']
            base=dict(run=run.name,environment=identity['environment'],seed=identity['seed'],complete=False,last3_mean=None,late_success_count=None)
            if not(run/'config.json').exists():rows.append(dict(**base,status='not_started'));continue
            expected_sources=self.plan['expected_training_sources'][run.name]
            # Source copies and dataset metadata are written before initialization;
            # a just-launched process can briefly expose only config.json.
            files=[run/'dataset.json']+[run/'source'/('vendor_'+Path(s).name if Path(s).parent.name=='vendor'else Path(s).name)for s in expected_sources]
            if not all(f.exists()for f in files):rows.append(dict(**base,status='initializing'));continue
            conf=self.config(run,identity,expected_sources);contracts+=1
            assert identity['evaluation_steps']==list(helper.STEPS)
            done_path=run/'completed.json';complete=done_path.exists();raw=records(run/'evaluations.jsonl',complete)
            assert all(e['step']in helper.STEPS for e in raw)
            per_step={};keys=[];counts={}
            for step in helper.STEPS:
                found=[e for e in raw if e['step']==step];assert len(found)<=1
                if found:
                    counts[str(step)]=episode_count(found[0],step);per_step[str(step)]=float(Fraction(counts[str(step)],50))
                    keys.append(self.model(run,conf,step))
                else:per_step[str(step)]=None
            stopped=json.loads((run/'stopped.json').read_text())if(run/'stopped.json').exists()else None
            probes=records(run/'diagnostics.jsonl',complete)
            assert all(p['finite']and p['q_updates']==p['step']and p['v_updates']==2*p['step']for p in probes)
            if complete:
                done=json.loads(done_path.read_text());assert done['kind']=='complete'and done['step']==done['q_updates']==1000000 and done['v_updates']==2000000
                assert not stopped and len(keys)==3 and len(counts)==3
                assert [p['step']for p in probes]==list(range(1000,1000001,1000))
            total=sum(counts.values())if complete else None
            rows.append(dict(**{k:v for k,v in base.items()if k not in('complete','last3_mean','late_success_count')},
                status='complete'if complete else'stopped'if stopped else'running',complete=complete,stopped=stopped,
                per_step=per_step,last3_mean=float(Fraction(total,150))if complete else None,late_success_count=total,
                observed_finite_probe_count=len(probes),config_sha256=self.verifier.digest(run/'config.json'),
                evaluations_sha256=self.verifier.digest(run/'evaluations.jsonl')if raw else None,
                diagnostics_sha256=self.verifier.digest(run/'diagnostics.jsonl')if probes else None,
                max_probe_Q_RMS_over_return_bound=max((p['q_rms_over_return_bound']for p in probes),default=None),
                max_probe_Q_above_upper_RMS=max((p['q_above_upper_rms']for p in probes),default=None)))
        seeds=[];seed_fractions=[]
        for seed in(9,10,11):
            chosen=[r for r in rows if r['seed']==seed];assert len(chosen)==9
            assert {r['environment']for r in chosen}==set(self.environments)
            ready=all(r['complete']for r in chosen);count=sum(r['late_success_count']for r in chosen)if ready else None
            fraction=Fraction(count,1350)if ready else None;seed_fractions.append(fraction)
            seeds.append(dict(seed=seed,completed_tasks=sum(r['complete']for r in chosen),nine_task_mean=float(fraction)if ready else None,
                exact_nine_task_mean=str(fraction)if ready else None,late_success_count=count))
        diversity=[]
        for env in self.environments:
            chosen=[r for r in rows if r['environment']==env];ready=all(r['complete']for r in chosen)
            if not ready:continue
            entries=[self.fingerprints[f'{r["run"]}:1000000']for r in chosen]
            checks={field:len({e[field]for e in entries})==3 for field in('BC_params','agent_RNG','replay_RNG')}
            assert all(checks.values()),(env,checks)
            diversity.append(dict(environment=env,three_seed_fingerprints_distinct=checks,
                fresh_task_mean=statistics.mean(r['last3_mean']for r in chosen),fresh_task_sample_SD=statistics.stdev(r['last3_mean']for r in chosen)))
        complete=all(r['complete']for r in rows);fresh=four=None;passed=None
        if complete:
            assert len(self.pins)==len(self.verifier.checkpoints)==81 and len(diversity)==9 and contracts==27
            fresh=sum(seed_fractions,Fraction(0))/3;four=(Fraction(discovery_count,1350)+sum(seed_fractions,Fraction(0)))/4
            passed=fresh>self.threshold and four>self.threshold and sum(s>self.threshold for s in seed_fractions)>=2
        return dict(time=time.time(),status='complete'if complete else'incomplete_with_stopped_runs'if any(r.get('stopped')for r in rows)else'incomplete',
            rows=rows,seeds=seeds,completed_parents=sum(r['complete']for r in rows),expected_parents=27,
            actual_finite_late_models_verified=len(self.verifier.checkpoints),expected_late_models=81,actual_checkpoint_pins=self.pins,
            source_and_frozen_copy_contracts_verified=contracts,seed_diversity=diversity,discovery_model_hashes_rechecked=27,
            discovery_nine_task_mean=float(Fraction(discovery_count,1350)),fresh_three_seed_nine_task_mean=float(fresh)if complete else None,
            fresh_exact_mean=str(fresh)if complete else None,all_four_seed_nine_task_mean=float(four)if complete else None,
            all_four_exact_mean=str(four)if complete else None,fresh_seed_sample_SD=statistics.stdev(float(s)for s in seed_fractions)if complete else None,
            passes_original_independent_confirmation_gate=passed,provided_plain_thresholds=self.thresholds,
            original_threshold=str(self.threshold),goal_complete=False,final_root_scientific_review_required=True,
            raw_metric='Equal means over50episodes,800K/900K/1M,nine tasks,andthree fresh seeds. Exact rational success counts determine strict thresholds.',
            limits=['No policy replay, new training, source mutation or goal-state change is performed by this auditor.',
                'Three independent training seeds provide descriptive SD, not a proof of universal stability or paired FQL statistical superiority.',
                'Finite recorded model/probe states do not prove exact conditional residuals, B_k, or global neural stability.',
                'The discovery seed is selected evidence; fresh and allfour means are reported separately.',
                'Numerical gate readiness does not establish individual necessity of finite-root and mean-head components; the separate mechanism pilot remains separately reported.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('contract',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--watch',action='store_true');args=parser.parse_args();contract=json.loads(args.contract.read_text())
    output=args.output or Path(contract['output']);output.mkdir(parents=True,exist_ok=False)
    lock=(output/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);auditor=Auditor(contract)
    while True:
        try:
            result=auditor.poll();result.update(pid=os.getpid(),execution_contract_sha256=auditor.verifier.digest(args.contract),audit_source_sha256=auditor.verifier.digest(__file__))
        except(Exception)as error:
            helper.atomic(output/'status.json',dict(time=time.time(),pid=os.getpid(),status='audit_error',error=repr(error),goal_complete=False));raise
        helper.atomic(output/'status.json',result)
        if result['status']=='complete'or not args.watch:
            helper.atomic(output/'audit.json',result)
            print(json.dumps({k:result[k]for k in('status','completed_parents','actual_finite_late_models_verified','fresh_three_seed_nine_task_mean','passes_original_independent_confirmation_gate')}),flush=True)
            return
        time.sleep(30)


if __name__=='__main__':main()
