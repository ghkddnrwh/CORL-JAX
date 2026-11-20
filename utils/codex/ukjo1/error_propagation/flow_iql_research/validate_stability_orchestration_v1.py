"""Exercise 16-slot refill, worker retries, and complete-only five-seed statistics."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
from utils.codex.ukjo1.error_propagation.flow_iql_research import run_stability_study_v1 as controller
from utils.codex.ukjo1.error_propagation.flow_iql_research import analyze_stability_study_v1 as analyzer


def scheduler_check():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); (root/'analysis').mkdir()
        (root/'analysis/numerical_validation_v1.json').write_text('{"status":"PASS"}')
        jobs=[dict(id=f'job{i}',output=str(root/'runs'/f'job{i}'),environment='synthetic',recipe='control',seed=i%5,
                   train_module='synthetic.train',eval_module='synthetic.evaluate') for i in range(17)]
        plan=dict(training_seeds=list(range(5)),discount=.999,sample_counts=[9],source_sha256={},
                  jobs=jobs,reused_jobs=[],total_cells=17,evaluation_steps=[300000,800000],
                  episodes_by_step={'300000':10,'800000':50},deployment_modes=['lean','actor'],environments=['synthetic'])
        (root/'registration_v1.json').write_text(json.dumps(plan))
        launches=[]; attempts={}; now=0
        def tick():
            nonlocal now; now+=31; return float(now)
        class Worker:
            def __init__(self,cmd,**kwargs):
                self.pid=100000+len(launches); self.returncode=0; launches.append((cmd,kwargs['env']))
                p=Path(cmd[cmd.index('--output')+1])
                if '--env' in cmd:
                    attempts[p.name]=attempts.get(p.name,0)+1
                    if p.name=='job0' and attempts[p.name]==1:
                        self.returncode=1; return
                    (p/'completed.json').write_text('{}')
                    for step in plan['evaluation_steps']:
                        for prefix in ('checkpoint','actor'): (p/f'{prefix}_{step}.pkl').write_bytes(b'test')
                else:
                    assert cmd[cmd.index('--episodes')+1]==('10' if cmd[cmd.index('--step')+1]=='300000' else '50')
                    p.write_text('{}')
            def poll(self): return self.returncode
        with contextlib.redirect_stdout(io.StringIO()), \
            patch.object(controller.sys,'argv',['test','--root',str(root)]), \
            patch.object(controller,'birth',return_value='synthetic'), \
            patch.object(controller.os,'sched_getaffinity',return_value=set(range(96))), \
            patch.object(controller.time,'time',side_effect=tick), patch.object(controller.time,'sleep'), \
            patch.object(controller.subprocess,'Popen',Worker), \
            patch.object(controller.subprocess,'run',return_value=subprocess.CompletedProcess([],0)):
            controller.main()
        status=json.loads((root/'controller_status.json').read_text())
        assert status['phase']=='complete' and status['training_complete']==17 and status['evaluations_complete']==68
        assert attempts['job0']==2 and len(attempts)==17
        first=[e for c,e in launches if '--env' in c][:16]
        assert [e['CUDA_VISIBLE_DEVICES'] for e in first].count('0')==8
        assert [e['CUDA_VISIBLE_DEVICES'] for e in first].count('1')==8
        assert all(len(c[c.index('-c')+1].split(','))==4 for c,e in launches)
        return dict(status='PASS',training_cells=17,first_wave_slots=16,per_GPU=8,retry_exercised=True,
                    evaluations=68,CPU_affinity=4)


def aggregation_check():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); (root/'analysis').mkdir(); jobs=[]
        for method in ('control','slow_target'):
            for seed in range(5):
                token=f'{method}_{seed}'; run=root/'runs'/token; directory=root/'evaluations'/token
                run.mkdir(parents=True); directory.mkdir(parents=True)
                jobs.append(dict(id=token,output=str(run),evaluation_directory=str(directory),environment='synthetic',
                    recipe=method,method=method,seed=seed,reused=False,head_aggregation='mean'))
                cfg=dict(discount=.999,expectile=.9,num_samples=9,backup_samples=9,target_clipping=True,
                         backup_Q_aggregation='mean',tau=.001 if method=='slow_target' else .005,critic_batch_size=256)
                (run/'config.json').write_text(json.dumps(dict(identity=dict(environment='synthetic',seed=seed,recipe=method,config=cfg))))
                (run/'completed.json').write_text(json.dumps(dict(step=1000000,q_updates=1000000,actor_updates=1000000,v_updates=0)))
                successes=[1.]*(15+5*seed if method=='control' else 24+seed)
                successes += [0.]*(50-len(successes))
                for step in (800000,900000,1000000):
                    model=run/f'checkpoint_{step}.pkl'; actor=run/f'actor_{step}.pkl'
                    model.write_bytes(b'model'); actor.write_bytes(b'actor')
                    for mode in ('lean','actor'):
                        v=dict(training_seed=seed,environment='synthetic',step=step,policy_mode=mode,
                            n_episodes=50,first_episode_seed=42,protocol='last3_cpu50_independent_seeded_episodes_v1',
                            no_optimizer_updates=True,episode_successes=successes,success_rate=sum(successes)/50,
                            config_sha256=analyzer.sha(run/'config.json'),checkpoint_sha256=analyzer.sha(model),actor_artifact_sha256=analyzer.sha(actor))
                        (directory/f'{step}_{mode}.json').write_text(json.dumps(v))
        plan=dict(jobs=jobs,reused_jobs=[],environments=['synthetic'],training_seeds=list(range(5)),
            methods=['control','slow_target'],deployment_modes=['lean','actor'],primary_evaluation_steps=[800000,900000,1000000],
            total_cells=10,source_sha256={},reused_evidence_sha256={})
        (root/'registration_v1.json').write_text(json.dumps(plan))
        def run(final):
            with contextlib.redirect_stdout(io.StringIO()),patch.object(analyzer.argparse.ArgumentParser,'parse_args',
                    return_value=argparse.Namespace(root=root,final=final)):
                analyzer.main()
        run(True)
        rows=list(csv.DictReader((root/'analysis/task_summary_final_v1.csv').open()))
        for row in rows:
            assert float(row['mean_pct'])==(50. if row['method']=='control' else 52.)
            if row['method']=='slow_target': assert row['descriptive_stability_candidate']=='True'
        missing=root/'evaluations/slow_target_4/1000000_actor.json'; saved=missing.read_text(); missing.unlink()
        run(False)
        rows=list(csv.DictReader((root/'analysis/task_summary_progress_v1.csv').open()))
        row=next(r for r in rows if r['method']=='slow_target' and r['deployment']=='actor')
        assert row['mean_pct']=='' and int(row['complete_seeds'])==4
        missing.write_text(saved)
        (root/'runs/control_0/checkpoint_800000.pkl').write_bytes(b'corrupt')
        try: run(True)
        except AssertionError: pass
        else: raise AssertionError('Model corruption not caught')
        return dict(status='PASS',five_seed_means=True,incomplete_seeds_not_pooled=True,corruption_detected=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=dict(status='PASS',scheduler=scheduler_check(),analysis=aggregation_check())
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__': main()
