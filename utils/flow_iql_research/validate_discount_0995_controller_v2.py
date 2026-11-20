"""Resource-free checks of two-seed orchestration and complete-only aggregation."""
import contextlib
import csv
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from unittest.mock import patch
from utils.flow_iql_research import run_discount_0995_n9_v2 as controller
from utils.flow_iql_research import analyze_discount_0995_n9_v2 as analyzer

def check_loop(retry, seed):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        output = root / "runs" / "synthetic"
        (root / "analysis").mkdir()
        (root / "analysis/numerical_validation_v1.json").write_text('{"status":"PASS"}')
        job = dict(id="synthetic", output=str(output), environment="synthetic",
                   recipe="n8", seed=seed, train_module="synthetic.train",
                   eval_module="synthetic.evaluate")
        plan = dict(training_seeds=[0, 1], new_training_seeds=[0, 1], discount=.995, source_sha256={}, jobs=[job], reused_jobs=[],
                    environments=["synthetic"], sample_counts=[9],
                    evaluation_steps=[800000, 900000, 1000000],
                    deployment_modes=["lean", "actor"], total_cells=1)
        (root / "registration_v2.json").write_text(json.dumps(plan))
        launches = []
        train_attempts = 0
        clock = 0

        def tick():
            nonlocal clock
            clock += 31
            return float(clock)

        class SyntheticWorker:
            def __init__(self, command, **kwargs):
                nonlocal train_attempts
                launches.append(command)
                self.pid = 100000 + len(launches)
                self.returncode = 0
                if "--env" in command:
                    assert command[command.index("--seed") + 1] == str(seed)
                    train_attempts += 1
                    if retry and train_attempts == 1:
                        self.returncode = 1
                        return
                    (output / "completed.json").write_text("{}")
                    for step in plan["evaluation_steps"]:
                        for prefix in ("checkpoint", "actor"):
                            (output / f"{prefix}_{step}.pkl").write_bytes(b"synthetic")
                else:
                    result = Path(command[command.index("--output") + 1])
                    result.write_text("{}")

            def poll(self):
                return self.returncode

        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(controller.sys, "argv", ["test", "--root", str(root)]), \
                patch.object(controller, "birth", return_value="synthetic-birth"), \
                patch.object(controller.os, "sched_getaffinity", return_value=set(range(96))), \
                patch.object(controller.time, "time", side_effect=tick), \
                patch.object(controller.time, "sleep"), \
                patch.object(controller.subprocess, "Popen", SyntheticWorker), \
                patch.object(controller.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            controller.main()
        status = json.loads((root / "controller_status.json").read_text())
        events = [json.loads(line) for line in (root / "controller_events.jsonl").read_text().splitlines()]
        assert status["phase"] == "complete" and status["analysis_exit_code"] == 0
        assert status["training_complete"] == 1 and status["evaluations_complete"] == 6
        assert train_attempts == (2 if retry else 1)
        assert sum(e["kind"] == "TRAIN_FINISH" for e in events) == 1
        assert sum(e["kind"] == "EVAL_FINISH" for e in events) == 6
        assert sum(e["kind"] == "TRAIN_RETRY_QUEUED" for e in events) == int(retry)
        assert all("event_time" in e and "time" in e for e in events)
        assert all(len(c[c.index("-c") + 1].split(",")) == 4 for c in launches)
        return dict(retry=retry, training_seed=seed, learner_attempts=train_attempts,
                    evaluation_finishes=6, controller_phase=status["phase"])


def check_analysis():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'analysis').mkdir()
        envs = [f'antmaze-fixture-{i}' for i in range(9)]
        plan = dict(training_seeds=[0, 1], discount=.995, environments=envs,
            deployment_modes=['lean', 'actor'], evaluation_steps=[800000, 900000, 1000000],
            jobs=[], reused_jobs=[], source_sha256={}, preserved_reference_sha256={}, reused_evidence_sha256={})
        for eidx, env in enumerate(envs):
            for seed in range(2):
                token = f'fixture_{eidx}_s{seed}'
                run, evaluation = root / 'runs' / token, root / 'evaluations' / token
                run.mkdir(parents=True)
                evaluation.mkdir(parents=True)
                job = dict(id=token, environment=env, seed=seed, recipe='g995_n9', output=str(run),
                    reused=False, head_aggregation='min', evaluation_directory=str(evaluation))
                plan['reused_jobs' if job['reused'] else 'jobs'].append(job)
                cfg = dict(discount=.995, expectile=.9, tau=.005, backup_samples=9,
                    num_samples=9, target_clipping=True, backup_Q_aggregation='min')
                config = dict(identity=dict(environment=env, seed=seed, recipe='g995_n9', config=cfg), source_sha256={})
                (run / 'config.json').write_text(json.dumps(config))
                (run / 'completed.json').write_text(json.dumps(dict(step=1000000, q_updates=1000000,
                    actor_updates=1000000, v_updates=0)))
                (run / 'diagnostics.jsonl').write_text(''.join(json.dumps(dict(step=step, q_updates=step,
                    actor_updates=step, v_updates=0, finite=True)) + '\n' for step in range(1000, 1000001, 1000)))
                successes = [1.] * (10 + 5*seed + eidx)
                successes += [0.] * (50 - len(successes))
                for step in plan['evaluation_steps']:
                    checkpoint, actor = run / f'checkpoint_{step}.pkl', run / f'actor_{step}.pkl'
                    checkpoint.write_bytes(b'synthetic checkpoint')
                    actor.write_bytes(b'synthetic actor')
                    for mode in plan['deployment_modes']:
                        value = dict(training_seed=seed, environment=env, recipe='g995_n9', run=token,
                            step=step, policy_mode=mode, n_episodes=50, first_episode_seed=42,
                            episode_successes=successes, success_rate=sum(successes)/50, no_optimizer_updates=True,
                            protocol='last3_cpu50_independent_seeded_episodes_v1', config_sha256=analyzer.sha(run / 'config.json'),
                            checkpoint_sha256=analyzer.sha(checkpoint), actor_artifact_sha256=analyzer.sha(actor))
                        (evaluation / f'{step}_{mode}.json').write_text(json.dumps(value))
        plan.update(expected_new_evaluations=108, expected_reused_evaluations=0)
        (root / 'registration_v2.json').write_text(json.dumps(plan))

        def analyze(final):
            argv = ['test', '--root', str(root)] + (['--final'] if final else [])
            with contextlib.redirect_stdout(io.StringIO()), patch.object(sys, 'argv', argv):
                analyzer.main()

        analyze(True)
        summary = json.loads((root / 'analysis/summary_final_v2.json').read_text())
        assert summary['complete'] and summary['episode_records'] == 5400
        assert summary['new_training_complete'] == 18 and summary['reused_training_complete'] == 0
        for row in summary['macros']:
            assert abs(row['environment_mean'] - .33) < 1e-12
            assert abs(row['training_seed_sample_sd'] - statistics.stdev([.28, .38])) < 1e-12
        missing = root / 'evaluations/fixture_0_s1/1000000_actor.json'
        saved = missing.read_text()
        missing.unlink()
        analyze(False)
        progress = json.loads((root / 'analysis/summary_progress_v2.json').read_text())
        assert not progress['complete']
        actor_row = next(r for r in progress['macros'] if r['deployment'] == 'actor')
        assert actor_row['environment_mean'] is None and actor_row['available_complete_seeds'] == 1
        missing.write_text(saved)
        (root / 'runs/fixture_0_s0/checkpoint_800000.pkl').write_bytes(b'corrupt')
        try:
            analyze(True)
        except AssertionError:
            pass
        else:
            raise AssertionError('Corruption was not detected')
        return dict(status='PASS', synthetic_training_cells=18, synthetic_evaluation_points=108,
            two_seed_mean_and_sample_SD=True, missing_evaluation_excluded=True, model_hash_corruption_detected=True)


def main():
    result = dict(status='PASS', cases=[check_loop(False, 0), check_loop(True, 1)], analysis=check_analysis(),
        scope='Resource-free orchestration and aggregation; no performance claim.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()



