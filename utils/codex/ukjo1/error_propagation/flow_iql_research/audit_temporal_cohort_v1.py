"""Audit raw K3 evaluations and retained checkpoints across three train seeds.

Task/checkpoint/seed completeness is mandatory. Incomplete cohorts never get a
pooled mean. Source hashes, optimizer clocks and actual episode arrays are
checked, rather than trusting the watchers' summary flags.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re
import statistics
import time
import numpy as np


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)); temp.replace(path)


def provided_thresholds(path):
    raw = path.read_text(); result = {}
    pattern = r'^\s*(FQL|ReBRAC)\s*\n(.*?)(?=\\\\)'
    for match in re.finditer(pattern, raw, re.MULTILINE | re.DOTALL):
        values = []
        for field in match.group(2).split('&')[1:]:
            value = re.match(r'\s*([0-9]+(?:\.[0-9]+)?)', field)
            assert value, field
            values.append(float(value.group(1)))
        assert len(values) == 10
        result.setdefault(match.group(1), []).append(values[-1])
    assert set(result) == {'FQL', 'ReBRAC'}
    return {k: max(v) for k, v in result.items()}, hashlib.sha256(raw.encode()).hexdigest()


class Verifier:
    def __init__(self):
        self.hashes = {}; self.checkpoints = {}

    def digest(self, path):
        path = Path(path); stat = path.stat()
        stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        cached = self.hashes.get(str(path))
        if cached is None or cached[0] != stamp:
            value = hashlib.sha256(path.read_bytes()).hexdigest()
            after = path.stat()
            assert stamp == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            self.hashes[str(path)] = (stamp, value)
        return self.hashes[str(path)][1]

    def checkpoint(self, path, config, step):
        digest = self.digest(path)
        if digest not in self.checkpoints:
            with Path(path).open('rb') as file:
                saved = pickle.load(file)
            assert saved['step'] == step
            assert json.loads(json.dumps(saved['identity'])) == config['identity']
            assert saved['source_sha256'] == config['source_sha256']
            assert int(saved['state']['q_updates']) == step
            assert int(saved['state']['v_updates']) == 2*step
            assert int(saved['state']['agent']['network']['opt_state']['0']['count']) == step
            assert int(saved['state']['v_opt']['0']['count']) == 2*step

            def finite(value):
                if isinstance(value, dict):
                    return all(finite(v) for v in value.values())
                if isinstance(value, (list, tuple)):
                    return all(finite(v) for v in value)
                if value is None:
                    return True
                array = np.asarray(value)
                assert array.dtype.kind in 'biufc', array.dtype
                return bool(np.isfinite(array).all())
            assert finite(saved['state'])
            self.checkpoints[digest] = (step, config['identity'])
        assert self.checkpoints[digest] == (step, config['identity'])
        return digest

    def seed(self, plan):
        seed = plan['training_seed']; valid = {}; errors = []; configs = {}
        for path, digest in plan['source_sha256'].items():
            assert self.digest(path) == digest, path
        for job in plan['jobs']:
            output = Path(job['output'])
            if not output.exists():
                continue
            try:
                run = Path(job['run']); config_path = run/'config.json'
                config = json.loads(config_path.read_text()); identity = config['identity']
                if 'expected_identity' in job:
                    assert identity == job['expected_identity']
                    assert config['source_sha256'] == plan['expected_training_sources']
                else:
                    assert self.digest(config_path) == job['config_sha256']
                assert identity['seed'] == seed and identity['environment'] == job['environment']
                assert identity['kind'] == 'ifql' and identity['profile'] == 'bounded_v2'
                assert identity['config']['discount'] == .999 and identity['config']['expectile'] == .9
                assert identity['steps'] == 1000000 and identity['episodes'] == 50
                for source, digest in config['source_sha256'].items():
                    assert self.digest(source) == digest, source
                configs[identity['environment']] = config
                result = json.loads(output.read_text())
                assert result['training_seed'] == seed and result['environment'] == job['environment']
                assert result['step'] == result['actor_training_step'] == job['step']
                assert result['selection_rule'] == 'temporal_mean_min_heads'
                assert result['selection_samples'] == 9
                assert result['critic_combination'] == 'mean_of_per_member_minimum_twin_Q'
                assert result['n_episodes'] == 50 and result['first_episode_seed'] == 42
                assert result['eval_device'] == 'cpu'
                assert result['protocol'] == 'last3_cpu50_independent_seeded_episodes_v1'
                successes = result['episode_successes']
                assert len(successes) == len(result['episode_rewards']) == len(result['episode_lengths']) == 50
                assert set(successes) <= {0., 1.}
                assert abs(statistics.mean(successes)-result['success_rate']) < 1e-12
                assert result['config_sha256'] == self.digest(config_path)
                assert result['checkpoint_sha256'] == self.checkpoint(job['checkpoint'], config, job['step'])
                assert len(result['critic_members']) == len(job['member_checkpoints']) == 3
                for offset, member, path in zip((-200000, -100000, 0), result['critic_members'], job['member_checkpoints']):
                    assert member['step'] == job['step']+offset
                    assert Path(member['checkpoint']).resolve() == Path(path).resolve()
                    assert member['sha256'] == self.checkpoint(path, config, member['step'])
                for field, source in [('selector_source_sha256', 'utils/flow_iql_research/temporal_selection_v1.py'),
                                      ('evaluator_source_sha256', 'utils/flow_iql_research/evaluate_temporal_selection_v1.py')]:
                    assert result[field] == plan['source_sha256'][source]
                valid[job['id']] = result['success_rate']
            except (AssertionError, KeyError, OSError, ValueError) as error:
                errors.append(dict(job=job['id'], error=repr(error)))
        rows = []
        for environment in plan['environments']:
            jobs = [j for j in plan['jobs'] if j['environment'] == environment]
            assert len(jobs) == 3 and {j['step'] for j in jobs} == {800000, 900000, 1000000}
            done = Path(jobs[0]['run'])/'completed.json'; complete = False
            if done.exists():
                counts = json.loads(done.read_text())
                complete = counts['step'] == counts['q_updates'] == 1000000 and counts['v_updates'] == 2000000
            scores = [valid.get(j['id']) for j in jobs]
            rows.append(dict(environment=environment, parent_complete=complete,
                per_step={str(j['step']): valid.get(j['id']) for j in jobs},
                last3_mean=statistics.mean(scores) if complete and all(v is not None for v in scores) else None))
        means = [r['last3_mean'] for r in rows if r['last3_mean'] is not None]
        return dict(training_seed=seed, valid_evaluations=len(valid), completed_tasks=len(means),
            nine_task_mean=statistics.mean(means) if len(means) == 9 and not errors else None,
            rows=rows, errors=errors), configs


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--watch', action='store_true'); args = parser.parse_args()
    root = args.root; verifier = Verifier()
    names = ['temporal_critic_full9_registration_v1.json',
             'confirmation_temporal_seed1_registration_v1.json',
             'confirmation_temporal_seed2_registration_v1.json']
    plans = [json.loads((root/name).read_text()) for name in names]
    plans[0]['training_seed'] = 0
    assert [p['training_seed'] for p in plans] == [0, 1, 2]
    threshold, criterion_sha = provided_thresholds(Path('algorithms/uk_offline/basic/propagation/criterion.txt'))
    while True:
        seeds, configurations = [], []
        for plan in plans:
            result, configs = verifier.seed(plan); seeds.append(result); configurations.append(configs)
        source_contracts = {}; config_contracts = {}
        for configs in configurations:
            for env, config in configs.items():
                cfg = dict(config['identity']['config']); cfg.pop('num_samples')
                if env in config_contracts:
                    assert cfg == config_contracts[env], env
                else:
                    config_contracts[env] = cfg
                common = {Path(k).name: v for k, v in config['source_sha256'].items()
                          if not Path(k).name.startswith('train_policy')}
                if env in source_contracts:
                    assert common == source_contracts[env], env
                else:
                    source_contracts[env] = common
        scores = [s['nine_task_mean'] for s in seeds]
        all_complete = all(score is not None for score in scores)
        independent = statistics.mean(scores[1:]) if all(s is not None for s in scores[1:]) else None
        overall = statistics.mean(scores) if all_complete else None
        audit = dict(time=time.time(), expected_training_seeds=[0, 1, 2], primary_steps=[800000, 900000, 1000000],
            rule='temporal_mean_min_heads', inference_samples=9, critic_members=3, seeds=seeds,
            all_three_seeds_complete=all_complete, three_seed_nine_task_mean=overall,
            three_seed_sample_SD=statistics.stdev(scores) if all_complete else None,
            independent_seed1_2_nine_task_mean=independent, provided_plain_competitor_thresholds=threshold,
            criterion_sha256=criterion_sha,
            overall_exceeds_provided=overall > max(threshold.values()) if overall is not None else None,
            independent_exceeds_provided=independent > max(threshold.values()) if independent is not None else None,
            configs_and_learning_sources_match=True,
            caveats=['Seed0 is exploratory and selected the inference recipe; seeds1/2 were locked before their performance existed.',
                     'Three correlated temporal snapshots are not three training seeds.',
                     'Original seed0 runner uses32inference samples and confirmation runners9; training does not use this setting.',
                     'Auxiliary evaluation schedules differ, with replay/learning independence separately validated.',
                     'Fresh same-protocol competitor completion and value ablations must also be inspected before a final claim.'])
        atomic(root/'analysis/confirmation_three_seed_audit_v1.json', audit)
        if not args.watch or all_complete:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
