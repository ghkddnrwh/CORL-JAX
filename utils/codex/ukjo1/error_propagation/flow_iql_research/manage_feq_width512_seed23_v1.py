"""Extend the frozen FEQ width experiment to seeds 2/3; aggregate seeds 0--3.

Training/evaluation and the queue engine are reused unchanged. Only job
registration and aggregation differ from the completed seed0/1 study.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import statistics as st
import time

from utils.codex.ukjo1.error_propagation.flow_iql_research import manage_feq_width512_v1 as scheduler
from utils.codex.ukjo1.error_propagation.flow_iql_research.default_gamma_pair_n9_v1 import (
    FAMILIES, PREFIX, STEPS, atomic, audit_config, audit_evaluation, sha, write_csv,
)

ROOT = Path('logs/codex/ukjo1/error_propagation/diagnosis/feq_width512_default10_g995_g999_seed23_20261128_v1')
PARENT = scheduler.ROOT
MODULE = PREFIX + 'manage_feq_width512_seed23_v1'
CHANGED = ('cube1', 'cube2', 'puzzle3', 'puzzle4', 'scene')


def parent_plan(plan):
    path = Path(plan['parent_registration'])
    assert sha(path) == plan['parent_registration_sha256']
    return json.loads(path.read_text())


def register(root):
    from utils.codex.ukjo1.error_propagation.flow_iql_research import feq_width512_core_v1 as core
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'registration.json').exists():
        return
    parent_path = PARENT.resolve() / 'registration.json'
    parent = json.loads(parent_path.read_text())
    assert json.loads((PARENT / 'controller_status.json').read_text())['phase'] == 'complete'
    assert json.loads((PARENT / 'analysis/summary.json').read_text())['complete']
    for name, digest in parent['source_sha256'].items():
        assert sha(name) == digest, name
    previous = json.loads((scheduler.BASELINE / 'registration_v1.json').read_text())
    previous_jobs = {Path(j['output']).resolve(): j for j in previous['jobs'] + previous['reused_jobs']}
    with (scheduler.BASELINE / 'analysis/per_seed_results_final_v1.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['seed']) in (2, 3) and r['deployment'] == 'lean']
    assert len(rows) == 40
    baselines, reused, jobs, evidence = [], [], [], {}
    for row in rows:
        run = Path(row['run_directory'])
        original = previous_jobs[run.resolve()]
        conf = json.loads((run / 'config.json').read_text())
        gamma, seed, code = float(row['gamma']), int(row['seed']), row['code']
        recipe = 'g995_w512_n9' if gamma == .995 else 'g999_w512_n9'
        width = 256 if code in CHANGED else 512
        cfg = conf['identity']['config']
        assert cfg['value_hidden_dims'] == cfg['actor_hidden_dims'] == [width] * 4
        expected = core.config_for(row['environment'], recipe)
        expected['value_hidden_dims'] = expected['actor_hidden_dims'] = (width,) * 4
        audit_config(original, conf, reference=expected)
        assert json.loads((run / 'completed.json').read_text())['step'] == 1000000
        checks = []
        for mode in ('lean', 'actor'):
            for step in STEPS:
                path = Path(row['evaluation_directory']) / f'{step}_{mode}.json'
                score = audit_evaluation(original, conf, step, mode, path, evidence)
                checks.append(dict(mode=mode, step=step, path=str(path), sha256=sha(path), success_pct=score))
        ref = dict(id=f'g{str(gamma)[2:]}_{code}_s{seed}', code=code, environment=row['environment'],
                   gamma=gamma, seed=seed, width=width, output=str(run),
                   config_sha256=sha(run / 'config.json'), evaluations=checks)
        baselines.append(ref)
        if width == 512:
            reused.append(ref)
            continue
        name = f'g{str(gamma)[2:]}_{code}_w512_s{seed}'
        job = dict(id=name, code=code, environment=row['environment'], gamma=gamma, seed=seed,
                   width=512, recipe=recipe, output=str(root / 'runs' / name),
                   evaluation_directory=str(root / 'evaluations' / name),
                   head_aggregation='mean', baseline_id=ref['id'])
        atomic(root / 'jobs' / f'{name}.json', job)
        jobs.append(job)
    assert len(jobs) == len(reused) == 20
    assert {(j['code'], j['gamma'], j['seed']) for j in jobs} == {
        (c, g, s) for c in CHANGED for g in (.995, .999) for s in (2, 3)}
    # Validate parent width512 evaluations before pooling them with new seeds.
    for job in parent['jobs']:
        run = Path(job['output'])
        conf = json.loads((run / 'config.json').read_text())
        audit_config(job, conf)
        assert json.loads((run / 'completed.json').read_text())['step'] == 1000000
        for mode in ('lean', 'actor'):
            for step in STEPS:
                audit_evaluation(job, conf, step, mode,
                                 Path(job['evaluation_directory']) / f'{step}_{mode}.json', evidence)
    plan = copy.deepcopy(parent)
    plan.update(study='FEQ width512 seed2/3 extension; pooled seeds0--3', training_seeds=[2, 3],
                report_training_seeds=[0, 1, 2, 3], baseline=baselines, reused_width512=reused,
                jobs=jobs, baseline_evidence=evidence,
                parent_registration=str(parent_path), parent_registration_sha256=sha(parent_path),
                aggregation='800K/900K/1M within each seed; pooled mean and sample SD over seeds0--3; paired width differences',
                controller_sources={str(Path(__file__).resolve()): sha(__file__),
                                    str(Path(scheduler.__file__).resolve()): sha(scheduler.__file__)})
    for name in [*plan['source_sha256'], *plan['controller_sources']]:
        source = Path(name)
        dest = root / 'source' / ('vendor_' + source.name if source.parent.name == 'vendor' else source.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    atomic(root / 'registration.json', plan)
    print(json.dumps(dict(new_training=20, reused_seed23_width512=20, pooled_training_seeds=[0, 1, 2, 3])))


def collect(plan, label):
    rows, evaluation_count = [], 0
    for ref in plan['baseline']:
        assert sha(Path(ref['output']) / 'config.json') == ref['config_sha256']
        for mode in ('lean', 'actor'):
            checks = sorted([c for c in ref['evaluations'] if c['mode'] == mode], key=lambda c: c['step'])
            assert len(checks) == 3
            for c in checks:
                assert sha(c['path']) == c['sha256']
            vals = [c['success_pct'] for c in checks]
            rows.append(dict(code=ref['code'], environment=ref['environment'], gamma=ref['gamma'], seed=ref['seed'],
                             width=ref['width'], deployment=mode, origin='baseline', points=3,
                             success_800K=vals[0], success_900K=vals[1], success_1M=vals[2],
                             last3_success_pct=st.mean(vals), run=ref['output']))
    for job in plan['jobs']:
        path = Path(job['output']) / 'config.json'
        conf = json.loads(path.read_text()) if path.exists() else None
        if conf:
            audit_config(job, conf)
            assert conf['identity']['config']['value_hidden_dims'] == conf['identity']['config']['actor_hidden_dims'] == [512] * 4
        for mode in ('lean', 'actor'):
            values = {}
            for step in STEPS:
                p = Path(job['evaluation_directory']) / f'{step}_{mode}.json'
                if p.exists():
                    assert conf is not None
                    values[step] = audit_evaluation(job, conf, step, mode, p)
                    evaluation_count += 1
            rows.append(dict(code=job['code'], environment=job['environment'], gamma=job['gamma'], seed=job['seed'],
                             width=512, deployment=mode, origin=label, points=len(values),
                             success_800K=values.get(800000, ''), success_900K=values.get(900000, ''),
                             success_1M=values.get(1000000, ''),
                             last3_success_pct=st.mean(values.values()) if len(values) == 3 else '', run=job['output']))
    return rows, evaluation_count


def analyze(root):
    plan = json.loads((root / 'registration.json').read_text())
    parent = parent_plan(plan)
    old_rows, old_evals = collect(parent, 'width512_seed01')
    new_rows, new_evals = collect(plan, 'width512_seed23')
    rows = old_rows + new_rows
    keys = [(r['code'], r['gamma'], r['seed'], r['width'], r['deployment']) for r in rows]
    assert len(keys) == len(set(keys)) == 240
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    write_csv(out / 'per_seed_results_0123.csv', rows)
    summary, paired, macro = [], [], []
    complete = old_evals == new_evals == 120 and all(
        (Path(j['output']) / 'completed.json').exists() for j in parent['jobs'] + plan['jobs'])
    lines = ['FEQ width256/512 비교: default10, seed0--3, gamma .995/.999',
             f'완료={complete}; 추가 seed2/3 평가 {new_evals}/120; 기존 seed0/1 평가 {old_evals}/120.',
             '각 시드의 800K/900K/1M 성공률을 평균한 뒤 네 시드 평균±표본SD.',
             '네 시드가 모두 완료된 조건만 평균을 보고한다. 개별 결과는 per_seed_results_0123.csv.',
             '기존 512인 AM/HM/Soccer 5환경은 재사용. Q/Flow/MDN 전체 폭을 함께 비교한다.', '']
    for code, env in FAMILIES:
        for gamma in plan['discounts']:
            for mode in ('lean', 'actor'):
                groups = {}
                for width in ((256, 512) if code in CHANGED else (512,)):
                    selected = sorted([r for r in rows if r['code'] == code and r['gamma'] == gamma
                                       and r['deployment'] == mode and r['width'] == width and r['points'] == 3],
                                      key=lambda r: r['seed'])
                    full = [r['seed'] for r in selected] == [0, 1, 2, 3]
                    vals = [r['last3_success_pct'] for r in selected]
                    item = dict(code=code, environment=env, gamma=gamma, deployment=mode, width=width,
                                completed_seeds=len(selected), complete=full,
                                mean_success_pct=st.mean(vals) if full else '', sample_sd_pct=st.stdev(vals) if full else '',
                                **{f'seed{s}_success_pct': next((r['last3_success_pct'] for r in selected if r['seed'] == s), '') for s in range(4)})
                    summary.append(item)
                    if full:
                        groups[width] = vals
                        lines.append(f"{code} gamma={gamma} {mode} width={width}: {st.mean(vals):.2f} ± {st.stdev(vals):.2f}%")
                    else:
                        lines.append(f'{code} gamma={gamma} {mode} width={width}: 대기 ({len(selected)}/4 seeds)')
                if 256 in groups and 512 in groups:
                    delta = [b - a for a, b in zip(groups[256], groups[512])]
                    paired.append(dict(code=code, gamma=gamma, deployment=mode, mean_delta_pp=st.mean(delta),
                                       sample_sd_delta_pp=st.stdev(delta), **{f'seed{s}_delta_pp': d for s, d in enumerate(delta)}))
    write_csv(out / 'task_mean_sd_0123.csv', summary)
    if paired:
        write_csv(out / 'paired_width_differences_0123.csv', paired)
    for scope, codes in [('changed5', CHANGED), ('default10', [c for c, _ in FAMILIES])]:
        for gamma in plan['discounts']:
            for mode in ('lean', 'actor'):
                for variant in ('baseline', 'all512'):
                    vals = []
                    for seed in range(4):
                        selected = [r for r in rows if r['code'] in codes and r['gamma'] == gamma and r['seed'] == seed
                                    and r['deployment'] == mode and r['points'] == 3
                                    and r['width'] == (256 if variant == 'baseline' and r['code'] in CHANGED else 512)]
                        if len(selected) == len(codes):
                            vals.append(st.mean(r['last3_success_pct'] for r in selected))
                    if len(vals) == 4:
                        macro.append(dict(scope=scope, gamma=gamma, deployment=mode, variant=variant,
                                          mean_success_pct=st.mean(vals), sample_sd_pct=st.stdev(vals),
                                          **{f'seed{s}_success_pct': v for s, v in enumerate(vals)}))
    if macro:
        write_csv(out / 'macro_mean_sd_0123.csv', macro)
        lines += ['', '환경 평균 (시드별 환경 평균을 계산한 뒤 네 시드 평균±표본SD):']
        lines += [f"{m['scope']} gamma={m['gamma']} {m['deployment']} {m['variant']}: {m['mean_success_pct']:.2f} ± {m['sample_sd_pct']:.2f}%" for m in macro]
    (out / 'RESULTS_KO_0123.txt').write_text('\n'.join(lines) + '\n')
    if complete:
        evidence = {}
        for job in plan['jobs']:
            run = Path(job['output'])
            conf = json.loads((run / 'config.json').read_text())
            done = json.loads((run / 'completed.json').read_text())
            assert done['step'] == done['q_updates'] == done['actor_updates'] == 1000000 and done['v_updates'] == 0
            for mode in ('lean', 'actor'):
                for step in STEPS:
                    audit_evaluation(job, conf, step, mode, Path(job['evaluation_directory']) / f'{step}_{mode}.json', evidence)
        atomic(out / 'new_evaluation_evidence.json', evidence)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
        for gi, gamma in enumerate(plan['discounts']):
            for mi, mode in enumerate(('lean', 'actor')):
                ax = axes[gi, mi]
                for wi, width in enumerate((256, 512)):
                    data = [next(r for r in summary if r['code'] == code and r['gamma'] == gamma
                                 and r['deployment'] == mode and r['width'] == width) for code in CHANGED]
                    x = [i + (wi - .5) * .34 for i in range(len(CHANGED))]
                    ax.bar(x, [r['mean_success_pct'] for r in data], .32,
                           yerr=[r['sample_sd_pct'] for r in data], capsize=3, label=f'Width {width}', alpha=.8)
                    for s in range(4):
                        ax.scatter([v + (s - 1.5) * .025 for v in x], [r[f'seed{s}_success_pct'] for r in data], s=12, color='black', zorder=3)
                ax.set_xticks(range(len(CHANGED))); ax.set_xticklabels(CHANGED)
                ax.set_ylim(0, 105); ax.set_ylabel('Success (%)')
                ax.set_title(f'gamma={gamma}, {mode}, seeds0--3'); ax.legend(); ax.grid(axis='y', alpha=.15)
        fig.tight_layout()
        fig.savefig(out / 'width_comparison_0123.png', dpi=200)
        fig.savefig(out / 'width_comparison_0123.pdf')
        plt.close(fig)
    atomic(out / 'summary.json', dict(complete=complete, new_evaluations=new_evals, expected_new_evaluations=120,
                                     parent_evaluations=old_evals, report_training_seeds=[0, 1, 2, 3], time=time.time()))
    return complete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('register', 'run', 'supervise', 'analyze'))
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    root = args.root.resolve()
    if args.action in ('run', 'supervise'):
        plan = json.loads((root / 'registration.json').read_text())
        for name, digest in plan['controller_sources'].items():
            assert sha(name) == digest, name
    # Use the existing queue/supervisor in this process, with the new report
    # callback and entry point for restarted controllers. Frozen learner and
    # evaluation module names remain unchanged.
    scheduler.analyze = analyze
    scheduler.THIS_MODULE = MODULE
    {'register': register, 'analyze': analyze, 'run': scheduler.run, 'supervise': scheduler.supervise}[args.action](root)


if __name__ == '__main__':
    main()
