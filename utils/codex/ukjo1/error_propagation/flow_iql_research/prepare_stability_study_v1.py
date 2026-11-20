"""Register 100 fixed cells before outcomes; reuse 11 matched N9 controls."""
import argparse
import csv
import json
from pathlib import Path
import time
from utils.codex.ukjo1.error_propagation.flow_iql_research import stability_study_core_v1 as core
from utils.codex.ukjo1.error_propagation.flow_iql_research.analyze_discount_0995_n9_v2 import sha, atomic, csv_file


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve(); root.mkdir(parents=True,exist_ok=True)
    assert not (root/'registration_v1.json').exists()
    folder=Path(__file__).resolve().parent; workspace=folder.parents[1]
    prior=workspace/'logs/diagnosis/finite_mean_gamma0995_n9_9task_2seed_20261119_v2/analysis/all_equivalent_seeds_gamma0999_20261120_v1'
    candidates=list(csv.DictReader((prior/'included_unique_seed_results.csv').open()))
    source_paths=core.source_paths()+list(folder.glob('*stability_study_v1.py'))
    sources={str(p.resolve()):sha(p) for p in source_paths}
    reused=[]; evidence={str(prior/'included_unique_seed_results.csv'):sha(prior/'included_unique_seed_results.csv')}
    historical=[]
    for row in candidates:
        if row['environment'] not in core.ENVIRONMENTS: continue
        historical.append(row)
        if row['deployment']!='lean' or int(row['training_seed']) not in range(5): continue
        run=Path(row['run_directory']); cfg=json.loads((run/'config.json').read_text())
        c=cfg['identity']['config']; env=row['environment']; seed=int(row['training_seed'])
        assert c['discount']==.999 and c['expectile']==.9 and c['tau']==.005
        assert c['num_samples']==c.get('backup_samples',9)==9 and c['batch_size']==256
        assert c['layer_norm'] and c['target_clipping']
        assert c['backup_Q_aggregation']==('min' if env.startswith('antmaze') else 'mean')
        for p,d in cfg['source_sha256'].items():
            assert sha(p)==d,p
            assert p not in sources or sources[p]==d
            sources[p]=d
        directory=Path(row['raw_files'].split(';')[0]).parent
        for step in (800000,900000,1000000):
            model=run/f'checkpoint_{step}.pkl'; actor=run/f'actor_{step}.pkl'
            evidence[str(model)]=sha(model); evidence[str(actor)]=sha(actor)
            for mode in ('lean','actor'):
                p=directory/f'{step}_{mode}.json'; v=json.loads(p.read_text())
                assert v['training_seed']==seed and v['n_episodes']==50 and v['first_episode_seed']==42
                assert v['checkpoint_sha256']==evidence[str(model)] and v['actor_artifact_sha256']==evidence[str(actor)]
                evidence[str(p)]=sha(p)
        for name in ('config.json','dataset.json','completed.json','diagnostics.jsonl'):
            p=run/name; evidence[str(p)]=sha(p)
        complete=json.loads((run/'completed.json').read_text())
        assert complete['step']==complete['q_updates']==complete['actor_updates']==1000000 and complete['v_updates']==0
        reused.append(dict(id=run.name,environment=env,seed=seed,method='control',recipe=cfg['identity']['recipe'],
            output=str(run),evaluation_directory=str(directory),reused=True,head_aggregation=c['backup_Q_aggregation']))
    assert len(reused)==11
    tags=dict(zip(core.ENVIRONMENTS,('aml','hmm','cube2','puzzle4')))
    existing={(j['environment'],j['seed']) for j in reused}
    jobs=[]
    for seed in range(5):
        for method in core.RECIPES:
            for env in core.ENVIRONMENTS:
                if method=='control' and (env,seed) in existing: continue
                token=f'stab1_{tags[env]}_{method}_s{seed}'
                jobs.append(dict(id=token,environment=env,seed=seed,recipe=method,method=method,
                    output=str(root/'runs'/token),evaluation_directory=str(root/'evaluations'/token),
                    train_module='utils.codex.ukjo1.error_propagation.flow_iql_research.train_stability_study_v1',
                    eval_module='utils.codex.ukjo1.error_propagation.flow_iql_research.evaluate_stability_study_v1',
                    reused=False,head_aggregation='min' if env.startswith('antmaze') else 'mean'))
    assert len(jobs)==89
    plan=dict(created_time=time.time(),purpose='User-authorized four-task stability investigation and from-scratch interventions.',
        environments=list(core.ENVIRONMENTS),training_seeds=list(range(5)),sample_counts=[9],
        methods=list(core.RECIPES),jobs=jobs,reused_jobs=reused,total_cells=100,
        discount=.999,expectile=.9,critic_heads=2,steps=1000000,
        head_rule='AM-L min; other tasks mean. TD target bounds retained.',
        actor_label_samples=9,lean_deployment_samples=9,
        evaluation_steps=[800000,900000,1000000,300000,500000,700000],
        primary_evaluation_steps=[800000,900000,1000000],deployment_modes=['lean','actor'],
        episodes_by_step={str(s):50 if s>=800000 else 10 for s in (300000,500000,700000,800000,900000,1000000)},
        evaluation_CPU_affinity=4,evaluation_reset_seeds='42..91 primary; 42..51 auxiliary',
        criterion='Report each seed, last3 mean and sample SD, worst seed, 1M and late range/drop. Lower SD alone is not stabilization.',
        descriptive_stability_flag='mean >= control AND SD < control AND worst seed >= control AND late range <= control',
        uncertainty='Five training seeds; exploratory comparison of four fixed interventions, no per-environment tuning or best checkpoint selection.',
        interventions=dict(slow_target=dict(target_EMA=.001),critic_batch1024=dict(critic_batch=1024,BC_actor_batch=256,
            extra_replay_rng='seed+823719; separately checkpointed',critic_samples_per_update_multiplier=4,
            caveat='Equal optimizer updates, not equal critic sample/compute budget. 300K auxiliary point roughly matches baseline 1M sampled transitions.'),
            dr3_0p001=dict(coefficient=.001),dr3_0p01=dict(coefficient=.01)),
        DR3_definition='mean_batch sum_heads mask * dot(phi_online(s,a), sum_N detached normalized expectile_weights * phi_online(sprime,aN)). Gradients through both features; actions and weights detached. Added to sum-head squared TD loss.',
        DR3_caveat='Finite-expectile adaptation, not a direct reproduction of the original CQL+DR3 algorithm.',
        no_tuning_on_evaluation=True,all_runs_from_scratch=True,
        evaluation_never_used_for_training=True,diagnostic_bank_only='Freeze 9 Flow candidates at 10K on fixed replay states solely to measure Q churn; never used by policy or update.',
        source_sha256=sources,reused_evidence_sha256=evidence,
        literature=[dict(title='DR3',url='https://arxiv.org/abs/2112.04716',use='feature dot-product regularization; coefficients .001/.01 from studied grid'),
                    dict(title='ReBRAC',url='https://papers.nips.cc/paper/2023/file/26cce1e512793f2072fd27c391e04652-Paper-Conference.pdf',use='large batches as a tested design choice; task-dependent outcomes, not guaranteed improvement'),
                    dict(title='FQL',url='https://arxiv.org/abs/2502.02538',use='existing AM min target rule and critic LayerNorm')])
    atomic(root/'registration_v1.json',plan)
    (root/'analysis').mkdir(exist_ok=True)
    csv_file(root/'analysis/historical_seed_evidence.csv',historical)
    (root/'EXPERIMENT_KO.txt').write_text(
        'AM-L / HM-M / Cube-Double / Puzzle4x4 안정화 대조 실험\n'
        'N=9, gamma=.999, expectile=.9, seed 0/1/2/3/4. 기존 방식 + 4개 개입 설정 = 100개 학습 셀.\n'
        '동일 조건 기준선 11개 재사용; 89개를 처음부터 1M까지 학습. GPU0/1 각각 8개, 자동 충원.\n'
        '개입: target EMA .001 / critic batch1024 / DR3 .001 / DR3 .01. 한 번에 한 요인만 변경.\n'
        '원래 critic에 LayerNorm이 이미 있음. DR3는 유한 expectile의 가중 다음 표현으로 확장.\n'
        'Flow와 actor는 원래 batch256, actor는 기존 순수 likelihood 학습. AM-L min, 나머지 mean 유지.\n'
        '평가: 각 모델의 Lean/actor, 800K/900K/1M 각 50회, reset seed42..91, CPU4코어.\n'
        '보조 곡선: 300K/500K/700K 각10회. 주 통계에는 포함하지 않음.\n'
        '시드별 평균/SD, 최악 시드, 최종 성능, 후반 하락을 함께 판단. 평균을 희생한 SD 감소만으로 안정화라 하지 않음.\n'
        '기존 800K Q 평균/고정 배포 방식은 적용하지 않음. 10K 고정 후보는 순위 변화 측정 전용.\n'
        '큰 critic 배치는 갱신당 표본/계산량이 약4배이므로 동일 계산량 개선으로 주장하지 않음.\n'
        '다섯 시드의 탐색적 비교이며 환경별 승자만 골라 일반적 개선으로 주장하지 않음.\n')
    print(json.dumps(dict(root=str(root),new_runs=len(jobs),reused=len(reused),cells=100)),flush=True)


if __name__=='__main__': main()
