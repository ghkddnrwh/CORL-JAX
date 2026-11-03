"""Audit and plot a provisional CPU profile; never change selection gates."""
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path('/data/sda2/ukjo1/finite_mean_onestep_20261031')
    out=root/'analysis/deployment_CPU_preliminary_v1'
    result_path=out/'result_v1.json'
    result=json.loads(result_path.read_text())
    plan=json.loads((out/'input_v1.json').read_text())
    assert result['status']=='complete' and len(result['rows'])==3
    assert not result['isolated_machine'] and not result['counts_toward_acceptance_speed_gate']
    assert result['teacher_updates']==result['registered_GPU_updates']==result['rollouts']==0
    for path,digest in plan['source_sha256'].items():assert sha(path)==digest,path
    for job in plan['jobs']:
        run=Path(job['run'])
        assert sha(run/'actor_1000000.pkl')==job['actor_sha256']
        assert sha(run/'checkpoint_1000000.pkl')==job['checkpoint_sha256']
    handle=json.loads((out/'handle_v1.json').read_text())
    proc=Path(f'/proc/{handle["pid"]}/stat')
    if proc.exists():
        fields=proc.read_text().rsplit(') ',1)[1].split()
        assert fields[19]!=handle['birth'] or fields[0]=='Z'
    status_path=root/'analysis/onestep_pipeline_v1/status.json'
    status_text=status_path.read_text()
    (out/'performance_status_snapshot_v1.json').write_text(status_text)
    performance={row['run']:row for row in json.loads(status_text)['rows']}
    rows=[]
    for row in result['rows']:
        p=performance[row['run']]
        assert p['complete'] and p['training_complete']
        assert set(p['per_step']['primary'])==set(p['per_step']['teacher'])=={'800000','900000','1000000'}
        assert row['portable_and_training_policy_actions_identical']
        m=row['metadata']
        assert m['actor_forward_passes']==1 and m['generated_candidate_actions']==m['flow_integration_steps']==0
        assert not m['requires_critic'] and not m['requires_BC_flow']
        rows.append(dict(environment=row['environment'],run=row['run'],seed=row['seed'],
            CPU_actor_ms=row['primary_ms_median'],CPU_teacher_ms=row['teacher_ms_median'],
            CPU_speedup=row['CPU_speedup'],actor_last3_CPU50=p['primary_mean'],teacher_last3_CPU50=p['teacher_mean']))
    ratio=float(np.exp(np.mean(np.log([r['CPU_actor_ms']/r['CPU_teacher_ms'] for r in rows]))))
    audit=dict(time=time.time(),status='pass',source_sha256=sha(__file__),
        profile_sha256=sha(result_path),input_sha256=sha(out/'input_v1.json'),
        performance_snapshot_sha256=sha(out/'performance_status_snapshot_v1.json'),
        no_model_or_GPU_queue_mutation=True,profiling_worker_terminal=True,rows=rows,
        CPU_latency_ratio_geomean=ratio,CPU_speedup_geomean=1/ratio,
        machine_isolated=False,acceptance_speed_gate_evaluated=False,GPU_latency_measured=False,
        performance_scope='Only three seed0 pilot tasks; nine-task/fresh-seed retention not established.',
        speed_scope='Exploratory CPU6 synchronous single-observation resident-weights latency while16GPUlearners continue. Not end-to-end environment or paper FQL hardware comparison.')
    (out/'profile_audit_v1.json').write_text(json.dumps(audit,indent=2))
    with (out/'pilot_speed_and_performance_v1.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    labels=['Antmaze large','Cube double','Humanoidmaze medium']
    x=np.arange(3);w=.34
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    colors=['#377eb8','#ef8a28']
    for i,(name,key) in enumerate([('Flow best9','CPU_teacher_ms'),('One-step actor','CPU_actor_ms')]):
        bars=axes[0].bar(x+(i-.5)*w,[r[key] for r in rows],w,label=name,color=colors[i])
        axes[0].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axes[0].set(ylabel='CPU milliseconds / action (lower is faster)',ylim=(0,5.4),title='Provisional deployment latency')
    for i,(name,key) in enumerate([('Flow best9','teacher_last3_CPU50'),('One-step actor','actor_last3_CPU50')]):
        bars=axes[1].bar(x+(i-.5)*w,[r[key] for r in rows],w,label=name,color=colors[i])
        axes[1].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    axes[1].set(ylabel='Success rate (higher is better)',ylim=(0,1.13),title='Same recipe: 800K / 900K / 1M average')
    for ax in axes:
        ax.set_xticks(x,labels,fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.01))
    fig.text(.5,.012,'bank_bc, seed 0. CPU6; concurrent 16 GPU learners. Not an isolated speed gate or a nine-task retention result.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(out/'pilot_speed_and_performance_v1.png',dpi=150)
    fig.savefig(out/'pilot_speed_and_performance_v1.pdf')
    plt.close(fig)
    report='''One-step actor의 CPU 배포 시간 예비 측정\n\n최종 선정 전의 중간 진단이다. finite-mean lean+best9 직접 모방(bank_bc)의 완료된 seed0 모델3개를 사용했다. actor-only export의 standalone client와 등록 학습 구현의 출력이 정확히 같았고, 실제1M critic/actor update와V0을 확인했다. 원래 모델·GPU 대기열은 수정하지 않았고 rollout/optimizer update도0이다.\n\n같은 체크포인트의 Flow10-step+9후보+minQ 선택과, one-step MLP 하나만 실행하는 정책을 동일한CPU6에서 비교했다. 가중치와 관측/난수키를CPU에 둔 뒤10회 warm-up하고3회×100번의 동기 호출을 측정했다. 학습16개는 계속 실행되었으므로 장비가 분리된 최종 speed gate가 아니다. 초기 load/compile/환경step/IO/통신도 제외한 policy-call 측정이다.\n\n'''
    for label,row in zip(labels,rows):
        report+=f"{label}: Flow best9 {row['CPU_teacher_ms']:.3f}ms → actor {row['CPU_actor_ms']:.3f}ms, {row['CPU_speedup']:.2f}배. last3 CPU50 성능 teacher {row['teacher_last3_CPU50']:.6f}, actor {row['actor_last3_CPU50']:.6f}.\n"
    report+=f'''\nCPU 속도비의3환경 기하평균은{1/ratio:.2f}배다. 이는 one-step화로 배포 연산을 줄일 수 있다는 예비 근거다. Cube-double/Humanoidmedium에서는 paired teacher보다 성능이 낮다. 세 pilot 평균에서 eligible한 이 방법을 현재 나머지6개 환경에도 같은 설정으로 학습 중이며, 전체9개/fresh3seed 성능과 분리된CPU/GPU속도 기준은 아직 통과하지 않았다. FQL 논문의A5000 측정과 직접 비교하지 않는다.\n\n실행 구현: utils/flow_iql_research/standalone_onestep_actor_v1.py\n측정: utils/flow_iql_research/profile_onestep_cpu_v1.py\n근거: result_v1.json, profile_audit_v1.json, pilot_speed_and_performance_v1.csv\n'''
    (out/'PRELIMINARY_CPU_DEPLOYMENT_KO_v1.txt').write_text(report)
    p=root/'CONTINUATION.json';v=json.loads(p.read_text())
    v['preliminary_CPU_deployment_profile']=dict(status='complete_audited_CPU_only_not_isolated_not_gate',audit=str(out/'profile_audit_v1.json'),report=str(out/'PRELIMINARY_CPU_DEPLOYMENT_KO_v1.txt'),plot=str(out/'pilot_speed_and_performance_v1.png'),CPU_speedup_geomean=1/ratio,nine_task_retention_established=False,goal_complete=False)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
    print(json.dumps(audit))


if __name__=='__main__':main()
