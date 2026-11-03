# IQL residual / propagation 실험

**9환경 × 3시드 최종 평균은 0.2600으로 요청한 IQL 기준 0.19를 초과했다.**
27개 실제 최종 모델의 Q/actor1M, V2M, 전체 state finite, episode 배열과
학습 계보를 검증했다. 총1350회 중351회 성공이다. 시드별9환경 평균은
0.3222 / 0.2156 / 0.2422이며, 방법 선정에 쓰지 않은 고정 후속 seed1/2의
평균도 **0.2289**다. 학습 중 최고 checkpoint를 선택하지 않았다.

같은 실행기로 원래 할인율의 native IQL을 재실행한 seed0 평균은0.2133이다.
제안 방법과 native는 Q/actor update 수가 같지만, 제안 방법은 V를2M회,
native는1M회 갱신하므로 **총 optimization compute가 같은 비교는 아니다.**
Native 재실행은 한 시드다. 아래 수치는 관측된 성능이며 통계적 유의성이나
전역 수렴을 주장하지 않는다. τ=.99의 장기9환경 검증도 수행하지 않았다.

## 최종 성능

| 환경 | 제공 IQL 평균 | 같은 실행기의 IQL seed0 | 제안 방법 3시드 평균 ± SD |
|---|---:|---:|---:|
| AM-L | 0.140 | 0.100 | 0.140 ± 0.080 |
| HM-M | 0.440 | 0.540 | 0.493 ± 0.070 |
| HM-L | 0.050 | 0.020 | 0.093 ± 0.076 |
| Soccer | 0.060 | 0.040 | 0.033 ± 0.023 |
| Cube-S | 0.840 | 0.920 | 0.900 ± 0.020 |
| Cube-D | 0.000 | 0.000 | 0.000 ± 0.000 |
| Puzzle3 | 0.010 | 0.000 | 0.007 ± 0.012 |
| Puzzle4 | 0.040 | 0.020 | 0.053 ± 0.042 |
| Scene | 0.110 | 0.280 | 0.620 ± 0.342 |
| **9환경 평균** | **0.190** | **0.2133** | **0.2600 ± 0.0555** |

SD는3개 training-seed별 점수의 sample SD다. 마지막 평균 행의 SD는
시드별9환경 평균으로 계산했다. 평가 episode와 reset 변동도 포함되며
순수 training variance 성분만을 추정한 값은 아니다. 제공 기준0.19는
criterion.txt의 평균 행을 그대로 사용했다. 원래 할인율은 task별 .99 또는 .995다.

재현성을 확인한 별도 frozen CPU50 평가에서는 다음을 얻었다. 기존 점수를
새 점수로 대체하거나 더 좋은 쪽을 선택하지 않았다.

| 평가 대상 | 주 평가 | 별도 frozen 평가 |
|---|---:|---:|
| Native IQL, 원래 할인율, seed0 | 0.2133 | 0.2244 |
| 제안 방법, 전체3시드 | **0.2600** | **0.2696** |
| 제안 방법, 독립 후속 seed1/2 | **0.2289** | **0.2444** |

모델과 실제 episode 검증은
[primary27 milestone](../../logs/diagnosis/propagation_research_20261025/analysis/candidate_full27_milestone.json),
[secondary27 milestone](../../logs/diagnosis/propagation_research_20261025/analysis/candidate_secondary27_milestone.json),
[최종 비교 JSON](../../logs/diagnosis/propagation_research_20261025/analysis/final_benchmark_comparison.json),
[CSV](../../logs/diagnosis/propagation_research_20261025/analysis/final_benchmark_comparison.csv)에 있다.
[최종 비교 그림 PNG](../../logs/diagnosis/propagation_research_20261025/analysis/final_benchmark_comparison.png)와
[PDF](../../logs/diagnosis/propagation_research_20261025/analysis/final_benchmark_comparison.pdf)를 함께 저장했다.

## 선정한 방법

실행 코드는 [iql_residual_control.py](../../algorithms/uk_offline/basic/iql_residual_control.py)의
`bounded_v2` profile이다. Expectile τ=.9, discount γ=.999이며 다음 세 flag를 고정했다.

```python
{"bound_backup": True, "v_steps": 2, "fresh_backup": True}
```

1. 기존 target-Q의 최솟값을 고정하고, 같은 minibatch에서 V expectile loss를
   Adam으로 두 번 fitting한다. V optimizer state도 두 번 갱신한다.
2. 새 V로 `y = r + γ mask V_new(s_next)`를 계산한다.
3. `y`를 알려진 반환값 범위 `[L,0]`로 제한해 critic을 한 번 갱신한다.
4. Target-Q EMA를 갱신하고 actor를 한 번 학습한다. Actor advantage는
   원래 순서인 `min(Q_target_old) - V_old`를 사용한다.

Network 크기, 데이터, optimizer, actor loss, actor schedule과 target-Q EMA rate는
유지했다. Expectile τ와 target EMA `tau=.005`는 다른 설정이다.
Reward 범위가 `[-n,0]`이면 보수적인 하한은 `L=-n/(1-γ)`다.
설치 OGBench 보상식을 확인해 navigation/soccer/Cube-S는 n=1, Cube-D는2,
Puzzle3은9, Puzzle4는16, Scene은5를 사용했다. 새 실행은 `--bound-source ogbench`다.

참 target이 범위 안에 있다면 interval projection은 그 target에 대한
pointwise error를 늘리지 않는다. 신경망 fitting과 Adam carry까지 포함한
전체 반복의 contraction이나 좋은 actor를 보장하는 결과는 아니다.
이 방법은 정확한 conditional residual을 진단했다는 가정을 사용하지 않는다.
추가 V fitting도 참 conditional residual 제거와 동일시하지 않는다.

초기 empirical interval과 known interval이 정확히 같았던8개 환경은 같은
연산으로 취급했다. Puzzle4만 보수적인 하한으로 별도 seed0 실행을 등록했고,
모든 seed1/2는 처음부터 known bound를 썼다. Legacy Puzzle4 실행은 별도로 보존했다.

## 안정성과 원인 분석

필수48개 모델 실험은40개 실제1M 최종 모델과8개 scale-stop으로 종료됐다.
중단된1M 점수는 미관측으로 두었고0이나 마지막 점수로 채우지 않았다.
후보27개는 모두1M을 완주했다. 전체 학습의 fixed-probe Q RMS / recorded
return scale 최댓값은0.2413 미만이었다. 이것이 모든 상태의 정확성이나
향후 학습의 안정성을 보장하지는 않는다.

같은 τ=.9, γ=.999의 Cube/Scene stress 비교는 다음과 같다.

| 방법 | Cube seed0/1/2 | Scene seed0/1/2 |
|---|---|---|
| Native IQL | 786k / 759k / 507k에서 scale stop | seed0는1M 성공0이나 큰 양의 Q; seed1/2는719k / 643k stop |
| 수학적으로 같은 custom 구현 대조 | seed0는834k stop; seed1은1M .92와 작은 값; seed2는1M .18이나 Q평균7935 | seed0는1M 성공0과 큰 양의 Q; seed1/2는778k / 916k stop |
| 제안 방법 | **1M .92/.90/.88**, 평균 .90 | **1M .94/.26/.66**, 평균 .62 |

따라서 수치 대조군의 일부 완주를 안정화 성공으로 세지 않는다.
Finite parameter나 완료 flag만으로도 안정성을 판단하지 않는다.
[3시드 경과 그림](../../logs/diagnosis/propagation_research_20261025/analysis/paired_stress_trajectories.png)과
[실제 비교표](../../logs/diagnosis/propagation_research_20261025/analysis/paired_stress_table.csv)를 따른다.
수치 차이만으로 한 시드의 경로가 크게 바뀔 수 있었지만, 선택한 조합이
이 여섯 사례에서는 더 일관되게 값 폭주를 피했다. 표본3시드의 관측이다.

진단에서 확인한 점은 다음과 같다.

| 관측 | 해석과 한계 |
|---|---|
| 기존 V21의5개 분기에서 B damping/relaxation 적용률0 | 시작 reference의 유효성 gate가 계속 닫혔다. 이 로그로 실제 B 제어의 효과가 없다고 결론 내릴 수 없다. |
| Native IQL은 PRNG split4, 기존 diagnostic 초기화는 split5 | 같은 정수 seed도 시작 가중치가 달랐다. 새 비교는 native 초기화를 공유했다. |
| 전역 평균 score와 batch loss는 상태별 conditional residual과 다름 | 상태 간 상쇄가 가능하며 neural gradient 정상조건도 모든 상태의 score0보다 약하다. Neighbor action reference도 behavior conditional oracle이 아니다. |
| 같은 batch의 추가 V fitting이 train loss를 낮춰도 validation loss를 높인 사례 | 온라인 value_loss는 마지막 V update 직전의 loss다. 실제 post-update trace와 validation을 따로 확인했다. |
| 정확한 expectile residual0인 선형 예에서 projected critic gain>1 | Forcing을 줄여도 불안정한 homogeneous 반복이 남을 수 있다. OGBench에 맞춘 모델이나 사용자 이론의 반박은 아니다. |
| κ로 V loss를 재가중하는 방법도 실패 사례를 보임 | Non-realizable 함수근사에서는 projection의 최적점도 달라질 수 있다. 같은 최적점을 더 빨리 찾는 preconditioner라고 단정할 수 없다. |
| 일부 작은-epsilon JVP/FD 비교가 불일치 | 수치 미분의 유효성 실패를 동역학적 불안정으로 읽지 않는다. 방향별 gain도 전체 operator norm은 아니다. |
| Frozen Cube actor는 오래된 Q/V로 .86, 늦은 Q/V로 .04; validation loss를 낮춘 V 재적합 후에도 .00 | Expectile loss를 낮추는 것만으로 actor가 복구되지 않았다. 정확한 residual 제거를 검증한 실험은 아니며 추가 actor compute 분석이라 benchmark에 넣지 않았다. |
| LayerNorm-only seed0의 최종9환경 평균 .0667 | 이 실험에서는 LayerNorm만으로 문제를 해결하지 못했다. 모든 구현에 대한 일반적인 부정은 아니다. |

Original-discount Puzzle3의687k probe는 Q평균5.65였지만 centered RMS3980.77,
absolute Q의 p99는271.46, twin-min upper0 위반 RMS는3969.87이었다.
같은 기록의 global score/κ는-2.09였다. 작은 전역 proxy가 상태별 conditional
residual의 정확성을 인증하지는 않는다. 사후 선택한 한 시점의 예이며
오류의 원인 분해나 정책 실패의 단독 원인을 증명한 것은 아니다.
[원본 record와 정의](../../logs/diagnosis/propagation_research_20261025/analysis/puzzle3_original_tail_diagnostic_687k.json)를 보존했다.

값의 크기가 안정돼도 성능 문제가 남았다. Cube-D는 세 시드 모두0,
Puzzle3 평균 .0067, Soccer .0333이고 Scene의 per-seed SD는 .342로 컸다.
Fresh native 대비 평균 이득도 Scene에 크게 집중됐다. Q의 공통 이동,
상태별 advantage 가중치, 행동 의존 Q 순서 변화를 구분해 분석했다.
Borrowed-action 순서 변화가 참 Q 순서 오류라는 증명은 아니다.
Scene의 같은600k actor를 old400k Q/V로 추가 fitting하면 .66,600k Q/V로는 .34였다.
두 실행의 초기50episode 결과가1개 달라 정확한 paired episode 추론은 하지 않는다.

데이터 연결 오류도 점검했다.9환경 train/validation의12,090,100개 non-end
인접 전이가 모두 연결됐고, goal 라벨과 mask도 일치했다. Cube-D train의
goal frame은581/1M, Puzzle4는0/1M이었다. Goal frame은 독립적인 성공
trajectory 수가 아니며, Puzzle3과 Scene은 빈도가 비슷한데 성능이 달라
이 빈도만으로 실패를 설명하지 않는다.

## 평가 재현성

설치 OGBench 조작 환경의 `action_space`는 접근마다 새 Box를 반환했다.
임시 Box에 준 seed가 reset warmup에서 다음 Box에 유지되지 않았다.
관측이 같아도 full physics state와 짧은 경로는 달랐으며, 해당 환경 instance에만
동일 bounds/dtype의 Box를 보존하고 seed를 주자 반복 결과가 정확히 같았다.
설치 환경 파일이나 live training은 바꾸지 않았다.

Puzzle4 하한 수정 전후의 최종 model/optimizer 배열은 모두 같았지만 native
CPU50은 .02/.04였다. 별도 seeded-space 조건에서는 두 actor의50회 성공·보상
배열이 모두 일치했고 둘 다 .06이었다. 어떤 secondary 점수도 보기 전에
[별도 평가 계획](../../logs/diagnosis/propagation_research_20261025/seeded_evaluation_plan.json)을 등록했다.
Fresh environment/global RNG 초기화도 새 protocol의 일부이므로 모든 차이를
action-space 수정 하나의 효과로 해석하지 않는다. 전체36개 frozen 모델과
동일 actor 대조1개가 끝났고 checkpoint/source hash와 episode 배열을 검증했다.
기존 점수는 그대로 두었으며 더 좋은 평가 쪽을 선택하지 않았다.
이 대조가 과거 모든 불일치 episode의 유일한 원인을 규명한 것은 아니다.

## 실행과 재현

Repository root에서 `ogbench` 환경의 Python을 사용한다. `suite.py`는 새
queue/source/config를 준비하며 바로 학습을 시작하지 않는다. 새 빈 경로를 쓴다.

```bash
python -m utils.propagation_research.suite logs/new_bounded_iql \
  --profile bounded_v2 --expectile .9 --discount .999 \
  --bound-source ogbench --seeds 0 1 2

python -m utils.propagation_research.suite logs/new_original_iql \
  --profile baseline --expectile .9 --discount original --seeds 0
```

GPU가 비었을 때 생성한 `run_suite.sh`를 실행한다. Suite별로2 GPU × GPU당8개,
`XLA_PYTHON_CLIENT_MEM_FRACTION=.1`을 사용하므로 두 suite는 순서대로 실행한다.
실행 전 source/config hash를 검증한다. 학습 RNG와 probe/evaluation RNG를
분리했고 continuation은 optimizer·학습 RNG·actor schedule을 이어받는다.
실패 replay나 actor refit은 독립 benchmark seed로 세지 않는다.

```bash
bash logs/new_bounded_iql/run_suite.sh
# 위 suite가 끝난 뒤:
bash logs/new_original_iql/run_suite.sh

python -m utils.propagation_research.benchmark_audit \
  logs/diagnosis/propagation_research_20261025
python -m utils.propagation_research.stress_comparison \
  logs/diagnosis/propagation_research_20261025 --plot
python -m utils.propagation_research.final_report \
  logs/diagnosis/propagation_research_20261025
```

Frozen actor의 별도 평가에는 실제 run 경로와 새 출력 파일을 지정한다.

```bash
python -m utils.propagation_research.evaluate_checkpoint RUN_DIR NEW_RESULT.json \
  --checkpoint checkpoint_1000000.pkl --device cpu --episodes 50 \
  --seeded-action-space
```

`evaluate_seeded_finals.py`는 현재 campaign의 고정된36개 모델과 equal-actor
대조를 관리한다. 일반 재학습 결과에는 위 단일 frozen evaluator를 사용할 수 있다.
준비된 재현 queue는 campaign의 `reproduction/bounded_v2_known_seeds012_final/`,
`reproduction/baseline_original_seed0_final/`에 있으며 이번 연구 종료 후 자동 재시작하지 않는다.

`capture_environment.py CAMPAIGN NEW_OUTPUT`은 repository source tar, 설치
package 버전, OGBench Python/XML hash, 실제18개 train/validation NPZ hash와
실험 계획을 보존한다. Dataset/model binary와 설치 package 자체는 bundle에
넣지 않는다. 복원한 repository에서는 새 output으로 suite 경로를 다시 준비한다.
하드웨어 전반의 bitwise 학습 재현을 주장하지 않는다.

## 분석 자료

탐색 기록은 기준·수치 대조군을 포함한22개 profile 이름, 33개 설정 그룹이다.
100k 이상 관측한 현행 기록은124개지만 continuation 때문에 독립 seed 수와 다르다.
[탐색 inventory](../../logs/diagnosis/propagation_research_20261025/analysis/exploration_inventory.json)는
학습 길이·평가 device·episode 수와 취소된 optional 실행을 구분하고 서로 다른
pilot 점수를 평균하지 않는다. 목표 검증 후 추가 optional 학습은 종료했으며
해당 취소를 발산이나 최종 점수로 처리하지 않았다.

| 코드 | 용도 |
|---|---|
| `diagnose_checkpoint.py` | Validation loss, 근사 conditional residual, V trace, optimizer 포함 JVP/FD |
| `advantage_drift.py` | Q 공통 이동·상태별 advantage·행동 순서 변화 |
| `backup_distribution.py` | 고정 V backup의 물리적 범위 위반 위치·크기 |
| `actor_counterfactual.py` | 같은 actor/optimizer/replay에서 frozen Q/V를 바꾼 추가 fitting |
| `dataset_profile.py` | Goal frame, mask, loader 전이 연결 확인 |
| `exact_residual_counterexample.py` | Residual0인 projected critic 및 coupled V-fitting의 구성적 예 |
| `benchmark_audit.py` | 실제 full checkpoint counter·finite state·episode·계보 검증 |
| `seeded_reset.py` | Evaluation instance에만 action space를 보존하는 별도 protocol |
| `verify_protocol.py` | Native 대비 sampling/update/resume 검증 |
| `final_report.py` | 최종 JSON/CSV/PNG/PDF 재생성 |

Analysis-only parameter archive는 학습이나 optimizer sensitivity 계산에 쓰지 않는다.
Actor counterfactual의 frozen Q/V 입력에만 명시적으로 허용했고, 초기 actor에는
실제 optimizer가 필요했다. Full optimizer가 있는 failure terminal snapshot도
분석 전용이며 학습 재시작을 허용하지 않았다. 진단의 source/checkpoint hash,
실패한 방법과 해석 한계는 [RESEARCH_LOG.md](RESEARCH_LOG.md)에 있다.

최종 소스·환경 snapshot은 campaign의
[`reproduction/environment_snapshot_final/`](../../logs/diagnosis/propagation_research_20261025/reproduction/environment_snapshot_final/)에 보존한다.
코드 export를 별도 임시 repository에 복원해 SHA, CPU update, 새 suite 생성·검증을
확인한 기록은
[`final_source_bundle_validation.json`](../../logs/diagnosis/propagation_research_20261025/analysis/final_source_bundle_validation.json)이다.
Source-only export에 Git metadata가 없을 때도 실행되도록 최종 포장 단계에서
revision 조회 실패만 처리했다. 학습 함수·클래스·profile의 AST는 바꾸지 않았다.
[해당 변경의 검증](../../logs/diagnosis/propagation_research_20261025/analysis/final_packaging_core_equivalence.json)과
각 실제 실행의 원본 source snapshot을 함께 남겼다. 이전 v3 queue도 보존했다.
