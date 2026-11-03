# IQL residual / propagation research

목표: OGBench 9개 default singletask 환경의 동일 Q/actor update 수 평가에서,
기준 IQL (tau=0.9, original gamma) 평균 약 0.19를 초과하는 안정적인 방법 검증.
파일 criterion.txt의 반올림한 환경별 수치를 직접 평균하면 0.18778이다.
완료 판정은 9개 환경 전체의 재현 가능한 실험 결과를 사용한다.

## 실험 규칙

- 탐색: cube-single, scene, seed 0, 300k step; actor cosine schedule은 1M 기준 유지.
- 탐색 평가는 checkpoint당 20 episode. 최종 비교는 1M step, 50 episode, 여러 seed.
- 최종 checkpoint를 주 지표로 사용하고 학습 중 최고값으로 선택하지 않는다.
- 환경을 누락하거나 실패한 run을 제거해서 평균을 높이지 않는다.
- 같은 seed에서 초기화와 NumPy minibatch 순서를 공유한다.
- V 추가 update는 gradient 계산량을 늘리므로, Q update 수와 V update 수를 구분한다.
- 원본 update를 그대로 호출하는 baseline도 동일 scan 실행기를 사용한다.
  5 step synthetic test에서 scan/eager parameter 차이 최대 약 8.7e-5 (V),
  optimizer 포함 최대 4.23e-4. Q/V 출력 allclose(atol=rtol=1e-4) 통과.
  scan의 XLA 부동소수점 연산 순서 차이를 bitwise 일치로 표현하지 않는다.
- 새로운 run은 config, source snapshot/hash, git HEAD, 데이터 보상 범위,
  evaluation episode별 성공/보상, fixed-probe diagnostics를 저장한다.
- 기존 파일·로그를 덮어쓰지 않는다. 새 로그 루트:
  `logs/diagnosis/propagation_research_20261025`.
- GPU별 최대 8개, memory fraction 0.1. scheduler가 완료 즉시 다음 job 실행.

## 이론 해석

사용자 PDF 식 (17): epsilon_next = M epsilon + gamma B (zeta/kappa) + e + eta + rho.
1/kappa가 커지는 것은 residual의 조건수 악화를 뜻한다. 실제 residual의 크기,
방향, 시간적 상관과 B의 작용을 측정하기 전에는 residual이 실험적으로
지배적이라고 단정할 수 없다. G는 조건부 기댓값이므로 한 action의 score나
batch 평균과 같지 않다. 신경망 V의 최적화 정상조건도 상태별 G=0보다 약하다.

## 2026-10-25: 기존 V21 감사

`utils/propagation_research/audit_existing.py`로 현재 V21 분기 로그를 집계했다.
결과: `analysis/historical_branches.csv` (새 로그 루트 기준).

- bk_independent_control: 기록된 5개 분기 전부 Q damping 적용 비율 0.
- bk_target_relax: 기록된 5개 분기 전부 target relaxation 적용 비율 0.
- 각 분기의 시작 reference v20_B_valid = 0.
- 시작 reference random-direction half-epsilon 상대 오차:
  0.9062, 0.8570, 0.6526, 1.1796, 0.9898. 유효성 문턱은 0.5.
- 제어기는 현재 측정치뿐 아니라 시작 reference의 유효성을 계속 요구하므로
  이 5개 분기에서는 적응형 B 보정이 영구적으로 꺼진다.
- 이는 이 로그가 B 제어의 효과를 검증하지 못했다는 구현상 사실이다.
  B 제어를 실제 적용해도 효과가 없다는 결론과 구별해야 한다.
- residual target control의 평균 적용 비율 약 0.234,
  mean applied target RMS displacement 약 0.00812.
  (서로 다른 분기 기록의 평균이며 seed별 성능 평균이나 인과 효과 추정이 아니다.)
- residual propagation joint의 적용 비율 약 0.00699.

## 첫 탐색군

wave 1: 두 환경 각각 baseline setting 1 / baseline setting 4,
V 4회+fresh backup, 조건부 kappa preconditioning, critic LayerNorm,
보상 범위 기반 target projection, target V EMA, 결합법.

wave 2 대기열: setting 1에서 LayerNorm/kappa/결합/V4,
setting 4에서 V4(old-V backup)/Q step 0.5/target-Q EMA 0.001/LayerNorm+kappa.

LayerNorm은 B를 직접 측정하는 제어기가 아닌 함수근사의 전파 민감도를
변경하는 개입이다. projection도 데이터 보상 범위에 의존한다.
어떤 방법도 전역 수렴이나 성능 향상을 보장한다고 가정하지 않는다.

## 초기 결과와 확인한 교란 요인

100k, cube, seed 0, 20 episode: original-setting baseline 0.75,
setting-4 baseline 0.50, target-V EMA setting-4 0.85, projection setting-4 0.75.
중간 checkpoint 소규모 평가이며 최종 성능 개선의 증거로 취급하지 않는다.

기본 IQL은 PRNG split(key, 4), 기존 PD는 split(key, 5)를 사용했다.
따라서 같은 정수 seed 0도 모델 가중치가 서로 다르다. 새 baseline은 기본 IQL의
초기화를 사용하며, 이전 발산 재현용 `--initialization diagnostic` 대조군을 추가했다.

100k cube baseline의 validation query 128개 / neighbor action 64개 근사:
local-action residual RMS 2.77, fresh V 20회 후 2.66.
neighbor 16 vs 64 expectile 차이 RMS 1.48로 참조 자체의 불확실성이 크다.
이를 정확한 expectile 오차로 단정하지 않는다.

조건부 kappa preconditioning 단독은 cube/scene에서 약 100k부터 양의 Q로
전환되어 baseline보다 빠르게 성장했다. cube의 100k local-action residual RMS는
5.56 (fresh V 20회 후 5.49). local action reference와 비교한 head 확률 평균은
0.044 vs 0.142였다. 다만 local reference도 behavior 조건부분포의 근사이며,
head 추정 오류만으로 이 발산을 전부 설명할 수 없다.

JVP 진단의 Adam sqrt 미분에서 0인 second moment의 0/0을 발견했다.
forward sqrt/Adam update는 유지하고 inactive coordinate tangent를 0으로 처리했다.
10 step의 실제 Optax update/state와 array_equal 테스트 통과.

동일 cube baseline 100k checkpoint, GPU default precision, query 64:
one-step random-direction JVP RMS gain 0.0107,
relative FD epsilon .001/.01/.1의 JVP 대비 상대오차 1.46/.456/.047.
CPU의 작은 epsilon에서 훨씬 작은 오차가 나왔다. GPU highest precision 대조 진행.
FD 일관성 실패를 동역학적 불안정으로 곧바로 읽으면 안 된다는 직접 검증이다.

후속: GPU highest matmul precision에서도 해당 gain/FD 오차는 동일했다.
따라서 TF32만의 문제라고 결론 내리지 않는다. 작은 파라미터 변위의 float32
표현·상쇄 오차와 비선형성을 구분해서 본다.

새 실제 제어 `jvp_fit_control`은 시작 reference의 유효성에 의존하지 않는다.
10k마다 독립 fixed fit/probe에서 20회 frozen-target critic fit의 JVP를 구하고,
Q parameter step 배율 1/.5/.25/.1 중 gamma × tested-direction gain <= .8인
가장 큰 배율을 선택한다. 실제 배율별 응답과 선택 결과를 모두 기록한다.
어떤 배율도 문턱을 만족하지 않으면 측정 gain이 가장 작은 배율을 선택한다.
이 두 방향의 RMS gain은 전체 B의 operator norm이나 joint-loop 안정성의 증명이 아니다.
같은 제어와 직접 residual target correction을 결합한 `jvp_residual`도 비교한다.

추가 후보: lagged expectile weights, 직접 G/kappa target correction (0.25/1.0),
actor baseline까지 함께 보정하는 대조. lagged weights는 target-V의 sign으로
가중치를 정하고 현재 V의 residual을 fitting한다. reference가 현재 V로 수렴하는
경우 원래 expectile 정상조건을 회복하지만 수렴 자체를 보장하지는 않는다.

조건부 kappa로 loss를 재가중해도 V가 realizable하지 않으면 projection metric과
실제 함수근사 최적점이 바뀔 수 있다. 따라서 "같은 residual을 더 잘 최적화"
한다고 자동 해석하지 않는다. 단독/결합 ablation과 validation 진단을 함께 본다.

## Residual을 정확히 없애도 가능한 발산: 작은 해석적 대조

`exact_residual_counterexample.py`는 두 상태, atomless 연속 behavior action,
선형 Q 함수근사에서 critic least squares와 expectile을 정확히 푼다.
모든 질의는 behavior support 안이다. V FOC residual은 상대 약 1e-17이다.

phi(s)=(1,2), state sampling=(.99,.01), 다음 상태는 항상 0,
reward=.1. action은 31/32 확률로 U[.99,1.01], 나머지는 U[9.99,10.01],
Q_theta(s,a)=theta*phi(s)*a. 테스트한 두 tau의 expectile은 두 action 구간 사이에
있어 폐형식으로 계산된다. critic fit도 action의 1·2차 moment로 정확히 계산한다.

| tau | gamma | 정확한 projected Bellman 반복의 scalar gain |
|---|---|---|
| .9 | .99 | .91908 |
| .9 | .999 | .92744 |
| .99 | .99 | 2.38622 |
| .99 | .999 | 2.40791 |

즉 residual=0에서도 높은 tau가 critic propagation 방향을 바꾸어 발산할 수 있다.
`||De_tau||=1`이라는 tau와 무관한 bound가 `B De_tau`의 실제 방향별 gain까지
tau와 무관하다는 뜻은 아니다. 이론을 반박하는 예가 아니라 식 M=gamma B De의
중요성을 보여주는 예다. OGBench 원인을 이 예만으로 확정하지 않는다.

critic ridge로 B를 줄이거나 target projection을 하면 이 예의 폭주는 막지만,
이것이 정확한 Q나 더 좋은 정책을 보장하지는 않는다. 그림/수치는 새 로그 루트
`analysis/exact_residual_counterexample/`에 있다.

## 외부 원문

- IQL: https://arxiv.org/abs/2110.06169
- OGBench: https://arxiv.org/abs/2410.20092
- SEEM / LayerNorm 기존 결과:
  https://proceedings.neurips.cc/paper_files/paper/2023/file/bd6bb13e78da078d8adcabbe6d9ca737-Paper-Conference.pdf
- Value / policy extraction 병목:
  https://arxiv.org/abs/2406.09329

이론의 작은 kappa만으로 실제 발산 원인을 확정하지 않고,
보정 작동 여부 → residual 변화 → 전파 응답 → advantage/actor 변화 → 성공률의
연결을 실험으로 확인한다.

## 첫 300k pilot 종료 후

cube seed 0 / 마지막 20 episode: setting-1 IQL .90,
setting-4 IQL .85, setting-4 target projection .85,
setting-4 critic LayerNorm .70. setting-4 scene은 이 시점에서 모두 성공률 0.
LayerNorm 단독도 setting 4 scene의 값 성장을 막지 못했다.
초기 100k의 성공률만으로 개선을 선언하지 않은 이유가 실제로 확인됐다.

setting-2 (tau=.9, gamma=.999) / 100k에서 LayerNorm은 cube .82, scene .68
(각 50 episode). 같은 gamma의 native baseline은 cube .85, scene .60
(각 20 episode)였다. 평가 표본 수가 다르고 아직 초기 단계다.
높은 할인율의 성능을 보존하는지 확인하기 위해 LayerNorm setting-2를
9개 환경 전체에서 1M / 50 episode로 검증하는 seed-0 실행을 대기열에 추가했다.
cube/scene/ant는 이미 시작했고 나머지는 빈 slot에서 자동 시작한다.
이것은 최종 선택/성공 판정이 아니며, seed 1/2 확인이 뒤따라야 한다.

진단: scene setting-4 100k에서 V 마지막 bias에 대한 full value-loop JVP의
Q 응답 RMS는 H20/H100/H500에서 baseline .606/.461/2.267,
projection .552/.746/3.102였다. projection은 실제 300k까지 훨씬 안정적이었지만
이 특정 local impulse gain은 더 작지 않았다. 따라서 이 gain 하나를
장기 안정성이나 spectral radius의 인증값으로 사용하지 않는다.

instrumentation 수정: 첫 ant LayerNorm 실행은 finite fraction의 float32 비율
.9999999403953552를 정확히 1과 비교해 1k에서 잘못 종료됐다.
GPU에서 동적 integer count의 나눗셈으로 이 값을 재현했다.
종료 판정을 boolean all-finite reduction으로 바꾸고 회귀 테스트를 추가했다.
원본 run은 `invalidated.json`으로 표시하고 보존했으며, `w8b_ant_...`로 재실행했다.
이를 알고리즘의 발산 사례로 집계하지 않는다. 수정 후 ant는 10k 이상 진행했다.

추가 후보 `negative_v`는 nonpositive reward의 zero upper return bound를
V=-softplus(-raw_V)로 구현한다. 정확한 expectile residual을 줄이기 위해
잘못된 양의 Q 추정을 더 증폭시키는 경로를 제한하려는 대조다.
OGBench 설치 소스에서 singletask reward가 nonpositive인 것을 확인했고,
runner도 데이터 max reward > 0이면 이 profile을 거절한다.

재현용 실행기: `algorithms/uk_offline/basic/iql_residual_control.py`.
진단 `diagnose_checkpoint.py --loop`는 Q/V/target/Adam 상태의 전체 value-loop를
미분한다. inner loss의 target은 parameter gradient에 대해 상수로 두지만,
outer sensitivity에서는 이전 state에서 target으로 이어지는 경로를 보존한다.
다섯 profile에서 real update와 sensitivity update의 primal Q/V/optimizer
state 일치 및 finite tangent를 CPU 테스트했다. 테스트 로그는 analysis에 있다.

새로 시작하는 run은 train 지표를 마지막 chunk만이 아니라 전체 probe interval로
집계하며, sampling / train / probe / evaluation / checkpoint / controller 시간을
구분해 저장한다. 이전 run의 지표는 당시 source snapshot과 함께 해석한다.

현재 목표는 미완료다. 9개 환경 최종 평균 > .19, 여러 seed 재현, 안정성 검증,
실패한 방법의 원인 분석을 완료하기 전에는 성공을 선언하지 않는다.

후속 선택: `bounded_v2` (V 2회 fitting + 갱신된 V로 backup + return-bound target
projection)는 setting-2 scene 100k / 50 episode에서 .88을 기록했다.
이 후보도 LayerNorm과 함께 동일한 9개 환경 전체 평가에 넣었다.
아직 초기 성능이며 장기 붕괴 여부와 다른 seed를 확인해야 한다.
실행과 후속 작업의 상태는 새 로그 루트 `CONTINUATION.json`에 기록했다.

## 평가 추론 병목과 공통 CPU 평가 프로토콜

16개 CUDA process가 경합하는 동안 동일 actor의 단일 상태 추론을 측정했다.

| 학습된 actor 크기 | 1,000회 GPU | 1,000회 CPU | 최대 action 차이 |
|---|---|---|---|
| 256 x 4 | 8.856 s | .0471 s | 1.54e-5 |
| 512 x 4 | 10.807 s | .1545 s | 9.24e-7 |

실제 ant 100k checkpoint의 5 episode 비교는 GPU 38.44 s / CPU 3.59 s였다.
성공은 각각 1/5, 2/5로 달랐다. 작은 action 차이가 chaotic dynamics와 이후 reset
sequence를 바꿀 수 있으므로 trajectory/성공률의 bitwise 동일성을 주장하지 않는다.
최종 후보와 새 IQL reference 모두 동일 CPU 추론 및 기존 seed/reset recipe로
평가한다. backend가 다른 최종 결과는 summary에서 다른 그룹으로 취급한다.

GPU 학습을 그대로 두고, snapshot의 model/optimizer/target/RNG를 보존한
`cpu_*` continuation으로 full runs를 옮겼다. 체크포인트 이후 미저장 update는
재실행될 수 있지만 최종 모델의 유효 학습 예산은 여전히 1M이다.
각 parent에는 `superseded.json`, child에는 parent checkpoint SHA와 계승 step을
기록했다. 이를 독립 seed나 알고리즘 실패로 세지 않는다.
`fork_run.py`는 parent 로그/모델을 보존하며 복사한 평가 기록에 원래 backend를
표시한다. 모든 model/optimizer/RNG 보존은 checkpoint SHA로 확인한다.

원본 IQL (tau=.9, original gamma)도 동일 실행기/CPU 평가로 9개 환경 seed-0
대조를 추가했다. cube/scene은 300k pilot의 정확한 optimizer/RNG 상태부터
연속 학습하며 actor cosine schedule은 처음부터 끝까지 1M 기준이다.
나머지 환경은 원래 설정의 0 step부터 시작한다. criterion .19를 넘는지만 보는
것에 더해 이 공통 프로토콜의 실제 IQL 성능도 대조한다.

긴 full runs가 원인 분석을 독점하지 않도록 대기열을 재정렬한다.
목표 배분은 full 12개와 짧은 실험 4개다. JVP fit control / JVP+residual이
실제 시작했고, 기록된 예에서 Q step .5가 선택되었다. 해당 시점에는 어떤 후보
배율도 gamma x tested gain <= .8을 만족하지 않아 `margin_met=false`도 기록됐다.
즉 보정은 작동하지만 그 문턱이 안정성을 보증한다는 뜻은 아니다.

첫 JVP fit control의 cube setting-4 결과: 299k에 Q RMS / return bound 104.51로
실패했다. controller 29회 측정 전부 finite, 17/29에서 tested gain margin 만족.
선택한 Q step 배율은 1.0: 3회, .5: 19회, .25: 6회, .1: 1회였다.
관측된 성공률은 100k .90 / 200k .70이며, 299k에서 중단했으므로 마지막
정책의 성공률을 0으로 추정해 기록하지 않는다.
같은 설정의 native baseline은 300k에서 아직 Q RMS 약 23.75 / 성공률 .85였다.
따라서 이 seed에서는 adaptive Q damping이 baseline보다 나빴다.
고정 Q step .5와 PRNG 5-key 초기화 대조를 다음 pilot 우선순위로 올렸다.

새 진단은 sampled validation action에 대한 frozen-Q expectile loss의 실제
20-step refinement 전후와 paired loss change의 standard error도 기록한다.
직접 residual correction head는 G/kappa와 sampled label의 global calibration,
local-action reference와의 방향 cosine을 함께 기록한다. 단일 score label의
RMSE를 조건부 G의 정확한 오차로 해석하지 않는다.

## Bellman index와 실제 gradient step의 차이

사용자 PDF의 식 (17)과 Section 4는 forcing term의 boundedness를 조건으로 둔다.
현재 B 진단은 초기 model/optimizer state를 고정하고 target만 바꾼다.
실제 partial fit에는 초기 model에서 다음 model로 직접 남는 carry 경로도 있다.
이 경로를 fitting error e에 포함할 수는 있지만, e가 이전 오차와 함께 자라는
상황에서는 uniformly bounded forcing 조건을 별도로 검증해야 한다.

위 atomless 선형 예에서 full-fit gain을 m, fit fraction을 u라 하면:

`theta_next = (1-u) theta + u fitted_target(theta)`

`actual gain = (1-u) + u*m`, 반면 고정 시작점 target-path gain은 `u*m`이다.
tau=.99, gamma=.999, u=.1에서 target-path gain=.24079,
gamma*||B_target||_inf=.61379 (<1)인데 actual gain=1.14079 (>1)이다.
u=.01에서도 actual gain=1.01408로 불안정하다. FOC residual은 여전히 0이다.
closed-form 결과를 central finite difference와 1e-9 이내로 대조했다.
`analysis/exact_residual_counterexample/warm_start_summary.json`에 수치가 있다.
이는 이론 자체를 반박하는 결과가 아니라, one-step/partial-fit B를 실제
Q/V/target/Adam 반복 전체의 contraction으로 해석할 수 없다는 구체적인 예다.

추가로 `numerical_control`은 새로운 intervention 없이 동일한 IQL 수식을
실험용 update 경로로 실행한다. 부동소수점 loss/exp 재배열만의 민감도가
개입 효과와 혼동되지 않는지 확인하는 대조이며 새로운 알고리즘으로 세지 않는다.

## 첫 1M 결과: LayerNorm 단독의 장기 실패

cube setting-2 LayerNorm seed 0은 CPU 평가에서 600k .92, 700k .74,
800k .82, 900k .90였지만 1M에서 .08이었다. 마지막 결과를 그대로 사용한다.
Q 평균은 900k -43.37에서 950k +2.04, 975k +51.98, 1M +32.34로 전환됐다.
Q RMS는 900k 46.22 / 1M 42.51로 오히려 작아졌고, 마지막 global G/kappa는
.023 정도였다. RMS 감소나 작은 global FOC 지표가 올바른 value나 좋은 정책을
보장하지 않는 직접적인 관측이다. nonpositive reward에서 이러한 양의 값은
물리적으로 타당한 return 범위를 벗어난다.

cube-double LayerNorm도 570k에 Q 평균 +6,798, V 평균 +6,807,
return-bound violation fraction .9995를 보였다. LayerNorm 단독을 성공 후보로
확정하지 않고 projection과 refinement를 결합한 bounded_v2의 나머지 6개 환경을
다음 full slots의 우선순위로 올렸다. LayerNorm runs는 대조 자료로 계속 추적한다.

동일 300k scene checkpoint를 CPU 50 episode로 재평가했을 때:
native setting-2 baseline .80, bounded_v2 .92였다. 한 seed의 중간 결과이며
표본 차이의 standard error가 약 .068이므로 유의한 우월성으로 단정하지 않는다.

`bounded_v2_independent`는 두 번째 V fit에 독립 minibatch를 사용한다.
별도 RNG를 checkpoint에 보존하므로 Q/actor replay 순서를 바꾸지 않는다.
동일 추가 batch를 쓰면 원래 v2와 state 차이 3.73e-9,
추가 batch만 바꾸면 V가 변하고 old-state actor update는 정확히 같았다.
2+2 step resume와 4 step 연속 실행의 모든 state 및 두 RNG가 정확히 일치했다.
검증 결과는 `analysis/independent_v*_validation.json`에 있다.
추가 V update를 같은 batch에 반복해 생기는 noise fitting과 실제 residual
최적화 효과를 구분하기 위한 대조다.

## 1M LayerNorm 붕괴의 heldout 재진단

Cube setting-2 seed 0 최종 모델에 validation 512개 query를 사용했다.
conditional 행동분포는 이웃 행동을 같은 query state에 재평가하는 근사다.
local residual RMS는 3.202에서 20회 독립 V update 후 3.084로 줄었으나,
16 vs 64 행동 reference의 RMS 차이가 1.145여서 oracle residual로 보지 않는다.
Validation expectile loss는 .8087 -> .7488, paired change -.05996 +/- .05603
(standard error)였다. 반면 nonpositive-return upper bound 위반 RMS는
target-Q 42.400, V 43.543 -> refit V 43.415로 거의 그대로다.
잔차를 더 fitting해도 이미 잘못된 Q의 값 자체는 교정되지 않는 예다.

LayerNorm scene도 최종 1M success .00, Q RMS 19,402.7을 기록했다.
모든 값은 finite여서 NaN나 overflow만 감시하면 이 실패를 놓친다.
이 결과들은 LN 단독의 안정성 개선 주장을 지지하지 않는다.

Bounded_v2의 backup projection은 매우 드물게 작동한다. 현재까지 cube
977k에서 clip fraction의 window max .0006914, scene 823k에서 .0005469였다.
희소한 projection이 핵심인지, 두 번째 V fit / fresh backup이 핵심인지
구별하려고 v2-only 및 independent-second-batch ablation을 short queue 앞에 둔다.
동일 batch 첫 update 전/후의 평균 training loss 감소만으로 validation
conditional residual의 감소나 성능 향상의 인과를 주장하지 않는다.

## Bounded_v2 첫 최종 결과와 고정 재현 계획

`cpu_w7_cube_bounded_v2_s2_seed0`이 1M Q update를 마쳤다. 최종 CPU 50 episode
success=.92, reward mean=-71.7, Q RMS=42.0567, V RMS=40.9504다.
이는 한 환경 / 한 training seed 결과이며, 9환경 평균 달성을 뜻하지 않는다.
제공된 Cube-S .84는 여러 seed 평균이므로 단일 .92와의 차이만으로
통계적 우월성도 주장하지 않는다.

`replication_plan.json`에 profile과 모든 학습/평가 설정을 고정했다.
9개 환경 각각 seed 1, 2 full runs를 추가해 재현을 확인한다.
기존 original-discount IQL 9개 대조에 더해 gamma=.999 IQL seed-0 9개도
추가했다. 후자의 cube/scene은 정확한 300k model/optimizer/RNG부터
이어가며 독립 seed로 세지 않는다.

full coverage와 재현이 현재 병목이므로 full slots 목표를 14개로 늘리고
2개는 핵심 mechanism ablation용으로 남긴다. 16개 전체 슬롯은 계속 채운다.

최종 Cube 모델 둘을 동일 validation state 512개 / 동일 neighbor bank로
재진단했다. 실패한 LN의 local-action residual RMS는 3.202, 성공한 bounded_v2는
4.891이었다. 비교하는 Q 함수 자체가 달라 raw residual 크기를 정책 성능의
순위로 해석할 수 없다. Validation target-Q upper-bound 위반 RMS는
LN 42.400 vs bounded_v2 .00342, V는 43.543 vs .14139다.
두 checkpoint SHA 및 모든 query 설정은
`analysis/cube_final_paired_diagnostic_512.json`에 함께 보존했다.

또한 Q와 V 모두에 같은 상수 c를 더하면 expectile residual 및 actor advantage는
변하지 않는다. Nonterminal TD residual에는 (1-gamma)c만 더해진다.
큰 gamma에서는 잘못된 공통 offset에 둔감한 진단이 될 수 있다.
하지만 순수한 공통 offset은 actor advantage도 바꾸지 않으므로 이것만으로
정책 성능 붕괴의 인과를 단정하지 않는다. Q/V의 상태별 왜곡이나 학습 이력을
별도로 통제해야 한다.

진단 출력 파일명에는 비기본 query/bank 수와 B 생략 여부를 붙이도록 수정했다.
서로 다른 해상도의 추가 진단이 기존 B/JVP 진단을 덮어쓰지 않게 했다.

## 붕괴 전 actor에서 Q/V만 교체한 통제 실험

Cube LayerNorm setting-2의 500k actor와 actor optimizer를 같은 출발점으로
사용했다. native actor update, 원래 1M cosine schedule의 500k optimizer count,
추가 20k minibatch 순서, CPU evaluation을 모두 공유한다. 한 군은 500k Q/V,
다른 군은 붕괴한 1M Q/V로 고정했다. 50 episode 결과:

| 고정 Q/V | 초기 | actor +10k | actor +20k |
|---|---|---|---|
| 500k 값 | .74 | .94 | .86 |
| 1M 값 | .74 | .02 | .04 |

초기 actor+optimizer SHA, 전체 replay index SHA, 초기 episode 성공 배열의
동일성과 Q/V가 끝까지 고정된 것을 확인했다. 최종 차이 .82의 단순 unpaired
binomial SE는 .0564다. 선택된 한 training seed / 50 episode 결과이며,
이 고정-value·20k-actor horizon에서 어떤 값을 쓰는지의 인과 효과를 보여준다.
Q의 action ranking 왜곡과 V에 따른 state weighting은 아직 구별하지 않는다.
순수한 공통 상수 offset은 advantage에 영향을 주지 않으므로, positive Q
자체만으로 정책 붕괴의 인과를 설명하지 않는다. 추가 actor 학습을 수행했으므로
이 결과는 예산이 같은 9환경 benchmark 후보에 포함하지 않는다.

세 번째 군은 같은 late target Q를 고정하고 V만 10k회 더 fitting한 뒤,
동일한 actor/optimizer/replay로 추가 20k actor 학습한다. V residual을 줄이는
개입이 이 actor 붕괴를 복구하는지 보려는 후속이며, 학습 예산 비교에서 제외한다.
실행/출력은 `analysis/cube_actor_counterfactual_plan.json`에 기록했다.

## 고정 Q/V에서 residual head 재학습

Cube direct residual setting-2 300k checkpoint에서 Q/V를 고정하고 moment head만
10k 추가 fitting했다. 같은 validation query 1024개의 G sample-label MSE는
.85936 -> .83167, change -.02770 (paired SE .01396)였다. Zero-prediction MSE는
.83464다. 근사 conditional-action 방향 cosine은 추가 fitting 뒤에도 -.1466이었다.
이는 moving target을 멈추면 일부 calibration 개선이 가능하지만, 현재 head가
신뢰할 수 있는 oracle residual이라고 볼 증거는 약하다는 관측이다.
Sample-label MSE에는 irreducible action noise가 포함되므로 낮은 R-squared만으로
조건부 G 추정 실패를 확정하지 않는다. Neighbor reference의 편향도 남아 있다.

운영상 보완: 정확히 마지막 1M step에서 수치 종료 조건이 생겨도 completed로
표시되지 않게 terminal status 우선순위를 수정했다. 지금까지 그런 최종-step
오표시는 관측하지 않았으며, 학습식에는 변화가 없다.

## Scene 최종 결과와 추가 구성요소 대조

Bounded_v2 setting-2 seed 0 Scene의 최종 1M / CPU 50 episode success=.94,
reward mean=-557.74, Q RMS=268.7906, V RMS=265.3230을 확인했다.
현재 동일 후보의 최종 coverage는 Cube-Single .92와 Scene .94의 2/9다.
나머지 7개 환경은 모두 학습 중이고 seed 1/2도 queued 상태다.
9환경 평균을 아직 계산하거나 성공을 선언하지 않는다.
`analysis/cube_scene_final_stability_comparison.png`에 두 환경 곡선을 보존했다.

추가 V fit 횟수(1/2), backup의 old/fresh V, return-bound projection의
2x2x2 구성요소 대조를 `factorial_plan.json`에 명시했다. 빠진 네 조합과
custom-update numerical-control을 두 환경에 추가했다. 새 조합들의 native
optimizer update 수 및 finite 동작은 protocol_validation_factorial.log에서
확인했다. 300k screening만으로 late-collapse 차이를 주장하지 않는다.
그래서 bounded-only cube/scene은 정확한 300k model/optimizer/RNG부터
1M까지의 full ablation으로 추가했고, v2_fresh_backup pilot도 종료 후
같은 방식으로 1M 대조를 이어갈 예정이다.

## V fitting을 개선한 뒤에도 남은 actor 붕괴

앞선 matched-actor 대조에 두 군을 추가했다. 모두 동일한 500k actor,
optimizer, actor replay SHA, 50-episode 초기 결과 .74를 확인했다.

| 고정 value에 대한 개입 | actor +20k success |
|---|---|
| 붕괴 전 500k Q/V | .86 |
| 붕괴 후 1M Q/V | .04 |
| 1M Q 고정, V 10k회 추가 native fit | .06 |
| 1M Q 고정, V batch4096 / step x.1 / 1k회 추가 fit | .00 |

Native V 10k 추가 fit은 validation loss를 1.3299 -> 1.7506으로 바꿨다
(paired change +.4207, SE .4839). 이 군은 residual을 성공적으로 줄인
개입이라고 볼 수 없다.

반면 큰 배치 / 작은 V parameter step 군은 같은 4096 validation query에서
loss 1.3299 -> 1.0678, paired change -.2621 (SE .08732)였다.
그런데 actor 성능은 복구되지 않았다. 여기서도 근사 local-action residual
RMS는 3.8563 -> 3.8642였으므로, 정확한 conditional sup-norm residual 제거를
검증한 것은 아니다. 검증된 결론은 이 frozen-Q 사례에서 out-of-sample
expectile fitting 개선만으로 좋은 actor를 복구하기에 충분하지 않았다는 것이다.

추가 V/actor compute를 쓴 분석이므로 위 결과를 9환경 benchmark 평균에
포함하지 않는다. 모든 SHA·episode·손실·소스는
`analysis/cube_actor_counterfactual_result.json` 및 각 arm 디렉터리에 있다.

## 재현 구성과 장기 ablation 후속

V2+fresh / no-projection Cube 300k pilot은 CPU 20 episode .85였다.
정확한 checkpoint/optimizer/RNG부터 `full_ablation_cube_v2_fresh_s2_seed0`로
1M 후속을 준비했다. Scene의 같은 pilot도 300k 종료 후 같은 방식으로 이어간다.

`utils/propagation_research/suite.py`는9환경 x3seed 실행 구성을 만들고,
runner/base-IQL/YAML/queue SHA를 검증한다. Queue 생성은 JAX를 초기화하지 않는다.
원래 discount에서는 override를 생략하고, 기본 최종 평가는1M / CPU50이다.
27개 unique job, V update count, shell syntax, 정상 hash 검증과 입력 변경 검출을
검증했다. 실제 재현 입력 스냅샷은 새 로그 루트 `reproduction/`에 보존했다.
실험 실행용 GPU manager를 추가 시작한 것은 아니다.

기존 native IQL 모델을 재평가해 fresh reference compute를 줄일 수 있는지도
확인했다. 확인한 OGBench native 모델은 일부 환경의 중간 checkpoint들이며,
기준의9개 환경 전체1M 모델 묶음을 찾지 못했다. Fresh matched reference를
중간 모델이나 기존 GPU score로 대체하지 않고 현재 대기열대로 이어간다.

## 보상식 감사와 Puzzle4x4 하한 구분

설치된 OGBench relabel source를 확인했다. Navigation은 success-1, manipulation은
sum(successes)-component_count다. 따라서 모든9환경의 upper return bound는0이다.
보수적인 global reward lower는 AM/HM/AS/Cube-S -1, Cube-D -2,
Puzzle3 -9, Puzzle4 -16, Scene -5다.

Puzzle4 데이터에서는 관측 reward가[-15,-1]였다. 기존 실행기는 terminal0을
포함한 경험적 return interval[-15000,0]을 사용했다(gamma=.999).
-15가 모든 가능한 상태에서의 global lower라는 별도 증명은 없으므로,
새 기준은 보상식으로 안전하게 유도한[-16000,0]을 사용한다.
다른8환경은 기존에 실제 사용한 interval과 알려진 interval이 정확히 같다.
`analysis/reward_bound_audit.json`에 env component 수, source SHA, 각 run의
실제 interval을 대조했다. 정확한 Q 또는 residual을 oracle로 제공한 것은 아니다.

`--bound-source ogbench`를 추가했다. 호환성을 위해 기존 실행기 기본값은
empirical로 보존하며, 재현 suite와 아직 미시작인 seed1/2의18개 job은 ogbench를
명시한다. 수정 전에 holdout seed1/2가 하나도 시작하지 않았음을 확인했다.
Puzzle4 seed0는 `full9_puzzle4_bounded_v2_known_s2_seed0`으로0부터 별도 학습한다.
기존 Puzzle4 empirical run은 별도 비교로 보존하며 결과를 섞지 않는다.

집계는 실제 numeric interval을 확인한다. 다른8환경에서는 같은 profile의
build_update에 들어가는 bounds가 정확히 같으므로 기존 결과가 그대로 같은
연산을 평가한다. Puzzle4의 서로 다른 interval은 다른 group으로 보고한다.
CPU smoke에서 실제 Puzzle4 데이터로 Q100 / V200 update, interval-16000,
identity의 bound source 기록 및 모든 state finite를 확인했다.
기존 완료된 Cube run의 identity도 호환됨(Already complete)을 확인했다.

재현 source pins는 `reproduction/bounded_v2_known_seeds012`와
`reproduction/baseline_original_seed0_v2`에 새로 준비했다. 이전 prototype도 보존한다.
이후 실제 전체9환경 검증과 seed 재현이 병목이므로 빈 슬롯은 우선 full run으로
채우도록 목표를16 full slots로 조정했다. 현재 실행 중인 short pilots는 계속한다.

V2+fresh / no-projection Scene pilot도 300k / CPU20에서 .60으로 종료됐다.
이를 exact checkpoint부터 `full_ablation_scene_v2_fresh_s2_seed0`의1M 후속으로
준비했다. Cube .85 / Scene .60의 screening score로 최종 효과를 판단하지 않는다.

## LayerNorm setting2: 첫 전체9환경 최종 대조

Seed0, tau=.9, gamma=.999, 모든 환경1M / CPU50의 평균은 .0666667이었다.
AM .08, HM-M .34, HM-L .04, AS .00, Cube-S .08, Cube-D .00,
Puzzle3 .00, Puzzle4 .06, Scene .00이다.
9개 immutable final checkpoint의 total_it와 Q/V/actor optimizer count가
모두1M인 것, 최종 episode 수/backend, 모든 최종 state finite를 직접 검증했다.
그런데 최종 Q 평균이 양수인 환경이6개였다. 수치가 finite라는 사실과
올바른 범위의 값 / 좋은 정책은 구별해야 한다. 한 seed의 이 대조군은
성능 목표 .19를 충족하지 못한다.

Puzzle4 데이터에서는 reward_max=-1, mask_mean=1이어서 학습 데이터에
완전 성공 상태가 없었다. 이는 bounds가 옳고 값이 안정적이어도 성능이
오르지 않을 수 있는 별도 데이터 제약이다. 이것만으로 모든 원인을
확정하지 않고, 현재 후보의 실제 최종 성능과 함께 해석한다.

## V loss의 기록 시점과 실제 post-update 추적

V2 profile의 live `value_loss_first`는 L(V0), `value_loss`는 L(V1)이다.
V2번째 gradient를 적용한 뒤 L(V2)를 기록한 것이 아니다. 앞선 설명의
"두 번째 update loss 감소" 표현을 "첫 update 전/후 감소"로 바로잡았다.
추가 fit의 효과를 이 두 log 차이만으로 판단하지 않는다.

새 `--v-trace`는 Q를 고정하고1/2/4/20회 fitting한 뒤 실제 손실을 다시
계산한다. 반복 첫 batch와 fresh batch를 비교하며, bootstrap next-state
근사 residual도 별도로 기록한다. 모든 trace point는 같은 V 평가 경로로
비교해 fused/eager 차이 몇 ulp가 초기 delta에 섞이지 않게 했다.

Cube 최종 후보에서 같은 batch2회 fit: Lbatch 1.2152 -> 1.1445,
Lval 2.4713 -> 2.4633 (paired delta -.0080, SE .0440).
Scene에서는 Lbatch 7.8828 -> 6.5220이나 Lval 33.8165 -> 34.2182
(delta +.4017, SE .1713)였다. Scene 같은 batch20회는 Lbatch2.3168까지
낮아져도 Lval37.9079였다. Fresh20회 Lval33.3787의 개선은 SE .8366으로
불확실성이 컸다. 이 관측만으로 추가 V update가 조건부 residual을
안정적으로 줄인다고 주장할 수 없다.

Neighbor action reference는 같은 query state에 행동을 재평가하지만,
빌려온 행동이 그 query의 진짜 behavior conditional support에 있다는
보장은 없다. 따라서 이 참조와 방향이 다르다는 사실만으로 head가 틀렸다고
단정하지 않는다. 본 후보의 실제 학습은 이웃 행동 질의를 사용하지 않는다.

## V를 더 잘 fitting해도 전체 반복은 더 빨리 불안정해질 수 있음

기존 atomless 선형 예에 critic relax fraction alpha, value relax fraction
beta와 K회 V fitting을 추가했다. w는 critic backup 기여로 rescale한 V이며
현재 theta에서 정확한 expectile target은 m*theta다. a=(1-beta)^K라 하면:

- w_next = a*w + (1-a)*m*theta
- old backup: theta_next = (1-alpha)*theta + alpha*(forcing+w)
- fresh backup: theta_next = (1-alpha)*theta + alpha*(forcing+w_next)

고정 Q에서는 V error가 정확히 a배로 줄지만, 두 coupled matrix 모두
det(I-M)=alpha*(1-a)*(1-m)이다. m>1이면 eigenvalue>1이 존재하므로
K를 늘리는 것만으로 안정화하지 못한다. m<1의 이 양의 선형 예에서는
전체 반복도 안정적이다.128개 matrix를 finite difference와 대조했다.

예: tau=.99, gamma=.999, alpha=beta=.1, m=2.4079.
K1 -> K20에서 fixed-Q V error factor .9 -> .1216으로 줄어도
fresh-backup spectral radius는 1.0597 -> 1.1368로 커졌다.
이는 OGBench Adam의 fitted model이나 이론 반박이 아니라, solver residual
감소와 coupled propagation 안정성의 차이를 명시적으로 보여주는 예다.
수치와 source는 `analysis/exact_residual_coupled_updates/`에 있다.

후보의 interval projection은 참 Bellman target이[L,U]에 있으면
|clip(y)-y_true| <= |y-y_true|이므로 pointwise target error를 늘리지 않는다.
그러나 신경망 critic fit과 optimizer carry까지 포함한 수렴 보장은 아니다.
추가 V fitting, fresh backup, projection의 역할은 현재 장기 ablation으로
분리해서 검증하고 있다.

## Ant 후보 최종과 paired stress baseline

Bounded_v2 setting2 seed0 AntMaze-Large가1M / CPU50에서 .22를 기록했다.
Q RMS173.5423, V RMS172.9077로 후반에도 음의 값이었다. 현재 확정된
최종 환경은 Cube .92 / Scene .94 / Ant .22의3개이고9환경 평균은 미완료다.

고할인율 자체와 안정화 개입을 구분하기 위해 Cube/Scene의 native setting2
baseline도 seed1/2를 추가했다. 기존 seed0와 합쳐3 seed paired 비교를 한다.
이 결정은 candidate holdout 결과를 보기 전에 기록했다.

동일 CPU protocol의 fresh original-discount Cube IQL seed0 최종은 .92였다.
후보의 .92와 동률이므로, 제공된 기준표 .84(여러 seed 평균)와만 비교해
Cube 개선을 주장하지 않는다. 전체 평균과 paired 고할인율 안정성을 확인한다.
첫 candidate holdout `replication_cube_bounded_v2_s2_seed1`이 실제 시작했고,
config에서 known-bound source / tau .9 / gamma .999 / seed1을 확인했다.
이후 holdout 결과를 수집하는 동안 선택한 update를 고정한다.

## 300k pilot의 공통 CPU50 재평가

기존 model을 추가 학습하지 않고 checkpoint SHA를 대조해6방법 x2환경을
CPU50으로 맞췄다. 평가 label50개가 모두0/1, 성공률 집계·보상 finite·SHA를
확인했다. 표는 training seed0 / tau .9 / gamma .999 / Q300k이다.

| 방법 | Cube | Scene |
|---|---|---|
| native IQL | .70 | .80 |
| bound target | .76 | .70 |
| V2 + fresh backup | .88 | .56 |
| bound + V2 + fresh | .98 | .92 |
| bound + independent V2 + fresh | .96 | .62 |
| nonpositive V + V2 + fresh | .86 | .70 |

Scene에서 grouped V2+fresh 효과는 clipping이 없을 때 -.24, 있을 때 +.22였다.
이는 한 seed의 조합 상호작용 관측이며 seed 간 인과 효과 추정은 아니다.
독립 V minibatch가 항상 좋지도 않았다. 현재 selected profile은 그대로
유지하고, 이 표를 최종1M 성능이나9환경 평균에 넣지 않는다.
원본 episode 배열, evaluator source와 hash, checkpoint SHA, CSV/그림은
`analysis/pilot300k_cpu50/`에 보존했다. 그림의 error bar는 episode 수준
Wilson interval이며 training-seed 변동성을 포함하지 않는다.

참고로 projection은 학습 operator 자체도 바꾸므로, 원래 unconstrained
projected-IQL의 fitting error만 제거했다고 해석하지 않는다. Target의
pointwise error 비증가 성질은 비교하는 참 target이 알려진 interval에
있다는 조건하에서의 성질이다.

Cube-Double 후보도1M / CPU50을 종료했고 success=.00이었다.
Q 평균-168.895, V 평균-167.430, Q RMS181.076으로,
LayerNorm의 큰 양수 방향 성장은 억제했지만 정책 성공에는 이어지지 않았다.
Q1M / V2M / actor1M의 실제 optimizer count와 전체 state finite를 확인했다.
범위 안의 값을 출력하는 것이 정확한Q나 좋은 actor를 의미하지 않는다.
현재 주 후보의 최종 coverage는4/9이며, 누락 환경을0으로 채우지 않고 평가한다.

## 추가 V fitting과 projection의 장기 분리 실험

모두 seed0 / tau .9 / gamma .999다. Bound-only는 Cube 최종 .86,
Scene 최종 .80이었다. 최종 Q 평균은 각각 -44.253, -265.745였다.
선정 조합(bound + V2 + fresh)은 .92 / .94이며, 아직 seed 간 효과는 미확정이다.
이 장기 ablation은 정확한300k checkpoint/optimizer/RNG에서 이어서
총 Q1M까지 학습한 결과다. 독립 시드나 추가1M 학습으로 집계하지 않는다.

V2 + fresh만 적용한 Cube는861k에서 Q RMS100153.08로 중단됐다.
마지막 실제 actor 평가는800k .00이며1M 평가는 관측하지 못했다.
따라서 V fitting 횟수와 갱신된 V 사용만으로 이 예의 발산을 막지 못했다.
Scene의 같은 조합도800k에서 Q RMS22658.98, success .02였지만,
그 시점은 중간 결과이므로 최종 성능으로 집계하지 않는다.

원래 할인율 fresh IQL의 Scene 최종은 .28(Q평균246.247)이었고,
고할인율 native IQL은 최종 .00(Q평균179509.40)이었다.
후자는 stop 기준인 Q RMS/return-scale100 미만이어서 completed지만,
참 반환값 상한0을 크게 위반하므로 안정적이라고 부르지 않는다.

Projection은 드물게 작동한다. Cube/Scene 후보의 기록된1k-window
backup_clip_fraction 최대는 각각 .000691/.000547이며, 최종 window도
.0005625/.00025였다. 이 CPU continuation의 Cube inherited1..400k 기록과
부모의 이후400..463k 기록은 다른 구간이므로 최대치를 혼합하지 않았다.
드문 위반 target을 누르는 것이 장기 drift를 바꾸었을 가능성은 있지만,
공통 custom-update의 FP 차이와 seed 효과를 분리해야 한다.
이를 위해 numerical_control 두 환경을1M까지 실행하고 있으며,
candidate seed1/2의 update는 고정한 채 검증한다.

## 실패 직전 전체 상태 보존 및 재현 검증

새 runner는 stop 발생 시 full ResearchState, optimizer, RNG를
terminal_checkpoint_STEP.pkl에 analysis_only/full_optimizer 표식으로 보존한다.
최신 정상 checkpoint는 그대로 두고, terminal model의 training resume는
거절한다. 모델 평가도 actor가 nonfinite면 거절한다.

고할인율 native Cube의700k 정상 checkpoint를 분석용으로 재생했고,
원본과 동일한786k에서 실패했다.701..786k의86 probe에서 비교한
9개 Q/V/TD/ESS/lag 지표가 모두 정확히 같았다. 정상 update와 I/O 변경을
분리한 실제 경로 검증이다. 종료 actor의 별도 CPU50 성공률도 .00이었다.
이 재생은 `analysis_replay_of_failure`로 표시해 모든 benchmark 평균에서
제외한다.786k 진단은 Q 상한 위반 RMS100619, V101786이었고,
fresh V20회 후 heldout expectile loss는727950 -> 826549였다.
오염된 Q에 V만 추가 fitting하는 것이 이 상태를 고치지 못했다.
재생·검증 원본은 analysis/replayed_cube_failure_actor.json과
runs/analysis_replay_cube_baseline_s2_seed0에 있다.

## 공통 상승과 행동 순서 변화의 분해

`advantage_drift.py`로 validation512 states와 training bank8192에서 빌린
16/32/64 actions를 두 checkpoint에 똑같이 질의했다. Q는 actor가 실제
사용하는 min(target-Q)다. Query index와 bank index, Q/V arrays, checkpoint
SHA, source를 보존했다. 이 행동들이 query state의 진짜 conditional
behavior support라는 보장은 없으며, 어떤 Q 순서가 참인지도 알 수 없다.

deltaQ(s,a)=c(s)+d(s,a), mean_a d=0으로 두면
deltaA(s,a)=[c(s)-deltaV(s)]+d(s,a)다. 이 분해는 수치적으로 원래
advantage 차이를 정확히 재구성한다. 공통 상수 상승, 상태 가중치 변화,
행동 의존 변화를 구분하는 기술적 진단이며 인과 효과 추정은 아니다.

같은300k ->1M 구간, 같은 query/bank를 맞춘 Cube seed0 /32행동 결과:

| frozen-model drift | LayerNorm IQL | 선정 bounded_v2 |
|---|---|---|
| Q 전역 이동 | +70.875 | +0.627 |
| V 전역 이동 | +71.302 | +1.172 |
| 상태별 advantage 이동 centered RMS | 4.976 | 1.970 |
| 상태 내 행동 의존 drift RMS | 5.107 | 1.567 |
| Q 행동 쌍 순서 역전 비율 | .4689 | .1428 |
| Q argmax 유지 비율 | .0918 | .5996 |
| uncapped 정규화 행동 가중치 TV | .6108 | .1591 |

LN의16..64행동 순서 역전 비율은 .469..476, 후보는 .135..146이었다.
후보가 안정적인 learned-Q 선호도를 보였다는 관측이며 정확한 Q ranking
또는 이 차이가 성능 개선을 매개한다는 증명은 아니다.500k ->1M의
기존 actor-counterfactual과 같은 두 model에서도 약47..48% 역전이었다.

고정 Q에서 cap이 없고 각 state의 policy를 독립적으로 완전히 표현할 수
있다면 normalized exp(beta*(Q-V))에서 V는 소거된다. V만 줄이는 개입은
그 조건하에서 행동 순서를 복구하지 못한다. 실제 구현의 weight cap,
shared neural policy, optimizer는 직접적인 V 효과를 남길 수 있다.
따라서 이 성질로 residual의 동적 Q propagation 역할을 부정하지 않는다.
공통 Q/V 상수 이동 불변성, 순서 전체 역전, V 소거를 작은 배열로 검증했다.
그림/표는 analysis/cube_advantage_drift_comparison.{png,pdf,csv}에 있다.

## 최종 검증 자동화 및 현재 coverage

`benchmark_audit.py`는 replication_plan의 candidate27 / original IQL9 /
paired stress6개 셀을 actual final full checkpoint와 episode arrays로
검증한다. Q1M, V2M(기준1M), actor Adam/schedule1M, state finite,
CPU50 binary 성공 labels와 집계, source profile, known numerical bounds를
확인한다. 추가 V compute는 숨기지 않으며 Q/actor-budget match라고 명시한다.
Incomplete 환경 평균은 null, stopped1M은 미관측, failure replay와
superseded continuation은 제외한다. 같은 seed의 최종 model이 여러 개면
평균내거나 최고점을 고르지 않고 ambiguous로 처리한다.
실제 checkpoint 기반의 변조 episode·잘못된 V budget·중복 seed·누락 환경·
replay/superseded 제외 검증은 analysis/benchmark_audit_validation.json에 있다.

Puzzle3 후보 최종 .02(Q평균 -306.751, Q RMS316.307)가 추가되어
seed0의 known-bound-equivalent 최종 coverage는5/9다. 전체 평균과
3seed 재현 목표는 여전히 미완료다. 느린 humanoid holdout과 원래 할인율
reference를 다음 빈 slot에 우선 배치해 검증 완료 시점을 줄였고,
기존16개의 학습은 그대로 유지했다. 변경은 queue 순서뿐이다.

현재 source를 pin한 재현 입력은 reproduction/bounded_v2_known_seeds012_v3
및 baseline_original_seed0_v3이며27/9개 queue와 source/config hash를
검증했다. 이전 prototype 입력은 당시 source와 함께 보존한다.

## Humanoid-Medium 최종 및 no-bound 장기 실패 추가

주 후보 HumanoidMaze-Medium seed0의1M / CPU50 성공률은 .56이었다.
최종 Q평균-180.125, V평균-179.621, Q RMS186.928이다.
실제 Q1M / V2M / actor Adam·schedule1M과 전체 state finite를 검증했다.
현재 주 후보 seed0는6/9 완료이며 HML, Soccer, 보수적 bound의 Puzzle4가
남았다. 전체 평균은 아직 계산하지 않는다.

V2 + fresh withoutbounds의 Scene도888k에서 Q RMS508268.06으로 중단됐다.
마지막 실제 평가는800k .02이며1M 점수는 미관측이다. 따라서 이 장기
ablation은 Cube861k와 Scene888k 모두 stop했다. 선정 후보와 bound-only는
둘 다1M에 도달했지만, numerical_control과 independent seeds가 끝나기 전
조합 차이의 모든 원인을 projection 하나에 귀속시키지 않는다.

## 실제 데이터에서 frozen backup 범위 위반 위치

`backup_distribution.py`로 환경별 동일한 training rows200000개를
비복원 추출했다. Cube/Scene 각각 candidate300k·1M, native300k·late
4개 model씩8개를 CPU에서 진단했다. Cube native late는 앞서 정확히
재현한786k terminal model이며 Scene은 실제1M이다. Frozen V와 float32
Bellman 식으로 계산하므로 online post-V2 minibatch 지표와 다르다.
모든 모델의 checkpoint/source hash와 원본 row index, Vnext, raw target을
보존했다.8개 모두 성공했고 같은 환경의 sample rows가 정확히 같음을 확인했다.

| 모델 | 상한 초과 /200000 | 초과량 평균(초과 시) | 초과량 최대 |
|---|---|---|---|
| Cube candidate300k | 6 | .232 | .663 |
| Cube candidate1M | 246 | .551 | 9.954 |
| Cube native300k | 8 | 1.289 | 5.201 |
| Cube native786k failure | 195870 | 100322.918 | 528189.813 |
| Scene candidate300k | 46 | 3.848 | 19.458 |
| Scene candidate1M | 71 | 2.065 | 12.076 |
| Scene native300k | 119 | 3.017 | 12.620 |
| Scene native1M | 199568 | 원본 JSON 참조 | 원본 JSON 참조 |

Scene candidate1M의 상한 초과71개 중68개가 reward=-1,3개가-2였다.
전체 sample에서 reward=-1은8230/200000=4.115%인데 초과집합에서는95.8%다.
목표 구성요소 하나가 남은 전이에 위반이 집중됐다. 반면 실제 다음 stored
state가 완전 성공(reward0)인 비율은, next_observation과 observations[i+1]가
정확히 일치하는 초과70개 중3개뿐이다. 따라서 모두 terminal/성공 경계에서
발생했다고 단정하지 않는다. 모든 관측된 upper violation은 비종료 전이였다.

이는 드문 target 제한이 어디서 작동하는지 보여주는 frozen-model 관측이다.
이 위치들이 전역 발산의 원인이라는 인과 증명은 아니다. 또한 fixed snapshot의
Cube 위반률 .123%와 online post-V2 window 지표를 동일한 값으로 취급하지
않는다. 비교 CSV, 그룹별 통계, 원본 배열과 검증은
`analysis/backup_distribution_cpu/`에 있다. 추가 training/GPU는 사용하지 않았다.

## 시드·resume 계보 검증과 독립 복원한 source bundle

`benchmark_audit.py`가 continuation의 부모 config, 원래 seed, profile,
학습 hyperparameters, inherited full checkpoint SHA, 실제 optimizer counter와
training RNG 존재까지 따라가도록 보강했다.6개 candidate final과 완료된
native3개가 통과했다. 가짜 독립시드 표식, 변경된 ancestor seed, 틀린 inherited
SHA를 임시 복사본으로 주입하면 모두 거절했다. 원본 파일은 바꾸지 않았다.
검증 파일은 `analysis/lineage_audit_validation.json`이다.

`capture_environment.py`로 repository source30개를 byte 단위로 export했고,
설치 package version, OGBench Python/XML source63개와 실제9환경의
train/validation NPZ18개 SHA256을 기록했다. Snapshot은
`reproduction/environment_snapshot_v1/`에 있다. JAX .4.1,
jaxlib .4.1+cuda11.cudnn86, Flax .6.1, Optax .1.3,
NumPy1.23.1, SciPy1.13.1, OGBench1.2.1, MuJoCo3.9.0 환경이다.

Bundle을 별도 임시 repository에 풀고 source SHA를 모두 대조했다.
공유 repository import 경로를 제거한 subprocess에서 실제 bounded_v2 CPU
update(Q2/V4/actor2)와 suite 준비가 성공했다. 이는 export의 실행 가능성
검증이며 training benchmark가 아니다. 검증은
`analysis/source_bundle_validation.json`에 있다. v1은 lineage 보강 전의
분석 utility를 포함한 당시 snapshot이므로 최종 완료 후 새 snapshot으로
갱신한다. 선정 training update는 그 사이 바꾸지 않았다.

사용법·선정 알고리즘의 update 순서·compute 차이·평가 원칙은
`utils/propagation_research/README.md`에 정리했다.

## 독립 seed1 stress baseline과 음수 값에서의 정책 하락

Native gamma=.999 IQL seed1도 Cube759k(Q RMS103420.01),
Scene719k(Q RMS509331.96)에서 stop했다. 마지막 실제 평가는 두 환경
모두700k이며 성공률은 .02/.00이다.1M 성능은 미관측으로 남긴다.
이 두 process는 terminal I/O 보강 전에 시작했으므로 terminal full state는
없다. 기존 log를 고치거나 미관측 checkpoint를 있다고 기록하지 않는다.

한편 같은 seed1 후보 Scene은 Q가 음수의 정상 scale에 머무르는데도
400k .80 ->500k .48 ->600k .22로 정책 성능이 떨어졌다.
이는 아직1M 최종 결과가 아니다.400/500/600k parameter archive와
실제600k full optimizer checkpoint를
`analysis/scene_seed1_policy_drop/`에 별도로 보존했다.

같은512 validation query에서400 ->600k의 frozen-Q/V drift를 측정했다.
32개 borrowed action에서 Q 쌍의 순서 역전 .1991,
상태별 advantage 이동 centered RMS7.3535, 행동 의존 drift RMS6.3200,
uncapped state-normalized action TV .5395였다. 전역 Q와 V 이동은
각각+11.6635/+11.1003으로 많이 상쇄되지만 상대적인 선호도는 변했다.
이 행동 집합도 oracle support가 아니므로 어떤 순서가 정확한지는 모른다.

실제 validation actions의 native beta10 weight ESS는 .1761 ->.2674로
올랐는데도 성공률이 하락했다. Scalar ESS를 좋은 actor의 충분조건으로
사용할 수 없다는 사례다. 같은 observed rows의 capped weight를 정규화한
old/late TV는 beta1/3/10에서 .6515/.7033/.6957이었다.
Frozen beta를 낮춘다고 이 사례의 가중치 이동이 사라지는 것은 아니었고,
이 진단만으로 beta를 바꾼 학습이 좋아진다고 예측하지 않는다.

GaussianPolicy는 state-independent log_std vector를 사용한다.
400k ->600k의 다섯 action std는 대략 .114/.119/.111/.156/.091에서
.111/.114/.107/.141/.088로 완만히 줄었다. 급격한 variance collapse를
원인으로 확정할 근거도 없다. 수치·입력 SHA·보존된 parameter SHA는
같은 폴더의 `temperature_sensitivity.json`에 있다.
모든 내용은 진단이며 candidate 학습 update를 수정하지 않았다.

## 추가 환경 완료 및 장기 ablation 검증표

Soccer seed0 최종은 .02, HML 최종은 .18(Q평균-229.756,
Q RMS237.934)이었다. 보수적 bound의 Puzzle4 최종 평가가 남아
그 전에는9환경 평균을 계산하지 않는다.

`analysis/audit_targeted_ablations.py`가 native/numerical/bound-only/
V2-fresh/bound-V2-fresh의 Cube/Scene10개 셀을 함께 기록한다.
현재 완료된5개 최종 model은 실제 checkpoint/optimizer/CPU50 episode/
resume 계보 검증을 통과했다.3개 no-bound 실패의1M 점수는 null이며
numerical controls2개는 계속 학습 중이다. 새 결과 후 script를 다시
실행하면 `final_ablation_audit.json` 및 `final_ablation_table.csv`를 갱신한다.

## 첫9환경 완료: seed0 평균 .322222

보수적인 reward bound의 Puzzle4가1M / CPU50에서 .04로 종료됐다.
9개 환경 모두 실제 최종 full model, Q1M / V2M / actor Adam·schedule1M,
전체 state finite와 CPU50 episode arrays, resume 계보 검증을 통과했다.
마지막 probe의 mean Q는9개 모두 음수이고 scale stop도 없었다.
이는 전체 state-space 값의 수학적인 안정성 보장이라는 뜻은 아니다.

| 환경 | 후보 seed0 최종 |
|---|---|
| AntMaze-Large | .22 |
| HumanoidMaze-Medium | .56 |
| HumanoidMaze-Large | .18 |
| AntSoccer-Arena | .02 |
| Cube-Single | .92 |
| Cube-Double | .00 |
| Puzzle3 | .02 |
| Puzzle4 known-bound | .04 |
| Scene | .94 |
| 9환경 macro | .3222222222 |

450개 실제 episode 중145개 성공으로 같은 평균을 교차 확인했다.
누락/중단된 환경을0으로 넣거나 best checkpoint를 고른 평균이 아니다.
제공된 IQL 원래 할인율 기준 .19는 초과했지만 방법 선정에 사용한
seed0의 결과다. 독립 seed1/2의 full9 최종 평가, 새 공통 protocol의
원래 할인율 IQL9환경 reference와 numerical/stress controls가 남아 있다.
따라서 전체 연구 목표는 계속 active로 둔다.
Milestone과 당시 model/episode audit 사본은
`analysis/seed0_full9_milestone.json`, `seed0_model_and_episode_audit.json`이다.

## 첫 독립 재현: Cube seed1 최종 .90, Scene seed1 최종 .26

Cube seed1도1M / CPU50을 마쳤고 success .90이었다(seed0 .92).
주 후보의 실제 model/optimizer/episode/seed 계보 검증을 통과했다.
반면 같은 gamma의 native Cube는 seed0/1/2에서 각각786k/759k/507k에
scale stop했다. Seed2의 full terminal state도 실제 보존되어 있다.
이 시점의 candidate seed2는 진행 중이므로 아직 candidate3seed 평균은 없다.

Scene seed1 최종은 .26이었다. 최종 Q 평균-248.84로 음수 scale은 유지했지만,
seed0 .94와는 정책 성능 차이가 컸다. 이는 범위 제한이 모든 정책 불안정성을
해결한다는 증거가 아님을 보여준다. Native Scene seed2는643k에서 stop했고,
seed1의719k stop과 seed0 최종 .00도 각각 그대로 기록한다.
중단된 두 seed의1M 점수는 미관측이다.

## 음수 값을 유지하는 Scene의 frozen-value actor 실험

앞서 보존한600k actor와 실제 optimizer(Adam/schedule600k)를 양쪽에
똑같이 사용했다. Q/V만400k와600k 모델로 바꾸어 고정하고,
같은 replay seed/indices로 native actor fitting을20k 추가했다.
추가 Q/V 학습은 없었다.400k parameter archive는 frozen Q/V 입력으로만
허용했고, 초기 actor나 V optimizer 재학습에는 archive를 허용하지 않았다.

| frozen Q/V | 초기 CPU50 | actor +10k | actor +20k |
|---|---|---|---|
| 400k | .24 | .48 | .66 |
| 600k | .22 | .26 | .34 |

초기 actor·optimizer hash, replay hash, source가 같고 Q/V가 고정됐음을
검증했다. 저장된 실제 actor state의 Adam/schedule counter도620k였다.
모든 평가50개 binary label과 finite reward, 성공률 집계를 대조했다.
다만 같은 초기 actor의 rollout도0-based episode46에서 하나 차이가 있었다.
49개의 나머지 episode는 성공 label뿐 아니라 reward도 정확히 같았다.
이 실행 간 차이의 원인은 aggregate만으로 확인하지 못했으므로,
동일 episode 결과나 paired-episode 추론을 가정하지 않는다.
환경 reset에는 실제 mj_resetData가 호출되는 것도 source에서 확인했다.

이 선택된 중간 하락 사례에서는 이전 Q/V로 actor를 더 잘 회복시켰다.
이는 음수 값의 scale을 유지해도 learned-value 선택이 actor fitting에
영향을 줄 수 있다는 추가 증거다. Q ranking과 V state-weighting의 기여,
정확한 residual 제거, full-training 인과 매개 경로는 분리하지 못한다.
추가 actor compute와 사후 선택한 case이므로9환경 benchmark에 포함하지 않는다.
원본·검증·그림은 `analysis/scene_seed1_policy_drop/actor_counterfactual_result.json`
및 같은 폴더의 비교 PNG/PDF다.

## 수식 구현 차이의3seed 비교와 경과표

Native update와 custom arithmetic의 작은 FP 차이가 late failure 시점을
바꿀 수 있어 numerical_control도 두 stress 환경의 seed1/2를 추가했다.
등록은1792920177이고 seed0 Cube numerical의 stop 관측 전이었다
(scheduler finished_at1792920442). 신규4개 job은 primary coverage/reference
뒤, 추가9환경 고할인율 baseline 앞에 배치했다. 선정 candidate는 그대로다.
Seed0 numerical Cube 자체도834k에서 Q 평균94203.52로 stop했다.
수식 재배열만으로 이 사례의 발산을 막지는 못했다.
Numerical Scene의 최종 결과와 신규 control seed들은 아직 미완료다.

`benchmark_audit.py`는 이제 candidate27 / original reference9 /
stress native6 / numerical control6개의48개 셀을 구분한다.
`stress_comparison.py`는 두 환경의18개 paired-seed 셀, 실제 경과,
scale/nonfinite stop, 관측된 policy drawdown을 따로 기록한다.
중단된 seed가 있으면 그 group의 final1M 평균을 계산하지 않는다.
실제18개 셀, matchingCPU50 point 수, native Cube3개 censored outcome과
null final mean을 확인했다.

새 비교 그림의 초안에 inherited GPU/20-episode pilot 점을 CPU50으로
표시한 오류를 찾아 수정했다. Candidate Cube/Scene seed0에서4/3개,
native Cube/Scene seed0에서3/3개 점이 제외됐다. 원본 기록은 유지하고,
정책 곡선과 drawdown은 이제 실제 CPU50 점만 사용한다.
기존 최종1M 평균은 원래부터 CPU50 검증이 적용되어 있어 영향을 받지 않는다.
Q/V probe 경과는 실제 학습의 고정 probe 기록을 사용한다.

## 후속 검증의 집계 원칙 보강

독립 seed의 일부 환경만 먼저 끝나면 환경별로1/2/3 seed를 섞은 평균이
시간에 따라 달라진다. `summarize.py`의9환경 평균은 이제9환경을 모두
마친 공통 seed cohort만 사용한다. 기존 available-seed 평균은 별도 이름으로
남긴다. 현재 common cohort는 seed0뿐이므로 .322222이며,
Cube/Scene seed1만 먼저 들어간 불균등 available mean .283333과 구분된다.
실제 데이터로 이 차이를 확인했고 `analysis/summary_common_seed_validation.json`에
기록했다. 최종 목표 판정은 여전히27개 셀을 요구하는 benchmark audit를 따른다.

Native stress seed2의 실제 full terminal checkpoint도 검증했다.
Cube507k와 Scene643k에서 Q/V/actor Adam·schedule counter가 각각 그 step과
같았고 전체 state는 finite였다. Q RMS는101938.93/504586.08이었다.
따라서 finite라는 조건만으로 안정적이라고 할 수 없다. 마지막 실제 CPU50
평가는 각각500k/600k .00이고1M 점수는 null이다. 원본 hash 및 실제
counter는 `analysis/paired_seed2_terminal_audit.json`에 있다.

## Cube3seed 완료 및 numerical Scene 최종

Cube 후보 seed2도1M / CPU50에서 .88로 종료됐고 실제 full model,
Q1M/V2M/actor1M 및 계보 검증을 통과했다. 세 seed의 .92/.90/.88은
평균 .90, training-seed sampleSD .02다. 실제150 episode 중135 성공으로
평균을 교차 확인했다. 독립 후속 seed1/2의 평균은 .89였다.
같은 gamma native Cube의 세1M 점수는 모두 censored이므로0 평균으로
바꾸지 않는다. 원래 할인율 fresh Cube seed0 .92와 후보 seed0는 동률이다.
이번 확인은 한 환경의 재현이며 전체9환경3seed 확인을 대신하지 않는다.
모델 hash, 실제 optimizer counter와 한계는
`analysis/cube_three_seed_confirmation.json`에 있다.

Numerical Scene seed0도1M에 도달했지만 최종 success .00이었다.
Q평균335436.75, Q RMS340392.99, V평균335370.04로,
수식 재배열만으로 이 사례의 값/정책 붕괴를 막지 못했다.
기존 scale-stop threshold100보다 Q RMS/return-scale이 작아 completed지만
안정적인 model이라고 부르지 않는다. 실제1M model과 CPU50 검증은 통과했다.
추가 numerical seed1/2 controls가 대기 중이므로3seed 인과 기여는 아직 미확정이다.
선정 candidate의9환경3seed 확인과 fresh original9환경 reference도 남아 있어
전체 목표는 계속 active로 유지한다.
현재 그림/표는 `analysis/paired_stress_trajectories.{png,pdf}`와
`paired_stress_comparison.json`, `paired_stress_table.csv`에 있다.

## Scene3seed 완료: 값 scale 안정화와 정책 변동의 구분

Scene seed2 최종1M / CPU50 성공률은 .66이었다. 실제 full model,
Q1M/V2M/actor1M, 전체 state finite와 계보 검증을 통과했다.
세 seed .94/.26/.66은 평균 .62, training-seed sampleSD .34176이다.
150개 실제 episode 중93 성공으로 교차 확인했다. 세 final probe의
Q 평균은 모두 음수였지만 정책 성능의 seed 차이는 컸다.
이는 값의 scale 안정화가 모든 정책 불안정성을 없앴다는 뜻이 아니다.
이1M seed2 결과는 추가 actor fitting의 .66과 숫자만 같으며 별개 모델이다.
원본 counter와 model hash는 `analysis/scene_three_seed_confirmation.json`에 있다.
이 시점에 주 후보는13/27개 최종 모델이 검증됐고 전체9환경3seed 평균은
아직 관측되지 않았다.

## 낮은 성능 환경의 데이터 점검

`dataset_profile.py`로 실제9환경 train/validation18개 split을 점검했다.
공식 loader의 regular mode는 마지막 state의 유효하지 않은 action을 제거하고
별도의 올바른 next observation을 보존한다. 원본 source와 실제 배열을 함께
확인했고, trajectory end를 제외한12,090,100개 인접 전이에서 mismatch0,
goal reward0 라벨과 mask0도 disagreement0이었다. 현재 학습 data의 크기,
state/action dimension, reward range, mask 평균도 모두 기존 metadata와 맞았다.
단순 next-state 연결 오류라는 가설을 이 데이터에서는 지지하지 않는다.

| 환경 | train goal frames / rows | validation goal frames / rows |
|---|---|---|
| Ant-L | 3444 / 1M | 512 / 100k |
| HM-M | 3921 / 2M | 791 / 200k |
| HM-L | 2450 / 2M | 338 / 200k |
| Soccer | 2123 / 1M | 157 / 100k |
| Cube-S | 20810 / 1M | 1691 / 100k |
| Cube-D | 581 / 1M | 58 / 100k |
| Puzzle3 | 2202 / 1M | 63 / 100k |
| Puzzle4 | 0 / 1M | 0 / 100k |
| Scene | 2198 / 1M | 404 / 100k |

Goal frame은 독립적인 성공 trajectory 수가 아니다. Puzzle4에 직접적인
완전 성공 anchor가 없고 Cube-D도 드물다는 관측은 남은 성능 문제의
맥락이지만, 조건부 행동 coverage 부족이나 task 불가능의 증명은 아니다.
실제로 Puzzle4 후보 seed0도 .04를 기록했다. Puzzle3과 Scene은 train
goal 빈도가 비슷한데 성능이 크게 달라, 이 빈도 하나로 설명할 수도 없다.
Train/validation의 경험적 state 분포 차이도 loss 비교 시 고려해야 한다.

표와 전체 histogram, source/loader hash, 검증은
`analysis/dataset_goal_profile.{json,csv}` 및 `dataset_goal_profile_validation.json`이다.
진단 당시 source도 보존했다. 이후 default EGL 설정과 output directory 생성만
보강했고, 학습 코드와 선택한 update는 바꾸지 않았다.

## 원래 할인율 Ant reference와 남은 큐 점검

Fresh native IQL setting1 / Ant seed0의 최종1M / CPU50은 .10이었다.
실제 full model, Q/V/actor Adam·schedule1M, episode arrays 및 계보 검증을
통과했다. 같은 평가 protocol의 candidate seed0는 .22다. 한 seed의
비교이며 유의성이나 전체9환경 평균의 결론으로 확장하지 않는다.
원래 할인율 reference의 완료 환경은 Cube .92, Scene .28, Ant .10으로
늘었고 나머지6개는 아직 미완료다.

모든18개 candidate 후속 seed1/2 run이 실제 시작됐고, 저장된 source의
initialize/build_update/make_scan AST hash와 selected profile, known bound가
그대로임을 확인했다. 앞으로 validation 중에는 이 core를 변경하지 않는다.
Queue의18개 후속 셀,9개 original-discount reference,4개 추가 numerical
control의 full1M/CPU50/seed/discount/bound 인자를 대조했다.
남은 필수 대기 작업이 optional exploration보다 모두 앞에 있으며 누락은 없다.
확인은 `analysis/remaining_queue_contract_audit.json`,
`selected_update_frozen_audit.json`에 있다.

## Humanoid-Medium original reference와 과거 model 재사용 점검

Fresh original-discount IQL seed0 HM-M의1M/CPU50은 .54였고 full model/
실제 optimizer counter/episode/계보 검증을 통과했다. 후보 seed0는 .56이다.
원래 할인율의 완료 reference는 Cube .92, Scene .28, Ant .10, HM-M .54다.
제공된 multi-seed 표보다 높은 일부 seed0 점수를 gain으로 오해하지 않으며
원래 정한9환경 fresh reference 전체를 완료한다.

추가 baseline 시드를 위해 과거 plain IQL model 재사용 가능성을 점검했다.
해당 original/.9 경로의 잔존 checkpoint는 일부 Puzzle 환경에 한정됐고,
예를 들어 Pz3 seed1 status는95k였다. 완전한9환경 공통 seed cohort를
대체하지 못하므로 primary benchmark에 넣지 않는다. 이 조회만으로
모든 checkpoint의 실제 optimizer count나 정확한 원본 source를 인증한 것은 아니다.
`analysis/historical_native_checkpoint_inventory.json`에 inventory/결정을 기록했다.
검증 범위는 기존48셀(candidate27, original9, stress6, numerical6)로 유지하고
추가18개 native run이나 새 방법은 등록하지 않았다. 후보 update는 고정한다.

## Cube-Double candidate seed1 최종 확인

선정 방법 seed1 Cube-D도1M/CPU50 성공률 .00이었다. 실제 Q1M/V2M/
actor1M, full state finite와 계보 검증을 통과했다. Final probe Q평균은
-174.10006841877475, Q RMS는187.36122816934497였다. 값이 음수
scale을 유지했지만 성공률은 오르지 않았다. 앞선 seed0도 .00이며
seed2는 아직 학습 중이다. 이 결과를 전체9환경 seed1 평균으로 바꾸지 않는다.
`analysis/cube_double_seed1_final_audit.json`에 검증 row를 보존했다.
후보 최종 검증은14/27이며, 같은 GPU 슬롯에 native original-discount
Cube-D seed0가 자동 시작됐다.16개 학습의 OS PID가 모두 live임을 확인했다.

## Humanoid-Large original reference 최종 확인

Fresh native original-discount HM-L seed0는1M/CPU50 .02였다. 실제
Q/V/actor1M, full state finite 및 계보 검증을 통과했다. 같은 seed의
선정 방법은 .18이지만 discount도 다르므로 이 단일 비교로 추가 V fitting의
인과 효과를 분리하지 않는다. Reference 검증은5/9이며 전체 평균은
아직 null이다. `analysis/hm_large_original_reference_seed0_audit.json`에
model hash와 episode 검증을 보존했다. 같은GPU에 original Pz3 seed0가
자동 시작됐고16개 학습 worker의 실제 OS PID가 live임을 확인했다.

## 탐색 범위 inventory

실제로100k 이상 학습한 현행 non-superseded/non-replay/non-invalidated
기록을 exact intervention, expectile/discount, initialization 및 유효 bound로
나눠 `analysis/exploration_inventory.json/csv`에 정리했다. 현재 profile 이름은
20개, exact setting group은30개다.
Group마다 실제 run ID와 단계/평가 device/episode 수를 보존하며 서로 다른
pilot 점수를 평균하지 않는다. Run count는 continuation 때문에 독립 seed 수와
다르다. 이 목록은 결과 인증이나 새 후보 선정을 대신하지 않으며, 고정한
48셀 validation plan과 학습 update를 바꾸지 않는다.

## Humanoid-Medium candidate seed1 최종 확인

독립 후속 seed1 HM-M은1M/CPU50 .50으로 종료했다. 실제 Q1M/V2M/actor1M,
전체 state finite 및 training lineage 검증을 통과했다. Final Q평균은
-178.49181712343125, RMS는187.3475303017782였다.
후보 seed0 .56, native original-discount seed0 .54와 구분해 보고하며
seed1 original reference가 없는 이 비교를 모든 시드의 paired 우위로
해석하지 않는다. 세 번째 후보 seed는 학습 중이며 전체9환경 seed1/2
cohort도 미완료다. `analysis/hm_medium_seed1_final_audit.json`에 보존했다.
Candidate 검증은15/27이고 같은 GPU에 native original Pz4 seed0가
자동 시작되어 original9환경 reference도 전부 시작됐다.16 worker가 live다.

## Puzzle3 candidate seed1 최종 확인

독립 seed1 Puzzle3은1M/CPU50 .00이었다. 실제 Q1M/V2M/actor1M,
전체 state finite와 계보 검증을 통과했고 final probe Q평균은
-282.1455100970343, RMS는293.5836411773714였다.
Seed0의 .02와 함께 이 환경의 남은 성공률 문제를 기록하되, 개별 환경의
낮은 성능만으로 고정 후보를 바꾸거나 전체 평균을 미리 판단하지 않는다.
`analysis/puzzle3_seed1_final_audit.json`에 보존했다. Candidate16/27이
검증됐고 첫 paired numerical Cube seed1이 같은GPU에 자동 시작됐다.
이 대조는 custom update의 FP 재배열만으로 차이가 설명되는지 확인하며
선정한 학습 update를 바꾸지 않는다. 실제16 worker가 계속 live다.

## 평가 reset의 fresh action-space 객체와 재현성 대조

Puzzle4 legacy/known-bound seed0 최종 Q/V/target/actor 및 optimizer 모든 배열이
array_equal(maxdiff0)였다. 그런데 nativeCPU50은 .02/.04로 달랐고 성공라벨은
0-based13/38/46, reward는50개 전부 달랐다. 최종 policy 차이나 하한 수정의
성능 효과로 읽으면 안 된다. `analysis/puzzle4_bound_revision_model_comparison.json`
에 model-tree hash와 원본 episode를 보존했다.

설치 OGBench ManipSpaceEnv.action_space는 접근마다 새 Box를 반환한다.
Reset warmup은 self.action_space.sample()을 호출하므로 한 임시 Box의 seed가
다음 새 Box의 RNG에 유지되지 않는다. 관측값만 비교한 첫 reset 시험은 같았지만,
Cube/Scene/Puzzle4의 full mjSTATE_INTEGRATION은 같은 env seed에서도 달랐다.
10개 zero-action 관측의 최대 차이는 각각8.33e-17/8.33e-16/1.77e-14였다.
동일 bounds/dtype의 Box를 local dynamic subclass에만 보존하고 seed를 주자
세 번의 반복 reset과10step 관측, full integration state가 모두 정확히 같았다.
설치 환경 source나 live training은 수정하지 않았다. 이 실험이 과거 모든
불일치 episode의 유일한 원인을 규명한 것은 아니다.
원본 array와 source hash는 `analysis/reset_action_space_physics_control.json` 및
NPZ에 있다. 이전 observation-only 시험의 hash는 서로 다른 임시 Box의
RNG hash이며 하나의 generator state를 반복 측정한 것이 아님을 명시했다.

어떤 seeded-space policy 점수도 보기 전에 별도 `seeded_evaluation_plan.json`을
등록했다. 주 학습/평가 protocol과48셀 validation은 유지하고, 모든 최종
candidate27와 original9의 frozenCPU50만 동일 cached-space seed 조건에서
재평가한다. Native 점수는 유지하며 secondary로 대체하거나 좋은 쪽을 선택하지 않는다.
학습 추가나 새 방법 선정도 없다. 먼저 bitwise-equal Puzzle4 actor 두 개를
별도 CPU process에서50회 평가해 success/reward arrays 전체 일치를 검증했다.
둘 다 .06이었고 primary .02/.04를 바꾸지 않는다. 현재 이 대조 통과 후
나머지36-model cohort의 준비된 모델을 CPU4slot에서 평가 중이다.
`evaluate_seeded_finals.py`, `seeded_reset.py`, evaluate_checkpoint의 명시적
flag로 재현하며 source/checkpoint hash를 검증한다. Primary runner는 그대로다.

## Humanoid 두 환경의3seed primary 확인

HM-M 최종은 .56/.50/.42, 평균 .493333(sampleSD .070238), 실제150회 중74성공이다.
HM-L은 .18/.06/.04, 평균 .093333(sampleSD .075719),150회 중14성공이다.
모두 실제1M/CPU50/Q1M/V2M/actor1M, finite full state와 계보 검증을 통과했다.
두 환경 모든 final probe의 Q평균은 음수다. SD는 per-seed50episode 점수의
산포이며 순수 training variance 성분만을 추정한 것이 아니다. 원래 할인율
reference seed0는 각각 .54/.02로, 한 시드 reference 대비 모든 시드 우위를
주장하지 않는다. `analysis/hm_medium_three_seed_confirmation.json`,
`hm_large_three_seed_confirmation.json`에 원본 model audit를 보존했다.
Puzzle4 seed1도 .10으로 종료되어 검증을 통과했다. 이들 결과가 추가된
시점의 candidate 검증은20/27이며 전체9환경 seed1/2 cohort는 아직 미완료다.

## Seeded-reset seed0 full9 및 Ant seed1

Secondary seed0의9환경이 모두 완료됐다. 실제450회 중144성공, 평균 .3200이며
모든 checkpoint hash가 primary model audit와 같았다. Primary .322222는
유지한다. `analysis/seed0_seeded_reset_confirmation.json`에 task별 두 점수와
model/actor hash를 보존했다. 새 protocol은 fresh env/global RNG 초기화도
포함하므로 모든 score차이를 action-space fix 하나의 효과로 해석하지 않는다.
Seed0는 여전히 방법 선정 시드이며 전체 독립seed1/2 검증을 대체하지 않는다.

Ant seed1 primary1M/CPU50은 .06으로 종료해 fullmodel/count/episode/lineage
검증을 통과했다. Seed0 .22와 차이가 크며, 원래 할인율 Ant seed0 .10에
비해 모든 seed가 개선됐다고 부를 수 없다. Finalprobe Q평균은-156.9535,
RMS170.8025였다. `analysis/ant_seed1_final_audit.json`에 보존했다.
Candidate21/27, original5/9이며 모든4개 numerical 후속 대조도 실제 시작됐다.

## 첫 독립 full9 followup seed1 완료

Soccer seed1이1M/CPU50 .06으로 종료해 검증을 통과하며 독립 seed1의
9환경 cohort가 완성됐다. AM .06, HM-M .50, HM-L .06, Soccer .06,
Cube-S .90, Cube-D .00, Puzzle3 .00, Puzzle4 .10, Scene .26이다.
평균은 .2155555556, 실제450회 중97성공이며 기준 .19를 초과했다.
모든9개 실제 model/count/episode/lineage가 통과했고 finalprobe Q평균은
음수였다. `analysis/seed1_full9_milestone.json`과 별도 원본 model audit에
보존했다. Seed0 선정 후 고정한 첫 independent followup이며, 유의성이나
전역 수렴의 주장은 아니다. Seed0보다 성능이 낮고 개별 환경의 실패도 남는다.

두 완료 seed0/1의 평균 .2688889와 미완료3seed 전체 평균을 구분한다.
Seed2의5환경, original4환경, numerical4후속, secondary36model검증 및
최종 보고/재현 bundle이 남아 있으므로 goal은 active다. Primary 학습/평가와
secondary평가를 혼합하지 않으며, 추가 Vcompute 차이도 계속 명시한다.

## 독립seed1 secondary full9 완료

동일한9개 frozen final actor의 별도 seeded-space CPU50도 모두 종료됐다.
실제450회 중108성공, 평균0.24이며 primary
.2155555556과 별도로 보존한다. 모든 model hash가 primary audit와 같다.
`analysis/seed1_seeded_reset_confirmation.json`에 task별 점수와 hash가 있다.
전체3seed 평균, original reference9환경 및 numerical controls는 여전히
미완료이며 새 점수로 primary를 대체하지 않는다.

## Secondary protocol의 global RNG 범위

설치 maze.py의 add_noise는565/566행에서 global np.random.uniform을
사용한다. Env private reset(seed)만으로 이 global RNG를 다시 seed하지
않으며 base.set_seed는 process 시작 때 global seed를 설정한다. 따라서
주 평가처럼 앞선 periodic eval을 거친 실행과, fresh frozen evaluator는
같은 정수 seed라도 global RNG history가 다르다. Secondary는 이 초기화도
통일한 protocol이며, nav 점수 변화까지 action-space cache의 단독 인과
효과로 해석하지 않는다. `analysis/secondary_rng_scope_source_audit.json`에
source hash와 해석 범위를 보존했다. 학습이나 원래 점수는 바꾸지 않았다.

## Cube-Double3seed primary 완료

Seed2의1M/CPU50도 .00으로 종료해 Q1M/V2M/actor1M, fullfinite state 및
계보 검증을 통과했다. Cube-D 세 점수는 .00/.00/.00이다. 모든 finalprobe
Q평균은 음수였으나 정책 성공률은 개선되지 않았다. 해당 환경을 전체 평균에서
빼지 않으며 데이터 sparsity 등을 이번 결과만으로 원인이라고 단정하지 않는다.
`analysis/cube_double_three_seed_confirmation.json`에 원본 세 model 검증을
보존했다. Candidate23/27, 독립seed2의 Ant/Soccer/Pz3/Pz4가 남아 있다.

## Ant 및 Puzzle3의3seed primary 완료

Ant seed2는1M/CPU50 .14로 종료했다. 세 점수 .22/.06/.14의 평균은
.14, per-seed score sampleSD .08이며 실제150회 중21성공이다. 모든
actualmodel/count/episode/lineage 검증을 통과하고 finalprobe Q평균도 음수다.
제공된 original-discount multi-seed mean .14와 같은 점수이고, fresh
original reference seed0 .10보다 높다는 단일 비교를 보편적 우위로 해석하지 않는다.
`analysis/ant_three_seed_confirmation.json`에 보존했다.

Puzzle3 seed2도1M/CPU50 .00이었다. 세 점수 .02/.00/.00은 평균
.0066667, sampleSD .011547이며150회 중1성공이다. 세 actualmodel/
optimizer/episode/lineage가 통과했고 finalprobe Q평균은 음수였다.
값 scale의 안정화가 이 환경의 성능을 해결하지 않았다. 실패 원인을
residual 하나나 goal빈도 하나로 단정하지 않으며 해당 환경을 평균에서
빼지 않는다. `analysis/puzzle3_three_seed_confirmation.json`에 보존했다.

이 시점 candidate25/27이다. 독립seed2에서 Soccer와 Puzzle4가 남았으며,
원래 할인율 reference4환경과 numerical4후속도 아직 진행 중이다.
16개 GPU worker와 secondaryCPUmanager의 실제 PID가 live임을 확인했다.
이번 turn은 새 최종 결과 전까지 live16worker의 증가하는 step을 확인한
verified wait였고, 종료 후 두 새 model 검증과 cohort artifact 작성으로
progress를 만들었다. 학습이나 고정한 secondary evaluator source는 수정하지 않았다.

## Original IQL Puzzle3의 드문 큰 값과 평균 진단 한계

Original-discount Puzzle3 seed0의687k fixed1024row probe에서 Q평균5.653,
Q RMS3980.778, centered RMS3980.774였다. 두 값은 모두 같은 ensemble-mean
Q로 계산된 것이며, p99 absolute Q는271.457이었다. 전체 RMS에 비해 대부분의
값은 훨씬 작아 이 fixedprobe의 tail이 RMS에 크게 기여했다. Centered/raw
RMS 비율 .999999는 공통 상수 이동만으로 이 진단을 설명할 수 없음을 보인다.
Twin-min의 upper0 위반 RMS도3969.866인 반면 critic disagreement RMS는
14.475였다. 두 critic의 일치가 올바른 값의 보장은 아니다.

이 기록은 전체 Q평균만으로 변동의 크기를 판단하기 어렵다는 구체적인 예다.
Global residual scalar의 한계는 상태 간 상쇄와 shared Q/V shift의 이론적
분석과 구분해서 해석한다. 단일 시점을 사후 선택한 illustrative diagnostic이며 tail의
모집단 빈도나 정책 실패의 단독 원인, residual/critic/propagation의 원인 분해를
추정한 것은 아니다. 실제 record와 probe/source hash, 정의는
`analysis/puzzle3_original_tail_diagnostic_687k.json`에 보존했다.
기존 성능 점수나 학습은 바꾸지 않았다.

## Soccer3seed primary 완료

Seed2의1M/CPU50 최종은 .02였다. 세 점수 .02/.06/.02의 평균은
.0333333, per-seed sampleSD .023094이며150회 중5성공이다. 모두 실제
Q1M/V2M/actor1M, fullstatefinite, episode 및 계보 검증을 통과했다.
세 finalprobe의 Q평균은 음수였으나 제공된 original-discount mean .06에
대한 task별 point gain은 없었다. 값의 scale 안정성과 성공률을 구분한다.
`analysis/soccer_three_seed_confirmation.json`에 보존했다.

Candidate26/27이며 마지막 primary는 Puzzle4seed2다. 전체3seed9task
mean은 아직 null로 둔다. Original reference4환경, numerical4후속,
secondary36finalmodel 검증과 최종보고/환경sourcebundle갱신도 남아 있어
goal은 active다. 실제16개 GPUworker와 pinnedCPUevaluator가 계속 live다.

At the same687k record, global_score_over_kappa=-2.094156 and score_RMS=13.204416.
The score compares target twin-min Q with V. A small global proxy does not
establish small statewise conditional residuals. This record does not identify
the origin of forcing versus critic fitting or propagation; keep these claims separate.

## All27 primary final models verified

Puzzle4 seed2 completed1M/CPU50 with success.02. All27 candidate final models
passed actualQ1M/V2M/actor1M optimizer/schedule, complete finite-state, episode,
profile/bounds and lineage audits. The9-task3-seed mean is exactly.2600,
351successes in1350episodes. Per-seed9-task means are.3222222/.2155556/.2422222;
the sampleSD of these3macro scores is.0555111. Independent fixed-followup
seeds1/2 average.2288889, both individually exceed the requested.19 threshold.
Every final probe meanQ is negative and no candidate hit the scale-stop rule.
This does not certify global contraction, exact residuals, or significant gains.

The complete task table, actual model audits and source-audit digest are saved in
analysis/candidate_full27_milestone.json, candidate_full27_model_and_episode_audit.json
and candidate_full27_table.csv. No task or training seed was omitted, no censored
score was filled with0, no best checkpoint was selected. Vcompute is explicitly
2M versus native1M. The main goal remains active: complete current-protocol
original references, numerical controls, separately registered secondary frozen
evaluation, final interpretation/figures and the current reproducibility bundle.

## Newly terminal comparison cases

Fresh original-discount CubeD seed0 completed1M/CPU50 success0 (Qmean-123.27,
RMS126.53). Original Puzzle3 also completed1M success0; its final Qmean-74.30
but RMS2347.80 again shows a negative mean is not a scale/correctness certificate.
Native-original reference coverage is now7/9, so its complete macro is stillnull.

The mathematical-loss numerical-control Cube seed1 completed1M success.92,
Qmean-43.22/RMS46.25, withQ/V/actor1M and full-model/episode audits passed.
Native same-gamma Cube seed1 stopped759k (no1Mscore); selected seed1 scored.90.
Implementation/numerical details can change the path even without the intended
residual/backup intervention. Do not attribute every single-seed difference to
projection or Vrefinement. These are finite50episode observations, not superiority
or significance claims. Numerical Scene seed1 instead hit the scale stop778k,
Q RMS500458 and allstatefinite; last actualCPU50 was700k success0. Its1Mscore
is unobserved/censored, never0-filled. Numerical seed2Cube/Scene still train.

## All27 secondary frozen evaluations verified

All27 candidate actors also completed the separately registered frozenCPU50
cached-space/fresh-process protocol. Actual364/1350 successes give mean.2696296;
seed0/1/2 macro scores are.3200/.2400/.2488889. The fixed-followup seed1/2 mean
is.2444444. Every checkpoint SHA matches the audited primary model and every
evaluator/helper SHA matches the pinned manifest. Equal-actor full50 validation
had already passed before this cohort was evaluated. Primary mean.2600 remains
unchanged; the better score is never substituted. Fresh global RNG initialization
is also part of the secondary protocol, so do not isolate every difference to
the cached-space fix. Full9 original reference and numerical controls still remain.
Evidence: analysis/candidate_secondary27_milestone.json, immutable summary snapshot
and candidate_primary_secondary_table.csv. No additional training or method tuning.

## Final comparison and resource closure

All48 fixed model outcomes passed the final audit:40 actual1M models plus8
censored scale stops. Primary9-task macro: proposed3-seed.260000 (351/1350),
fresh native original-discount seed0.213333 (96/450). Separate frozen protocol:
proposed.269630 (364/1350), native.224444 (101/450). Proposed fixed-followup
seed1/2 means are.228889 primary and.244444 secondary. No primary score was
replaced. The fresh native reference has1seed, so no significance claim is made.

Numerical Cube seed2 completed1M success.18 but Qmean7934.75/RMS8420.53;
completion did not mean safe values. Numerical Scene seed2 scale-stopped916k,
Q RMS501668.62; its1Mscore remains unobserved. Its terminal fullstate and Scene
seed1 terminal778k both passed actualQ/V/actor-counter and finite-state checks.
All40 actual1M models also have completed independent deep diagnostic records.

After all required model experiments ended, optional queue launches were stopped.
Only15 own optional process groups were terminated after verifying module,
output directory, PGID and exclusion from all48requiredIDs; one listed job was
already terminal. No required experiment was cancelled, and no checkpoint or
historical user log was deleted. All own training/managers exited and GPUs were
released. Cancelled optional stages are labelled and never called divergences.

Final result/source artifacts: final_benchmark_comparison.{json,csv,png,pdf},
paired_stress_trajectories.{png,pdf}, final_resource_audit.json and current22-profile/
33-setting exploration inventory. Figures were visually inspected. Archive-only
execution exposed a Git-metadata dependency, so packaging now tolerates absent
Git revisions while retaining source hashes. All training/probe functions,
classes and profiles outside CLI main remained AST-identical. Final27/9 suites
were prepared and verified without launching training; historicalv3 files remain.
The current environment/source export and isolated restore validation are recorded
in reproduction/environment_snapshot_final and analysis/final_source_bundle_validation.json.
