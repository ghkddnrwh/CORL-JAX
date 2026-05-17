# Offline RL on a Two-Path Branching MDP

이 코드는 **단일 seed 실험만** 수행하도록 정리된 버전입니다.

## 환경

- 상태:
  - `s_0`
  - short branch: `s_1_1, ..., s_1_{N-1}`
  - long branch: `s_2_1, ..., s_2_{2N-1}`
  - shared terminal: `s_1_N = s_2_{2N}`
- `s_0`에서는 action 2개만 유효:
  - action `0`: short branch로 진입
  - action `1`: long branch로 진입
- branch 내부에서는 action `A`개가 유효하고, 모두 전이상 동일합니다.
- terminal에 진입하는 transition에서만 `reward_value`를 받고 episode가 종료됩니다.

## 중요

`gamma < 1`일 때만 short path가 유일하게 optimal입니다.

- short path return: `reward_value * gamma^(N-1)`
- long path return: `reward_value * gamma^(2N-1)`

## Offline buffer

버퍼에는 정확히 `2A`개의 trajectory가 들어갑니다.

- short branch + branch action `0`
- short branch + branch action `1`
- ...
- short branch + branch action `A-1`
- long branch + branch action `0`
- ...
- long branch + branch action `A-1`

## 구현된 알고리즘

- `clipped_q_learning`
  - target network 사용
  - clipped min-Q target 사용
  - `sample` / `minibatch` 업데이트 모드 지원
  - `minibatch` 모드에서 batch 후 Q-table 전체에 Gaussian noise 주입 가능
- `monte_carlo`
  - full Monte Carlo return target 사용
  - `sample` / `minibatch` 업데이트 모드 지원
- `frozen_ratio_value_learning`
  - `Q(s,a)`와 `V(s)`를 함께 학습
  - `Q_f, V_f` snapshot을 이용해 `V` 업데이트 강도를 조절
  - `sample` / `minibatch` 업데이트 모드 지원

## 설치

```bash
pip install -r requirements.txt
```

## 실행 예시

```bash
python -m offline_rl_mdp.run_experiment \
  --algo clipped_q_learning \
  --N 10 \
  --A 4 \
  --gamma 0.99 \
  --num-epochs 500 \
  --learning-rate 0.1 \
  --update-mode sample \
  --seed 0 \
  --save-path ./results
```

또는 프로젝트 루트에서:

```bash
python run_experiment.py \
  --algo clipped_q_learning \
  --N 10 \
  --A 4 \
  --gamma 0.99 \
  --num-epochs 500 \
  --learning-rate 0.1 \
  --update-mode sample \
  --seed 0 \
  --save-path ./results
```

## 출력

학습이 끝나면 terminal에는 아래만 출력됩니다.

- chosen branch
- learned return
- optimal return
- return gap
- save path

## 저장 경로

각 실험은 항상 seed별로 아래에 저장됩니다.

```text
{save_path}/{seed}/
```

예:

```text
./results/0/
./results/1/
```


## 저장된 결과 집계

단일 seed만 학습하더라도, 저장된 여러 seed 폴더를 나중에 다시 모아 평균 성능을 계산할 수 있습니다.

```bash
python aggregate_saved_runs.py   --root ./results   --seed-start 0   --seed-end 99
```

또는 패키지 방식으로:

```bash
python -m offline_rl_mdp.aggregate_saved_runs   --root ./results   --seed-start 0   --seed-end 99
```

이 스크립트는 각 `root/{seed}/`에서 저장된 `env.json`, `algorithm_config.json`, `greedy_policy.json`을 다시 읽어서 policy를 재평가하고,
다음 값을 평균내어 출력합니다.

- mean learned return
- mean optimal return
- mean return gap
- short path selection probability
- optimal path selection probability

그리고 `root/aggregate__seed_{start}_to_{end}.json` 파일도 같이 저장합니다.

## 주요 저장 파일

- `env.json`
- `dataset.json`
- `summary.json`
- `algorithm_config.json`
- `train_history.csv`
- `q_error_curve.png`
- `optimal_q.json`
- `greedy_policy.json`

앙상블 Q 알고리즘(`clipped_q_learning`, `monte_carlo`)은 추가로:

- `q_tables.npy`
- `mean_q.npy`
- `min_q.npy`

FRVL은 추가로:

- `v_values.npy`
- `fixed_q.npy`
- `fixed_v.npy`

## Update mode

모든 알고리즘은 두 가지 업데이트 모드를 지원합니다.

- `--update-mode sample`: transition 하나씩 순차 업데이트
- `--update-mode minibatch`: `--batch-size` 기준 batch 업데이트

체인형/두갈래형 환경에서는 `sample` 모드가 더 빠르게 reward를 전파하는 경우가 많습니다.
