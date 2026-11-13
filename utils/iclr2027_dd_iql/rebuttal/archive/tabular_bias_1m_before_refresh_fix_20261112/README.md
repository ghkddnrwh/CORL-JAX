# Tabular bias: six algorithms, one million updates

Min Q, Double Q, DF-min, DF-mean, SD-IQL, DD-IQL의 재현 코드입니다.
Greedy(Q-learning), reversed Min Q, Mean Q는 이 패키지의 실험 대상에서 제외했습니다.

## 결과 위치

저장소 루트 기준 `logs/iclr2027_dd_iql/rebuttal/tabular_bias_1m/`입니다.

- `raw/`: 272개 조건의 NPZ와 원본 메타데이터. 20 seed의 학습 기록, 1M Q/V와 100k 체크포인트를 포함합니다.
- `figures/`: S=A=2·4·8·16 각각의 100k·1M bias 대 D 그래프, 총 8개 PNG입니다.
- `tables/`: `tabular_bias_1m.xlsx`, 전체 수치와 seed별 CSV입니다.
- `tables/latex/`: 그림에 대응하는 LaTeX 표 8개입니다. 각 표는 noise × τ의 네 블록으로 나누고, D와 무관한 baseline은 네 열을 합칩니다.
- `provenance/`: 원본 파일 해시, 이동 내역, 수치 검증과 실행 환경입니다.

4-state·1M에서 filtering output과 oracle 오차/coupling을 함께 기록하는 추가 실험은
[diagnostics 안내](diagnostics/README.md)를 참고하세요. 기존 업데이트와 결과를 보존하는 replay입니다.

## 그림과 표 다시 만들기

저장소 루트에서 실행합니다. 이전 로그 폴더와 기존 100k 결과는 필요하지 않습니다.

```bash
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_1m/plot_results.py
```

다른 위치의 데이터를 사용하려면 `--results-dir /path/to/tabular_bias_1m`을 지정합니다.
`--check-only`는 원시 데이터 해시와 bias 식만 검사합니다.
`--latex-only`는 LaTeX 표 8개만 생성합니다. 표에는 `booktabs`, `graphicx`, `wrapfig` 패키지를 사용합니다.
그림 생성에는 NumPy, Matplotlib, openpyxl이 필요합니다.

## 업데이트 구현 검증

```bash
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_1m/validate.py
```

Numba 실행기를 로컬 `reference/`의 NumPy 구현과 72개 설정에서 비교하고,
272개 저장 결과의 Q 테이블에서 bias를 재계산합니다.
추가 의존성은 Numba와 SciPy이며, 원래 실행 환경의 버전은 결과 폴더의
`provenance/original_environment.json`에 있습니다.
`requirements.txt`에는 Python 3.10.9에서 검증한 라이브러리 버전을 고정했습니다.

## 실험을 새로 실행하기

```bash
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_1m/run_experiment.py --dry-run
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_1m/run_experiment.py \
  --results-dir logs/iclr2027_dd_iql/rebuttal/tabular_bias_1m_rerun
```

기본값은 8개 실험 프로세스와 4개 noise 생성 스레드입니다.
`--workers`, `--noise-workers`로 조정할 수 있습니다.
`--sizes`, `--noise-modes`, `--methods`로 일부 조건만 실행할 수도 있습니다.
이미 완료된 NPZ는 해시를 확인하고 건너뛰므로 중단 후 재개할 수 있습니다.
16-state/16-action에서는 공통 noise 배열만 약 82 GB RAM을 사용합니다.

## 실험 정의

- S=A=2,4,8,16; 모든 보상은 0, 종료 없음; 다음 상태는 `action % num_states`.
- uniform 행동 데이터; N=2; seed=200..219; 각 seed마다 1,000,000 업데이트.
- γ=0.999, αQ=αV=0.1, Q-target EMA=0.005, σ=1.
- IID와 NN-like field(ρ=0.99), τ=0.9/0.99, D=1/10/100/1000.
- Min Q와 Double Q는 τ와 D를 쓰지 않는 baseline이므로 각 크기·noise당 한 조건입니다.
- bias는 각 seed의 `mean_s max_a mean_i clean Q_i(s,a)`이며 참값은 0입니다.
  평균 ± seed 간 표본 표준편차를 표시하며, 학습된 V bias는 별도입니다.
- 100k 값은 동일 NPZ의 Q 체크포인트에서 계산합니다.
- D=1은 현재 관측값으로 필터를 정하고, D>1은 업데이트 후의 noisy Q-target/V를 D마다 고정합니다.
- 상태·행동 크기를 같이 늘리며 전체 업데이트 수를 고정하므로 항목당 방문 횟수가 다릅니다.
  1M 결과를 수렴값으로 가정하지 않습니다.

`engine.py`는 기존 1M 수치 실행기를 보존하고 공개 메서드 목록을 여섯 개로 제한했습니다.
`reference/`의 수식은 원본과 같으며, 외부 코드에 의존하지 않도록 helper import만 바꿨습니다.
