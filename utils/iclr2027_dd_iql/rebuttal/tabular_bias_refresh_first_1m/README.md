# Current-first tabular DF/SD/DD

기존 코드와 로그는 보존하고 새 파일로 작성한 4-state/4-action 실험입니다.
각 블록 첫 업데이트 직전에 현재 target-Q + 현재 noise와 현재 V를 복사해
filter 전체를 만들고, 해당 업데이트부터 D회 동안 같은 filter를 사용합니다.
실제 noise 시차는 각 블록에서 **0, 1, …, D−1**입니다.
이전의 업데이트 후 snapshot 갱신 방식이나 호환 옵션은 제공하지 않습니다.

SD는 자기 head로 filter를 만들고, DD는 다른 head로 만듭니다.
DF-min은 두 head의 min으로 filter와 V의 receiver를 구성합니다.
Receiver readout은 매 업데이트의 target-Q + 현재 noise이며, Q는 현재 V로
MSE/EMA 업데이트하고 V는 고정된 expectile weight로 업데이트합니다.
Q/V 학습 후 target-Q를 EMA로 갱신합니다. 행동은 공통 uniform 데이터이고,
최종 greedy action과 max-Q bias는 clean ensemble-mean Q에서 계산합니다.

## 설정과 실행

- S=A=4, N=2, 1M updates, seed 200–219.
- DF-min, SD-IQL, DD-IQL: 3 methods × 2 noise × 2 τ × 4 D = 48 conditions, 960 seed runs.
- IID / NN-like field (ρ=0.99), τ=0.9/0.99, D=1/10/100/1000.
- γ=0.999, αQ=αV=0.1, target EMA=0.005, σ=1.
- 기존 실험과 같은 입력 noise와 행동 난수열을 사용합니다.
- 모든 보상은 0이며 종료가 없으므로 Q*=V*=0입니다.

저장소 루트에서:

```bash
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_1m/test_engine.py
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_1m/run_experiment.py --workers 48 --noise-workers 12
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_1m/report.py
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_1m/validate.py
```

결과: `logs/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_s4_1m/`.
`run_experiment.py --dry-run`으로 설정을 확인할 수 있습니다.
48개 조건은 동일한 seed별 입력을 공유하며 프로세스별로 독립 실행됩니다.
두 noise의 전체 배열 약 10.24 GB를 fork로 공유합니다.

## 지표와 진단

- 최종 max-Q bias: mean_s max_a mean_i clean Q_i(s,a).
- 순간 filtering output: F=sum_a(wY)/sum_a(w), noise-free reference 0.
  이는 진단용이며 학습의 expectile SGD를 대체하지 않습니다.
- Filtering bias/RMSE, learned Q/V bias/RMSE, over/under를 구분해 저장합니다.
- Oracle, readout, TD, drift, LOO와 weight의 temporal/cross-sectional coupling을 기록합니다.
- 학습 직전의 실제 snapshot·weight·backup과 noise 시차를 저장합니다.
  진단 sampling은 별도 RNG로 구성하며 학습 RNG를 소비하지 않습니다.
- 같은 1M 실행에서 100k 체크포인트도 저장합니다.
- 마지막 100k의 sparse 진단은 각 상태·행동당 101개 시점입니다.
  독립인 경우에도 평균 절대 표본 correlation은 약 0.08일 수 있으므로,
  signed correlation·covariance와 함께 해석해야 합니다.
- 평균 ± 표본 표준편차는 20 seed 기준입니다.

## 이전 버전 보존

기존 `../tabular_bias_1m/`는 원래 상태를 그대로 유지합니다.
추가 백업은 `../archive/tabular_bias_1m_before_refresh_fix_20261112/`에 있습니다.
기존 두 결과 폴더 `tabular_bias_1m/`, `tabular_bias_diagnostics_s4_1m/`도 유지합니다.
새 학습 코드는 기존 패키지를 import하지 않으며 새 timing만 구현합니다.
기존 D=1 learned Q/V와 noise가 비트 단위로 같은지도 별도로 검증합니다.
