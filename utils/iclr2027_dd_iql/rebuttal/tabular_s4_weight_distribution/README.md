# Four-state Q / weight distribution inspection

기존 current-first 4-state, 1M 로그의 실제 gate와 Q/V/noise를 읽습니다.
48조건 × 20 seed를 분석하며 재학습이나 원본 변경은 하지 않습니다.

```bash
python utils/iclr2027_dd_iql/rebuttal/tabular_s4_weight_distribution/analyze.py
```

입력:
`logs/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sizes_1m/raw/s4_a4/`

출력:
`logs/iclr2027_dd_iql/rebuttal/tabular_s4_weight_distribution/`

- `weight_q_distribution_report.pdf`: 분포, 학습 궤적, D별 요약 16쪽.
- `figures/`: 각 그림의 PNG/PDF.
- `last_window_summary.csv`: 48조건의 마지막 100k 집계.
- `per_seed_metrics.csv`: 960개 trajectory별 집계.
- `time_summary.csv.gz`: 진단 시점별 seed 평균과 표본 SD.
- `example_snapshots.csv`: seed200의 4개 state와 모든 action/head의 Q/V/weight.
- `source_manifest.json`, `verification.json`: 원본 경로/해시와 gate 검증.

핵심 구분:

1. Clean learned target Q와 noise가 섞인 source readout을 나눠 봅니다.
2. State별 clean Q 평균을 뺀 분포로 공통 수준과 행동별 돌출을 구분합니다.
3. 실제 frozen source Q/V로 weight를 검증합니다.
4. 시간에 걸친 pooled ESS와 state 안의 네 행동 ESS를 구분합니다.
5. Clean Q의 행동 간 차이만 없애는 counterfactual은 V와 noise를 고정합니다.
   학습 이력까지 바꾸는 인과 실험으로 해석하지 않습니다.

Python 의존성은 numpy, scipy, pandas, matplotlib입니다.
분석용 그림은 tau=.99에 초점을 맞추고, 표에는 tau=.9도 포함합니다.
