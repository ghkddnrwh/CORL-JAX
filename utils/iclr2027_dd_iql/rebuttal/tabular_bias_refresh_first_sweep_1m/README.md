# Current-first tabular size sweep: Double Q, DF, SD, DD

기존 4-state 패키지는 그대로 보존하고 별도 파일로 확장했습니다.
S=A=2,4,8,16, 1M updates, 20 seeds (200–219)입니다.
DF-IQL·SD·DD는 IID/NN-like × τ=0.9/0.99 × D=1/10/100/1000으로 실행합니다.
Double Q는 τ/D와 무관하므로 각 크기·noise마다 한 번만 실행합니다.
전체 200조건·4000 seed 실행 중 완료된 current-first S=4의 48조건은 복사해 재사용하며,
새로 152조건·3040 seed 실행을 수행합니다. 기존 코드와 원시 로그는 수정하지 않습니다.

## 실행

```bash
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sweep_1m/test_engine.py
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sweep_1m/run_experiment.py --workers 48 --noise-workers 12
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sweep_1m/report.py --size 2 --workers 8
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sweep_1m/report.py --size 4 --workers 8
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sweep_1m/report.py --size 8 --workers 8
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sweep_1m/report.py --size 16 --workers 8
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sweep_1m/validate.py --workers 8
```

결과: `logs/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sizes_1m/`.
`--dry-run`으로 조건을 확인하고 `--sizes`로 크기를 제한할 수 있습니다.
크기별로 두 noise를 생성해 프로세스에서 공유합니다. S=16에서는 약 164 GB RAM을 사용합니다.

## 알고리즘과 비교 기준

DF/SD/DD는 블록 첫 업데이트 직전에 현재 target-Q+현재 noise와 V로 filter를 만들고
그 첫 업데이트를 포함해 D회 사용합니다. 시차는 0..D−1이며 다른 timing 옵션은 없습니다.
SD는 self filter, DD는 peer filter, DF는 min receiver와 shared V를 사용합니다.
Q ← 0.9Q+0.1γV, V ← (1−0.1w)V+0.1wY, target-Q EMA=0.005입니다.
γ=0.999, σ=1, NN-like ρ=0.99이며 모든 보상은 0입니다.

Double Q는 기존 실험과 같은 두 head 동시 업데이트입니다.
각 head의 noisy online Q에서 행동을 선택하고 다른 head의 같은 시점 readout으로 평가한
backup을 사용합니다. Expectile V와 filter가 없으므로 coupling은 N/A입니다.
Double Q의 관측 operator는 head별 cross-evaluation이며 weighted expectile output과 구분합니다.

모든 방법의 주 bias는 mean_s max_a mean_i clean Q_i(s,a), 참값은 0입니다.
동일한 1M 실행의 100k 체크포인트도 보존합니다.
100k/1M의 크기별 주 그래프 총 8개는 `figures/`에,
추가 oracle/correlation 그림은 `diagnostics_figures/`에 저장합니다.
20 seed 평균 ± 표본 표준편차를 보고합니다.

## 난수와 진단

기존 실험과 noise/행동 난수열이 동일합니다. S=2는 원래 네 action field의 앞 두 값을 쓰고,
행동은 integers(4)%2를 사용합니다. 다른 크기는 각 크기에 맞는 action field를 사용합니다.
4-state의 재사용 배열은 이전 로그와 완전히 동일합니다.
Double Q 전체와 IQL D=1의 Q/V는 이전 결과와 비트 단위 일치 여부를 검사합니다.

진단은 업데이트 직전의 snapshot, 실제 weight와 backup을 기록합니다.
Filtering F=sum(wY)/sum(w)는 진단용이며 학습 SGD를 대체하지 않습니다.
누적 clean Q 오차와 주입 readout 오차를 분리하고 TD/drift/LOO proxy도 저장합니다.
마지막 100k는 unit당 sparse 시점 101개이므로 평균 |corr|에는 약 0.08의 표본 잡음이
있을 수 있습니다. Signed correlation과 covariance도 함께 해석해야 합니다.
크기에 비례해 항목당 방문 수가 줄어들므로 1M을 수렴값으로 가정하지 않습니다.

## 그림과 표 다시 만들기

```bash
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sweep_1m/refresh_figures.py
python utils/iclr2027_dd_iql/rebuttal/tabular_bias_refresh_first_sweep_1m/generate_latex_tables.py --compile-preview
```

표시 라벨은 Clipped Double Q, Double Q, DF-IQL, SD-IQL, DD-IQL입니다.
4-state 개별 그림 12개는 `figures_s4_individual/`에 저장합니다.
제목은 없으며 DF-IQL·SD-IQL·DD-IQL만 표시합니다.
Clipped Double Q는 max-Q bias에만 값이 있고 expectile filter bias/coupling은 N/A입니다.
