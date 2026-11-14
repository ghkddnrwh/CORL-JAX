# IQL / DD-IQL value explosion audit

논문의 9개 환경에서 보관된 IQL 144 run과 DD-IQL 144 run을 비교하고,
V 진단이 있는 작은 rho 제어 16 run을 추가 확인합니다.
원본 코드/로그와 이전 Figure 1/Table 2 결과는 수정하지 않습니다.

```bash
python utils/iclr2027_dd_iql/rebuttal/value_explosion_audit/scan_logs.py
```

출력은 `logs/iclr2027_dd_iql/rebuttal/value_explosion_audit/`에 저장합니다.

- `README.md`: 판정 기준, 환경/설정별 관측 seed 수와 누락 범위.
- `value_trajectories_overview.pdf`: 네 tau/gamma 설정의 9개 환경 궤적.
- `figures/`: 각 설정의 PNG/PDF.
- `condition_summary.csv`, `run_summary.csv`: 조건별/seed별 수치와 원본 경로.
- `v_trajectories.csv.gz`: 사용한 V 시계열.

기본 비교는 고정 batch의 `diag/v_mean`, 1M 이내이며 rho=.005입니다.
초기 5k–100k median(|V|)의 100배와 절댓값 1000을 모두 초과한 상태가
연속 관측 3개 이상, 10k step 이상 유지되면 큰 증가로 집계합니다.
수학적 발산이나 oracle value bias의 증명을 뜻하지 않습니다.
10배/1000배 기준의 민감도도 함께 저장합니다.

로그가 1M보다 짧은 run과 1M 전체에서 기준을 넘지 않은 run을 구분합니다.
평균 V만으로 일부 state/head의 상쇄된 폭발을 배제할 수는 없습니다.
