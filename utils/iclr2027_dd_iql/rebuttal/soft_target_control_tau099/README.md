# τ=0.99 slow-target IQL control

기존 로그에서 IQL(ρ=.005), DD-IQL(ρ=.005, N=20, D=100),
IQL(ρ=.0005)을 비교합니다. 원본 코드/로그를 수정하거나 W&B에 접속하지 않습니다.

저장된 제어 로그가 있는 AntSoccer Arena와 Cube Single을 각각 Original γ와
γ=.999에서 비교합니다. 각 조건은 seed 0–3입니다.

```bash
python utils/iclr2027_dd_iql/rebuttal/soft_target_control_tau099/plot_control.py
```

결과 경로:
`logs/iclr2027_dd_iql/rebuttal/soft_target_control_tau099/`

- `figures/`: V, mean weight, 두 correlation의 궤적과 Table 2 대응 막대 그림.
  각각 PNG/PDF. 진단은 1M까지입니다.
- `tables/`: 원 Table 2 집계 방식(5k–100k, 최대 20 checkpoints/seed)의 LaTeX와
  더 긴 공통 구간의 별도 집계, Figure 삽입용 LaTeX.
- `data/`: 원본에서 추출한 진단, per-seed 집계, mean/SD 궤적,
  제어 실험의 1M/2M 평가 성공률.
- `source_manifest.json`: 사용한 config/history/NPZ 경로와 SHA256.
- 출력 `README.md`: 그림 링크, 수치, 지표 정의와 해석의 범위.

4쪽 미리보기 PDF는 출력 디렉터리에서 다음 명령으로 만들 수 있습니다.

```bash
pdflatex -interaction=nonstopmode -halt-on-error comparison_preview.tex
```

Python 의존성: numpy, pandas, matplotlib, PyYAML. orjson은 선택 사항입니다.
기존 Fig. 1의 Times New Roman/STIX 및 symlog 스타일을 따릅니다.

V는 모든 방법에서 고정 batch의 `diag/v_mean`을 사용합니다.
원 Figure 1의 학습 minibatch `v_mean`과 batch가 다릅니다.
AntSoccer DD-IQL의 보관된 네 seed는 800k–900k까지만 진단이 있으므로
네 seed의 곡선은 공통 끝점인 800k에서 끝납니다. 외삽하지 않습니다.
2M 성공률은 작은 ρ 제어의 실제 1M/2M 평가 checkpoint 값입니다.
