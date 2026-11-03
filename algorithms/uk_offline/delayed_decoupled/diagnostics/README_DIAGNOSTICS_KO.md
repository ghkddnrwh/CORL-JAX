# DDIQL / LDDIQL 진단 및 학습 재개 가이드

## 파일 배치

기존 프로젝트의 `algorithms/uk_offline/common/checkpointing.py`는 그대로 유지합니다.

- `ddiql_diagnostics_jax.py`: 기존 DD-IQL 스크립트의 수정본
- `lddiql_diagnostics_jax.py`: 기존 LDD-IQL 스크립트의 수정본
- `algorithms/uk_offline/common/diagnostics.py`: **두 스크립트가 공통으로 import하는 신규 모듈**. 프로젝트의 정확히 이 경로에 복사하세요.
- `compare_ddiql_diagnostics.py`: 로컬 로그만으로 세 실험을 비교하는 선택적 분석 스크립트
- `test_offline_rl_diagnostics.py`: 진단 계산·내구성·재개 처리 단위 테스트

각 알고리즘 스크립트는 원래 스크립트가 있던 프로젝트의 위치에 놓으세요. `algorithms` 디렉터리를 찾아 프로젝트 루트를 설정하는 기존 코드가 있으므로 이 파일들만 프로젝트 밖에서 실행하면 오류가 납니다.

## DDIQL의 인덱스 갱신 주기

새 설정은 다음과 같습니다.

```python
filter_index_update_period: Optional[int] = None
```

| 설정 | 의미 |
| --- | --- |
| `None` | 지연 Q/V 갱신 주기와 동일. 원래 DDIQL 동작을 그대로 보존 |
| `-1` | **처음 선택된 필터 인덱스를 학습 종료까지 고정**. 무한 주기의 명시적 표현 |
| 양의 정수 | 해당 스텝마다 필터 인덱스를 변경. 지연 Q/V 갱신과 독립 |
| `0` 및 `-2` 이하 | 오류 |

`None`과 `-1`은 서로 다릅니다. `None`은 기존처럼 주기적으로 바뀌고, `-1`은 절대 바뀌지 않습니다.

`cross` 모드에서 (N\ge3)이어야 순환과 고정의 차이가 의미 있게 나타납니다. `self` 모드는 주기와 관계없이 자신의 모델을 선택하고, (N=2) Cross도 상대 모델 하나밖에 없으므로 주기 변경 효과가 없습니다.

기존 `periodic_self` 설정은 이제 *Delayed 갱신 횟수*가 아니라 **Filter Index 갱신 횟수**를 기준으로 순환합니다. 기본 `None`일 때에는 기존과 완전히 같습니다.

## 세 실험 추천 설정

세 실험에 같은 `env`, `seed`, `ensemble_size`, `batch_size`, `iql_tau`, `beta`, `diagnostic_batch_size`, 데이터 정규화, 학습 스텝을 사용하세요. **각 실험의 `checkpoints_path` 루트는 반드시 다르게 설정**하세요. DDIQL 두 실험의 알고리즘 기본 이름이 같아 경로가 충돌할 수 있습니다.

아래는 실험 설정의 핵심 값입니다. 전체 하이퍼파라미터는 기존 설정을 유지하면 됩니다.

| 실험 | 알고리즘 | `ensemble_size` | `filter_index_update_period` | 예시 `checkpoints_path` 루트 |
| --- | --- | ---: | --- | --- |
| A | DDIQL, `cross` | 5 | `None` | `runs/ddiql_rotating` |
| B | DDIQL, `cross` | 5 | `-1` | `runs/ddiql_fixed` |
| C | LDDIQL | 5 | 적용되지 않음 | `runs/lddiql` |

예시: `--log_wandb False --checkpoints_path runs/ddiql_fixed --filter_index_update_period -1 --diagnostic_freq 5000 --diagnostic_batch_size 1024`. 사용하는 Pyrallis 버전에 맞는 CLI 형식 또는 YAML로 지정하세요.

LDDIQL은 필터 인덱스 변경 주기를 학습에 사용하지 않으므로 해당 인자를 추가하지 않았습니다.

## 로컬 파일 형식과 저장 주기

모든 저장은 **W&B 활성화 여부와 무관**합니다.

| 파일 | 내용 | 갱신 |
| --- | --- | --- |
| `metrics.jsonl` | 기존 train scalar, 추가 gradient norm, 진단 scalar+상관행렬, 평가, 재개 이벤트 | Train: `log_every`마다 / 진단: `diagnostic_freq`마다 / Eval: `eval_freq`마다 |
| `eval_logs.npz` | 기존 Evaluation 기록 전체 | 평가 직후, 시작·재개 직후, 정상 종료 시 원자적 저장 |
| `diagnostics_manifest.json` | 첫 실행의 설정과 고정 진단 배치의 결정적 식별자 | 최초 한 번 |
| `abandoned_segments/*.jsonl` | 체크포인트 재개로 다시 학습해야 하는 구간의 이전 로그 및 미완성 기록 | 재개 시 필요할 때 |
| `refit_metrics_*.jsonl` | Actor-only Refit 학습/평가 로그와 동결된 Critic 진단 | 별도 Refit 실행마다 새 파일 |

기본적으로 로그는 기존 `checkpoints_path`의 **최종 run 디렉터리**에 저장됩니다. `local_logs_dir`를 지정하면 해당 디렉터리에 저장합니다. 체크포인트 경로가 없으면 새로운 `local_logs/<run>/<seed>/<timestamp>_<id>` 디렉터리를 만듭니다.

JSONL은 한 줄이 하나의 이벤트입니다.

```json
{"timestep":5000,"event":"train","metrics":{"q_loss":0.1,"v_grad_norm":0.2}}
{"timestep":5000,"event":"diagnostic","metrics":{"scalars":{"q/corr_mean":0.85},"matrices":{"q/corr":[[1,0.85],[0.85,1]]}}}
```

예시는 형식을 보여주는 가상 숫자입니다. 상관계수 계산이 불가능한 상수 예측이나 앙상블 크기 1의 쌍별 통계는 JSON `null`로 기록됩니다.

`LocalMetricStore`는 **이벤트를 추가할 때마다 flush + fsync**합니다. 서버가 강제 종료되어도 이미 기록된 이벤트는 별도의 W&B 동기화 없이 로컬에 남습니다. `eval_logs.npz`는 임시 파일을 완성한 뒤 원자적으로 교체합니다.

### 재개 시 로그 일관성

기존 `TrainingCheckpointManager`의 모델 복원 방식을 유지합니다. 예를 들어 마지막 안전 체크포인트가 25,000스텝인데 종료 직전 28,000스텝까지 진단 로그를 썼다면, 복원 후에는 25,000부터 다시 학습해야 합니다.

이때 25,000 이후의 로그를 원래 `metrics.jsonl`에 그대로 두면 **서로 다른 두 학습 궤적이 한 파일에 섞이는 문제**가 생깁니다. 따라서 25,000 이후 로그는 `abandoned_segments/`에 보존하고, 주 로그는 안전 체크포인트 시점까지 맞춘 다음 재개합니다. 원래 기록은 삭제하지 않습니다.

가능하다면 `checkpoint_freq`를 5,000 정도로 줄여 재학습 구간을 줄이세요. 체크포인트 저장 비용과 디스크 용량은 증가합니다.

**제한:** 체크포인트 경로 자체가 없거나 자동 재개 대상이 아니라면 학습을 복원할 수 없습니다. 로컬 JSONL만으로 모델 상태를 재구성하는 기능은 제공하지 않습니다. 기존 Actor Refit은 원래 코드와 동일하게 새로운 Refit으로 시작하며, 이 변경은 Refit의 자동 재개까지 추가하지 않습니다.

## 핵심 진단 지표

별도 고정 진단 배치는 학습 전과 `diagnostic_freq`마다 같은 오프라인 전환 샘플을 사용합니다. 다른 알고리즘에서도 같은 `seed`/데이터셋/`diagnostic_batch_size`라면 같은 인덱스의 샘플입니다. 전처리까지 같아야 유효한 비교가 됩니다.

| 지표 (`metrics.scalars` 내부) | 해석 |
| --- | --- |
| `q/corr_mean`, `q/corr_i_j` | Q 예측값의 모델 간 Pearson 상관계수 |
| `q/ensemble_std_mean`, `q/pair_rmse_mean` | 모델들의 Q 예측 불일치 정도 |
| `td_residual/corr_mean` | 모델별 Bellman Residual 상관관계. 실제 Q 추정 오차의 직접 측정은 아님 |
| `v/corr_mean`, `value_residual/corr_mean` | V 예측과 (Q_i^{target}-V_i) 잔차의 상관관계 |
| `delayed_adv/corr_mean` | 지연된 (Q_i^d-V_i^d)의 상관관계 |
| `active_filter/corr_mean` | 학습에서 **실제로 사용하는 필터** (F_i)끼리의 상관관계 |
| `loo_filter/corr_mean` | 동일한 모델 상태에서 계산한 가상 LOO 필터 상관관계 |
| `active_filter/pair_sign_agreement_mean` | 모델 간 필터 부호(고/저 가중치) 일치율 |
| `filter/active_vs_loo_sign_disagree_fraction` | 현재 알고리즘의 필터와 LOO 필터가 서로 다른 판단을 하는 비율 |
| `filter/cross_vs_loo_sign_disagree_fraction` | 기본 순환 Cross와 LOO가 다른 판단을 하는 비율 |
| `filter/active_vs_value_residual_sign_disagree_fraction` | 지연 필터 부호와 현재 V 오차 부호의 불일치율 |
| `filter/loo_small_margin_fraction` | LOO 평균이 0 근처에 있는 비율. 기준은 전체 지연 Advantage 표준편차의 10% |
| `critic/delayed_q_target_mae`, `critic/delayed_v_online_mae` | 지연 네트워크가 현재 네트워크에서 얼마나 멀어졌는지 |
| `critic/td_residual_rmse` | Bellman Residual 크기 |
| `actor/exp_weight_clipped_fraction` | Actor AWBC 가중치가 최대 100으로 포화된 비율 |
| `q_grad_norm`, `v_grad_norm`, `actor_grad_norm` | 학습 스텝에서의 기울기 크기 (`event=train`) |

각 쌍별 상관행렬도 `metrics.matrices`에 저장되므로 단순 평균에 숨은 특정 모델 쌍의 문제를 확인할 수 있습니다.

### 중요한 통계적 주의

**Q 예측 상관계수가 높다는 사실만으로 Q 추정 오차의 상관관계가 높다고 결론 내리면 안 됩니다.** 상태에 따른 공통의 실제 가치 신호가 있기 때문입니다. `td_residual/corr_mean`, `value_residual/corr_mean`, 필터 부호 일치율, 최종 성능을 함께 보세요. TD Residual 역시 참된 Q 오차와 동일하지 않고, 공통 보상과 부트스트랩 목표에 의한 상관이 포함될 수 있습니다.

LOO 필터는 구조적으로 입력을 공유합니다. **각 모델 Advantage의 독립·동분산 오차**를 가정하면 서로 다른 두 LOO 평균의 오차 상관계수는 ((N-2)/(N-1))입니다. 이는 실제 학습 후 Q 상관계수에 대한 공식이 아닙니다.

성능 저하 원인에 대해 아래 패턴을 특히 확인하세요.

1. LDDIQL에서 `active_filter/corr_mean`과 부호 일치율은 올라가는데 `q/ensemble_std_mean`이 내려가는가? → 필터 판단의 동조화와 Critic 다양성 감소 가설.
2. `q/corr_mean`만 높고 `td_residual/corr_mean`은 높지 않은가? → 공통 가치 신호 때문에 원시 Q 상관이 높았을 가능성.
3. `filter/active_vs_value_residual_sign_disagree_fraction`이 커지고 V/TD 오차가 함께 증가하는가? → 지연된 필터의 판단 불일치 가설.
4. DD 고정 인덱스에서 특정 모델 쌍만 일관되게 편향되는가? → 한 파트너를 오래 참조하는 영향.
5. `actor/exp_weight_clipped_fraction`이 커지는가? → Value 차이가 Actor 가중치 포화로 전파되는지 확인.

이들은 원인 **가설**을 구분하는 실험 지표이며 인과관계의 확증은 아닙니다.

## 로컬 비교 그래프

NumPy/Matplotlib이 있다면 다음과 같이 세 경로를 넣어 비교할 수 있습니다.

```bash
python compare_ddiql_diagnostics.py \
  --ddiql runs/ddiql_rotating/DDIQL-JAX-halfcheetah-medium-expert-v2/0 \
  --ddiql_fixed runs/ddiql_fixed/DDIQL-JAX-halfcheetah-medium-expert-v2/0 \
  --lddiql runs/lddiql/LDDIQL-JAX-halfcheetah-medium-expert-v2/0 \
  --output compare_results
```

`compare_results/ddiql_comparison.png` 및 `ddiql_comparison_summary.csv`를 생성합니다. 비교 도구는 세 고정 프로브의 인덱스 해시가 다르면 비교를 중단합니다.

단위 테스트:

```bash
python -m unittest -v test_offline_rl_diagnostics.py
```

본 패키지의 순수 NumPy·파일 저장 기능은 테스트했지만, JAX/Flax/Optax/D4RL 등 학습 의존성이 없는 환경에서 작업했으므로 실제 GPU 학습 및 체크포인트 매니저와의 통합 실행은 별도 서버에서 확인해야 합니다.
