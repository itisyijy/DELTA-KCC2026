# DLST 실험 설계 및 결과 보고서

기준일은 2026년 4월 13일입니다. 본 문서는 DLST에서 지금까지 수행한 실험 설계, 핵심 결과, 프레임워크 변경 히스토리를 슬라이드 제작과 논문화 관점에서 바로 활용할 수 있도록 정리한 요약본입니다.

## 1. 연구 목표와 정리 기준

DLST는 DLinear 기반 장기 시계열 예측에 연합학습과 Test-Time Adaptation(TTA)을 결합해, 분포 이동이 발생하는 환경에서도 라벨 없이 적응 가능한 경량 프레임워크를 구축하는 것을 목표로 합니다. 현재까지의 실험은 크게 두 갈래로 진행됐습니다.

1. 기본 5개 베이스라인을 구축하고, 직접 가중치 업데이트 방식의 TTA와 피드백 루프가 실제로 유효한지 검증했습니다.
2. 초기 방식의 불안정성을 확인한 뒤, Murata를 중심으로 안정화 장치와 경량 adapter 기반 TTA로 설계를 전환했습니다.

본 문서의 표는 최종 파이프라인 기준 수치를 우선 사용했습니다. 다만 2026년 4월 11일 Murata parquet scaling 복원 이전의 초기 탐색 결과는 수치 체계가 달라 본 비교표에서는 본문 참고 수준으로만 취급했습니다.

평가 지표는 다음과 같습니다.

- MSE, MAE: global scale 공간 기준입니다.
- sMAPE: 원 단위 역정규화 기준입니다.
- 표의 값은 클라이언트 평균입니다.

## 2. 전체 프레임워크 구조와 동작 방식

### 2.1 전체 구조

DLST의 현재 구조는 `백본 학습 단계`, `테스트 시 적응 단계`, `선택적 피드백 단계`의 3단으로 정리할 수 있습니다.

1. 백본 학습 단계
중앙학습 또는 FedAvg로 DLinear 백본을 먼저 학습합니다. 입력 길이와 커널 크기는 데이터셋별로 다르게 설정했습니다.

2. 테스트 시 적응 단계
각 클라이언트가 테스트 구간에서 최근 관측 윈도우를 이용해 hindcast 손실을 계산하고, TTA를 수행합니다. 초기 버전은 DLinear의 trend/season 가중치를 직접 수정했고, 이후 버전은 백본을 동결한 뒤 adapter만 업데이트하도록 바꿨습니다.

3. 선택적 피드백 단계
FED-TTA Loop 계열에서는 로컬 적응으로 생긴 변화량을 서버에 올리고, clipping과 decay를 거쳐 글로벌 모델에 반영합니다. 이후 각 클라이언트는 갱신된 글로벌 가중치를 다시 받아 다음 스텝에 사용합니다.

### 2.2 현재 기준 동작 흐름

1. 입력 시계열을 global scaling 후 모델에 넣고, RevIN으로 윈도우 단위 분포 이동을 흡수합니다.
2. DLinear가 trend와 season 성분을 분리해 예측합니다.
3. TTA는 최근 관측값 일부를 다시 맞추는 hindcast 손실을 사용합니다.
4. 버전에 따라 적응 대상이 달라집니다. V1·V2는 DLinear의 trend/season 선형 가중치를 직접 수정했습니다. V3는 DLinear 가중치를 고정하고 RevIN의 affine 파라미터(γ_RevIN, β_RevIN)만 업데이트하는 `norm-only` 방식으로 전환했습니다. V4 이후는 backbone 전체를 동결하고 독립적인 affine/time-wise adapter(γ, δ)만 업데이트합니다.
5. rollback guard, hard gate, acceptance gate로 불안정한 업데이트를 건너뜁니다.
6. loop 모드에서는 수용된 변화량만 서버에 반영하고, anchor와 optimizer state를 함께 갱신합니다.

### 2.3 내부 손실 함수 설계와 변화

#### 2.3.1 Legacy direct-weight TTA 손실

초기 DLST는 DLinear backbone의 trend/season 가중치를 직접 수정하는 방식으로 TTA를 수행했습니다. 이 시기의 기본 목적함수는 다음과 같습니다.

\[
\mathcal{L}_{legacy}
= \mathcal{L}_{hind}
+ \alpha \mathcal{L}_{reg}
+ \lambda_{func}\mathcal{L}_{func}
\]

여기서 각 항의 의미는 다음과 같습니다.

1. Hindcast self-reconstruction loss

\[
\mathcal{L}_{hind}
= \frac{1}{k}\sum_{i=1}^{k}\left\|\hat{y}_{i}-x^{recent}_{i}\right\|^{2}
\]

현재 시점 직전의 최근 관측 구간을 라벨 대용 신호로 사용해, 예측의 앞부분 \(k\) step이 실제 최근 관측값을 복원하도록 강제했습니다. 이는 완전한 비지도 TTA 환경에서 사용할 수 있는 가장 직접적인 self-supervised objective 역할을 했습니다.

2. Dynamic component regularization

\[
\mathcal{L}_{reg}
= \lambda_{trend}\left\|W^{TTA}_{trend}-W^{anchor}_{trend}\right\|^{2}
+ \lambda_{season}\left\|W^{TTA}_{season}-W^{anchor}_{season}\right\|^{2}
\]

\[
\lambda_{trend}
= \lambda_{0}\exp\left(-\gamma\frac{|\mu_{curr}-\mu_{hist}|}{\sigma_{hist}}\right)
\]

\[
\lambda_{season}
= \lambda_{0}\exp\left(-\gamma\frac{|\sigma_{curr}-\sigma_{hist}|}{\sigma_{hist}}\right)
\]

이 항은 backbone drift를 억제하기 위한 weight-space regularization입니다. 평균 이동은 trend 쪽 anchor 보존 강도를, 분산 이동은 season 쪽 anchor 보존 강도를 조절하도록 분리했습니다. 즉, 도메인 시프트가 약할 때는 anchor 보존을 강하게 유지하고, 시프트가 클 때는 적응 여지를 더 주는 설계입니다.

단, \(\mathcal{L}_{reg}\)는 DLinear의 trend/season 가중치가 실제로 변경될 때만 의미를 갖습니다. V3에서 사용한 `update_scope=norm`은 DLinear 가중치를 고정하고 RevIN affine 파라미터만 업데이트하므로, \(W^{TTA}_{trend} = W^{anchor}_{trend}\)이고 \(W^{TTA}_{season} = W^{anchor}_{season}\)이 항상 성립합니다. 따라서 V3에서는 \(\mathcal{L}_{reg} = 0\)이고, `alpha`, `lambda0`, `gamma` 값은 Config에 명시돼 있더라도 실질 효과가 없습니다. V3의 유효 손실은 \(\mathcal{L}_{hind}\)만으로 구성됩니다.

3. Functional regularization

\[
\mathcal{L}_{func}
= \left\|\hat{y}_{TTA}-\hat{y}_{anchor}\right\|^{2}
\]

후속 단계에서는 출력 공간에서 anchor 예측과의 괴리를 직접 억제하는 functional regularization을 추가했습니다. 이는 weight-space 규제만으로는 예측 함수의 국소 변형을 충분히 제어하지 못한다는 판단에 따른 보완입니다.

또한 Murata 안정화 단계에서는 inactive 구간이 hindcast 손실을 과도하게 지배하지 않도록 near-zero masking과 inactive-window skip을 추가했습니다. 이 변화는 손실 함수 자체를 바꾸기보다는, 유효한 self-supervision이 존재하는 구간에서만 손실을 계산하도록 만드는 데이터 선택 규칙에 가깝습니다.

#### 2.3.2 Affine adapter hybrid 손실

직접 weight update가 고차원 파라미터 공간에서 불안정하다는 점이 확인된 뒤, 후속 버전에서는 backbone을 완전히 동결하고 저차원 adapter만 업데이트하도록 바꿨습니다. 이 시기의 예측은 다음과 같이 표현할 수 있습니다.

\[
\hat{Y}_{final} = \mathcal{A}_{\phi}(\hat{Y}_{backbone})
\]

여기서 \(\mathcal{A}_{\phi}\)는 channel-wise affine, time-wise affine, 또는 horizon-conv adapter입니다. 가장 주력으로 사용한 time-affine의 경우 \(\phi=(\gamma,\delta)\)이며,

\[
\hat{Y}_{final} = \gamma \odot \hat{Y}_{backbone} + \delta
\]

로 정의됩니다. 이때 \(\gamma, \delta\)만 학습되므로 적응 자유도가 매우 작고, backbone의 시계열 동역학 자체는 유지됩니다.

이 단계의 목적함수는 다음과 같습니다.

\[
\mathcal{L}_{affine}
= \alpha_{eff}\mathcal{L}_{cons}
+ \beta_{eff}\mathcal{L}_{hind}
+ \lambda_{anchor}\mathcal{L}_{anchor}
\]

1. Hindcast grounding loss

\[
\mathcal{L}_{hind}
= \frac{1}{k}\sum_{i=1}^{k}\left\|\hat{y}^{curr}_{i}-x^{recent}_{i}\right\|^{2}
\]

여전히 실제 최근 관측값을 가장 중요한 grounding signal로 사용했습니다.

2. Temporal consistency loss

\[
\mathcal{L}_{cons}
= \frac{1}{H-1}\sum_{i=1}^{H-1}\left\|\hat{y}^{curr}_{i}-\text{sg}\left(\hat{y}^{prev}_{i+1}\right)\right\|^{2}
\]

여기서 \(\text{sg}(\cdot)\)는 stop-gradient입니다. 이 항은 연속 창 사이에서 예측 함수가 급격히 바뀌는 현상을 줄이는 temporal smoothness prior로 해석할 수 있습니다.

3. Identity-anchor regularization

\[
\mathcal{L}_{anchor}
= \frac{\|\gamma-1\|^{2}+\|\delta\|^{2}}{C}
\]

adapter가 항상 항등변환 근방에 머물도록 제한해, 장기적으로 불필요한 누적 drift가 생기는 것을 막았습니다. 이는 direct-weight TTA에서의 anchor 보존을 저차원 파라미터 공간으로 옮긴 형태라고 볼 수 있습니다.

구현상 분모 \(C\)는 채널 수(마지막 차원)만을 사용합니다. `channel_affine`에서는 \(\gamma \in \mathbb{R}^{1\times1\times C}\)이므로 분자 합산 원소 수도 \(C\)여서 원소당 평균이 되지만, `time_affine`에서는 \(\gamma \in \mathbb{R}^{1\times H\times C}\)이므로 분자 합산 원소 수가 \(H\times C\)가 됩니다. 따라서 `time_affine` 모드에서 \(\mathcal{L}_{anchor}\)는 `channel_affine` 대비 \(H\)배(pred\_len=96이면 96배) 큰 스케일을 가집니다. 실험에서 `lambda_anchor=0.1`로 고정한 상태에서 두 adapter 모드 간 비교를 진행했으므로, 이 스케일 차이는 V4 scout 단계에서 channel-wise 대비 time-wise adapter의 `L_anchor` 기여분이 암묵적으로 더 강하게 작동했음을 의미합니다.

4. Bounded adaptive weighting

\[
boost = \min(1 + \rho \cdot \text{sg}(\mathcal{L}_{hind}),\; boost_{max})
\]

\[
\alpha_{eff} = \frac{\alpha}{boost},\qquad
\beta_{eff} = \beta \cdot boost
\]

즉, 현재 창의 hindcast 오차가 커질수록 일관성 항의 비중은 줄이고, 실제 관측값 복원 항의 비중은 키우도록 설계했습니다. 학술적으로는 self-supervised online adaptation에서 발생하는 `stability-plasticity trade-off`를 창 난이도에 따라 동적으로 조절하는 방식으로 해석할 수 있습니다.

#### 2.3.3 손실 함수 외부의 의사결정 규칙

후속 버전의 hard gate와 acceptance gate는 손실항 그 자체가 아니라, 손실값을 바탕으로 업데이트를 수행할지 말지를 결정하는 메타 규칙입니다.

1. Rollback guard
현재 창의 사전 hindcast 오차가 과거 rolling mean의 일정 배수를 넘으면 업데이트를 생략합니다.

2. Hard gate

\[
\mathcal{L}_{hind}^{pre} \ge \tau_{hard}\cdot \bar{\mathcal{L}}_{hind}
\quad \text{and} \quad n_{finite} \ge \texttt{hard\_gate\_min\_history}
\]

두 조건을 동시에 만족하는 충분히 어려운 창에서만 적응을 수행합니다. 첫 번째 조건은 쉬운 창에서의 불필요한 adaptation을 줄여 효율을 높이는 목적이며, 두 번째 조건은 warm-up 요건입니다. 이력이 `hard_gate_min_history`개 미만이면 rolling mean 추정이 불안정하므로 hard gate 판단 자체를 보류하고 해당 창의 적응을 건너뜁니다. V6 실험에서는 `hard_gate_min_history=20`을 사용했으므로, 초기 20개 창 동안은 hard gate 조건을 채우지 못해 적응이 항상 생략됩니다.

3. Acceptance gate

\[
\mathcal{L}_{hind}^{post} \le (1-m)\mathcal{L}_{hind}^{pre}
\]

를 만족할 때만 adapter 갱신을 수용합니다. 여기서 \(m\)은 acceptance margin입니다. 즉, 손실이 실제로 개선된 경우에만 업데이트를 채택하는 일종의 one-step line search 근사로 볼 수 있습니다.

4. Reset rule
업데이트 후 \(\mathcal{L}_{hind}\)가 `reset_threshold`를 넘으면 adapter를 항등변환으로 초기화합니다. 이는 이상치 창에 대한 bounded recovery 장치입니다.

#### 2.3.4 손실 함수 변화의 의미와 실험적 해석

손실 함수의 변천은 단순한 구현 변경이 아니라, `어디를 적응 대상으로 삼을 것인가`에 대한 가설 수정 과정으로 해석할 수 있습니다.

1. 1단계는 backbone weight 자체를 직접 수정하는 고자유도 적응이었습니다.
이 방식은 표현력은 충분했지만, parameter drift와 feedback instability에 매우 취약했습니다.

2. 2단계는 functional regularization으로 예측 함수의 변화폭을 직접 억제하려는 시도였습니다.
Murata의 legacy FED-TTA에서는 \(\lambda_{func}=0.5 \rightarrow 2.0\)로 갈수록 MSE가 `0.3171 -> 0.3141`로 완만하게 줄어, 출력 공간 규제가 부분적인 안정화 효과를 보였습니다. 반면 Electricity FED-TTA에서는 같은 sweep에서 MSE가 `0.2599 -> 0.2740`으로 악화되어, 이 보정이 데이터셋 전반에 일관되게 작동하지는 않았습니다.

3. 3단계는 적응 대상을 backbone에서 adapter로 축소하고, 손실도 weight anchoring 중심에서 `grounding + temporal consistency + identity anchoring` 중심으로 재구성한 단계였습니다.
학술적으로는 고차원 비선형 함수 자체를 업데이트하는 대신, 이미 학습된 예측 함수 위에 저차원 보정층을 얹는 residual correction 관점에 가깝습니다. 이 전환 이후 catastrophic failure가 크게 줄었고, Murata에서는 backbone에 거의 근접한 수준까지 회복했습니다.

요약하면, DLST의 손실 함수 변화는 `강한 적응`에서 `제한된 적응`, `weight-space 규제`에서 `function-space 및 low-dimensional correction`, `항상 적응`에서 `선택적으로 적응`으로 이동한 과정이라고 정리할 수 있습니다.

### 2.4 공통 실험 설정

- 장비는 모두 `cuda:1` 기준으로 실행했습니다.
- 실험은 모두 백그라운드 세션 기준으로 운영했습니다.
- 배치 크기는 256입니다.
- 중앙학습은 15 epoch, FedAvg는 15 global round 기준입니다.
- 데이터셋별 설정은 다음과 같습니다.

| 데이터셋 | 입력 길이 | 예측 길이 | 커널 크기 |
| --- | ---: | ---: | ---: |
| Murata | 192 | 96 | 49 |
| Solar | 288 | 96 | 73 |
| Electricity | 336 | 96 | 25 |

### 2.5 데이터셋 및 버전별 하이퍼파라미터

#### 2.5.1 데이터셋별 backbone 학습 하이퍼파라미터

| 데이터셋 | 샘플링 간격 | seq_len | pred_len | kernel_size | batch | 학습률 | 중앙학습 | 연합학습 | 공통 설정 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Murata | 15분 | 192 | 96 | 49 | 256 | 0.001 | 15 epoch, patience 7 | local epoch 3, global round 15 | `individual=false`, `revin_affine=true`, `seed=0` |
| Solar | 10분 | 288 | 96 | 73 | 256 | 0.001 | 15 epoch, patience 7 | local epoch 3, global round 15 | `individual=false`, `revin_affine=true`, `seed=0` |
| Electricity | 1시간 | 336 | 96 | 25 | 256 | 0.001 | 15 epoch, patience 7 | local epoch 3, global round 15 | `individual=false`, `revin_affine=true`, `seed=0` |

#### 2.5.2 V1~V2 기본 5개 베이스라인 하이퍼파라미터

V1과 V2에서 사용한 5개 베이스라인은 데이터셋별 backbone 설정만 다르고, TTA와 loop 파라미터 구조는 동일했습니다.

| 실험 타입 | 시작점 | 업데이트 대상 | k_ratio | alpha | lambda0 | gamma | lambda_func | TTA lr | grad clip | rollback | 추가 파라미터 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Centralized | 원시 데이터 | 없음 | - | - | - | - | - | 0.001 | - | - | epoch 15, patience 7 |
| FedAvg | 원시 데이터 | 없음 | - | - | - | - | - | 0.001 | - | - | local epoch 3, global round 15 |
| DLinear-TTA | 중앙학습 checkpoint | trend/season weight 직접 수정 | 0.25 | 1.0 | 1.0 | 1.0 | 0.0 | 0.001 | 1.0 | threshold 3.0, window 20 | backbone은 centralized |
| FED-TTA | FedAvg checkpoint | trend/season weight 직접 수정 | 0.25 | 1.0 | 1.0 | 1.0 | 0.0 | 0.001 | 1.0 | threshold 3.0, window 20 | backbone은 federated |
| FED-TTA Loop | FedAvg checkpoint | trend/season weight 직접 수정 + 서버 피드백 | 0.25 | 1.0 | 1.0 | 1.0 | 0.0 | 0.001 | 1.0 | threshold 3.0, window 20 | `delta_clip_norm=1.0`, `decay_factor=0.9` |

Legacy direct-weight 단계에서의 functional regularization 탐색은 주로 `lambda_func ∈ {0.5, 1.0, 2.0}`로 수행했고, Murata 일부 탐색에서는 `5.0`, `10.0`까지 확장했습니다.

#### 2.5.3 V3 Murata direct-weight 안정화 하이퍼파라미터

이 단계는 Murata 전용 실험이었습니다. 공통 설정은 `lr=1e-4`, `update_scope=norm`, `k_ratio=0.25`, `alpha=1.0`, `lambda0=1.0`, `gamma=1.0`, `lambda_func=0.0`, `grad_clip=1.0`, `rollback_threshold=3.0`, `rollback_window=20`, `drift_gate_threshold=0.0`, `ema_beta=1.0`입니다. `update_scope=norm`이므로 DLinear 가중치는 고정되어 `alpha`, `lambda0`, `gamma`는 실질 효과 없고, 유효 손실은 `L_hind`만입니다.

| 실험 버전 | hindcast_mask_threshold | min_active_frac | 목적 |
| --- | ---: | ---: | --- |
| baseline_norm | 0.0 | 0.0 | norm-only direct update 기준선입니다. |
| m1_mask_t0p1 | 0.1 | 0.0 | inactive 구간을 hindcast 계산에서 제외합니다. |
| m3_skip_0p3 | 0.0 | 0.3 | inactive window 자체를 skip합니다. |
| comb_m1m3_t0p1_f0p3 | 0.1 | 0.3 | masking과 skip을 동시에 적용합니다. |

#### 2.5.4 V4~V6 Murata affine adapter 계열 하이퍼파라미터

Affine 계열로 전환한 뒤에는 backbone을 동결하고 adapter만 업데이트했습니다. 공통 설정은 `lr=1e-4`, `alpha=0.3`, `beta=1.0`, `lambda_anchor=0.1`, `lambda0=1.0`, `gamma=1.0`, `lambda_func=0.0`, `sensitivity=1.0`, `max_boost=5.0`, `grad_clip=1.0`, `rollback_threshold=3.0`, `rollback_window=20`, `drift_gate_threshold=0.0`입니다.

| 버전 | 실험 이름 | adapter_mode | k_ratio | reset_threshold | hard_gate_scale | acceptance_margin | 비고 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| V4 | channel_base | channel_affine | 0.25 | inf | - | - | channel-wise 기준선입니다. |
| V4 | channel_short_k | channel_affine | 0.125 | inf | - | - | short-k만 반영했습니다. |
| V4 | time_base | time_affine | 0.25 | inf | - | - | time-wise 기준선입니다. |
| V4 | time_short_k | time_affine | 0.125 | inf | - | - | time-wise + short-k입니다. |
| V4 | time_short_k_reset_guard | time_affine | 0.125 | 2.5 | - | - | reset guard를 처음 반영했습니다. |
| V5 | time_k00625_reset2p5 | time_affine | 0.0625 | 2.5 | - | - | Murata 주력 설계의 최종 control입니다. |
| V5 | time_k0125_reset1p5 | time_affine | 0.125 | 1.5 | - | - | reset을 더 공격적으로 적용했습니다. |
| V5 | time_k0125_reset2p5 | time_affine | 0.125 | 2.5 | - | - | short-k보다 긴 hindcast를 비교했습니다. |
| V5 | time_k0125_reset4p0 | time_affine | 0.125 | 4.0 | - | - | reset을 느슨하게 적용했습니다. |
| V6 | time_best_control | time_affine | 0.0625 | 2.5 | 0.0 | - | hard gate 비교용 기준선입니다. |
| V6 | time_gate1p00 | time_affine | 0.0625 | 2.5 | 1.00 | - | `hard_gate_min_history=20`입니다. |
| V6 | time_gate1p02 | time_affine | 0.0625 | 2.5 | 1.02 | - | `hard_gate_min_history=20`입니다. |
| V6 | time_gate1p05 | time_affine | 0.0625 | 2.5 | 1.05 | - | `hard_gate_min_history=20`입니다. |
| V6 | time_gate1p10 | time_affine | 0.0625 | 2.5 | 1.10 | - | `hard_gate_min_history=20`입니다. |
| V6 | conv_base | horizon_conv | 0.0625 | 2.5 | 0.0 | - | adapter 구조만 horizon-conv로 교체했습니다. |
| V6 | conv_hgate1p10 | horizon_conv | 0.0625 | 2.5 | 1.10 | - | horizon-conv + hard gate입니다. |
| V6 | time_control | time_affine | 0.0625 | 2.5 | 0.0 | -1.0 | selective activation 비교용 기준선입니다. |
| V6 | accept_nonworse | time_affine | 0.0625 | 2.5 | 0.0 | 0.0 | 손실이 나빠지지 않을 때만 수용합니다. |
| V6 | accept_0p25pct | time_affine | 0.0625 | 2.5 | 0.0 | 0.0025 | 0.25% 이상 개선 시만 수용합니다. |
| V6 | accept_0p50pct | time_affine | 0.0625 | 2.5 | 0.0 | 0.005 | 0.50% 이상 개선 시만 수용합니다. |

#### 2.5.5 Electricity, Solar 전이 점검용 affine 하이퍼파라미터

Murata에서 고른 time-affine 설계를 Electricity와 Solar에 그대로 이식해 비교했습니다. 공통 설정은 `adapter_mode=time_affine`, `alpha=0.3`, `beta=1.0`, `lambda_anchor=0.1`, `lambda0=1.0`, `gamma=1.0`, `lambda_func=0.0`, `lr=1e-4`, `grad_clip=1.0`, `rollback_threshold=3.0`, `rollback_window=20`, `reset_threshold=2.5`입니다.

| 데이터셋 | 실험 버전 | k_ratio | 비고 |
| --- | --- | ---: | --- |
| Electricity | time_k00625_reset2p5 | 0.0625 | Fed backbone 위에서 time-affine를 적용했습니다. |
| Electricity | time_k0125_reset2p5 | 0.125 | 더 긴 hindcast를 비교했습니다. |
| Solar | time_k00625_reset2p5 | 0.0625 | Fed backbone 위에서 time-affine를 적용했습니다. |
| Solar | time_k0125_reset2p5 | 0.125 | 더 긴 hindcast를 비교했습니다. |

## 3. 변경 히스토리

| 날짜 | 버전 | 핵심 변경 | 변경 이유 |
| --- | --- | --- | --- |
| 2026-04-10 | V1 초기 프레임워크 | DLinear, RevIN, Centralized, FedAvg, DLinear-TTA, FED-TTA, FED-TTA Loop의 5개 베이스라인 구축 | 기본 비교군과 실험 러너를 한 번에 확보하기 위함입니다. |
| 2026-04-10 | V1.1 데이터 파이프라인 정리 | 데이터셋별 kernel/seq 길이 조정, parquet manifest 지원, checkpoint metadata, 자동 prerequisite 실행 추가 | 데이터셋 간 설정 불일치와 재현성 문제를 줄이기 위함입니다. |
| 2026-04-11 | V2 루프 안정화 | FED-TTA Loop의 anchor 갱신 및 optimizer state 초기화 추가 | 피드백 루프가 누적 drift를 재증폭하며 발산하는 문제를 막기 위함입니다. |
| 2026-04-11 | V3 Murata 안정화 | Murata parquet scaling 복원, update scope를 `norm` 중심으로 축소, inactive step masking 및 skip 제어 추가 | Murata에서 직접 weight TTA가 쉽게 흔들리는 문제를 줄이기 위함입니다. |
| 2026-04-12 | V4 Affine adapter 전환 | backbone 동결, affine adapter 도입, hybrid loss와 bounded adaptive weighting 추가 | backbone 직접 수정 방식의 불안정성을 줄이고 더 경량화된 적응을 만들기 위함입니다. |
| 2026-04-12 | V5 Time-affine 정교화 | channel-wise보다 time-wise adapter 중심으로 전환, short-k, reset guard 튜닝 | Murata에서 더 안정적인 적응 축을 찾기 위함입니다. |
| 2026-04-12~13 | V6 효율화 장치 | hard gate, horizon-conv adapter, selective activation gate 추가 | 동일 정확도를 유지하면서 실제 adaptation 횟수를 줄일 수 있는지 검증하기 위함입니다. |

## 4. 버전별 실험 설계와 결과

### 4.1 V1. 초기 5개 베이스라인 구축

이 단계에서는 3개 데이터셋에 대해 `Centralized`, `FedAvg`, `Centralized+TTA`, `FedAvg+TTA`, `FED-TTA Loop`를 모두 실행했습니다. 초기 질문은 단순했습니다. `직접 weight TTA와 loop feedback이 backbone 대비 실제 이득을 주는가`였습니다.

#### Murata

| 방법 | MSE | MAE | sMAPE |
| --- | ---: | ---: | ---: |
| Centralized | 0.3064 | 0.3151 | 131.02 |
| FedAvg | 0.3032 | 0.3075 | 130.73 |
| DLinear-TTA | 0.3223 | 0.3378 | 131.10 |
| FED-TTA | 0.3188 | 0.3315 | 130.84 |
| FED-TTA Loop | 1822091.9375 | 324.2297 | 147.61 |

#### Electricity

| 방법 | MSE | MAE | sMAPE |
| --- | ---: | ---: | ---: |
| Centralized | 0.1464 | 0.2434 | 12.57 |
| FedAvg | 0.1533 | 0.2467 | 12.56 |
| DLinear-TTA | 0.2460 | 0.3064 | 15.21 |
| FED-TTA | 0.2536 | 0.3116 | 15.33 |
| FED-TTA Loop | 4023617.0882 | 667.0454 | 59.01 |

#### Solar

| 방법 | MSE | MAE | sMAPE |
| --- | ---: | ---: | ---: |
| Centralized | 0.2232 | 0.2561 | 146.01 |
| FedAvg | 0.2242 | 0.2567 | 146.00 |
| DLinear-TTA | 0.2254 | 0.2588 | 146.14 |
| FED-TTA | 0.2268 | 0.2597 | 146.12 |
| FED-TTA Loop | 4084808.5347 | 664.1307 | 160.39 |

#### 발견한 한계

- FedAvg backbone 자체가 이미 강했고, direct weight TTA는 세 데이터셋 모두에서 일관되게 성능을 떨어뜨렸습니다.
- FED-TTA Loop는 심각하게 발산했습니다.
- 즉, 초기 설계는 `적응`보다 `드리프트 누적`이 더 크게 작동했습니다.

#### 이후 변경

- FED-TTA Loop의 anchor 갱신 방식과 optimizer state 관리 로직을 수정했습니다.
- Murata에서는 update scope를 줄이고, inactive 구간을 다루는 안정화 장치를 따로 실험하기 시작했습니다.
- backbone 직접 수정 대신 작은 adapter만 조정하는 방향으로 설계를 전환했습니다.

### 4.2 V2. FED-TTA Loop 발산 수정 검증

발산 원인은 서버 피드백 후 client anchor가 갱신되지 않아 이미 반영된 drift가 다시 누적되는 구조적 버그였습니다. 이를 수정한 뒤 세 데이터셋에서 loop를 다시 측정했습니다.

| 데이터셋 | 수정 후 FED-TTA Loop MSE | 수정 후 MAE | 수정 후 sMAPE |
| --- | ---: | ---: | ---: |
| Murata | 0.4068 | 0.3872 | 133.70 |
| Electricity | 0.1691 | 0.2731 | 14.11 |
| Solar | 0.3268 | 0.3424 | 148.74 |

#### 발견한 한계

- 수치 폭주는 사라졌지만, 성능은 여전히 backbone보다 크게 나빴습니다.
- 즉, loop 버그 수정은 필요조건이었지만 충분조건은 아니었습니다.

#### 이후 변경

- loop 자체를 바로 논문화 핵심으로 삼기보다, 먼저 `로컬 적응을 얼마나 안정화할 수 있는가`로 연구 초점을 좁혔습니다.
- Murata에서 직접 weight 업데이트를 더 세밀하게 제한하는 실험을 진행했습니다.

### 4.3 V3. Murata 안정화 제어 실험

이 단계는 Murata를 대상으로 `직접 weight TTA를 완전히 버릴지, 아니면 제한적으로 살릴 수 있을지`를 확인하는 단계였습니다. 공통 backbone은 FedAvg이며, 학습률을 낮추고 update scope를 `norm`으로 제한했습니다. `update_scope=norm`은 DLinear의 trend/season 선형 가중치를 건드리지 않고 RevIN의 learnable affine 파라미터(γ_RevIN, β_RevIN)만 업데이트합니다. 따라서 이 단계의 실질 적응 대상은 윈도우 단위 분포 이동을 흡수하는 정규화 스케일뿐이며, DLinear 내부 동역학은 완전히 고정된 상태입니다.

세 가지 안정화 메커니즘을 독립·조합하여 실험했습니다.

- **Mechanism 1 (m1)**: hindcast 계산 시 near-zero(야간 휴지 구간) 스텝을 masking해 손실을 왜곡하는 비활성 스텝을 제외합니다 (`hindcast_mask_threshold`).
- **Mechanism 2 (EMA anchor)**: EMA로 anchor를 천천히 이동시켜 장기 드리프트에 적응합니다 (`ema_beta`). 탐색 결과 Murata에서 유의미한 개선이 없어 이후 실험에서는 `ema_beta=1.0`(비활성)으로 고정했습니다.
- **Mechanism 3 (m3)**: 비활성 구간 비율이 기준 이상이면 창 전체의 TTA를 skip합니다 (`min_active_frac`).

| 방법 | 설계 포인트 | MSE | MAE | sMAPE |
| --- | --- | ---: | ---: | ---: |
| Fed backbone | 적응 없음 | 0.3032 | 0.3075 | 130.73 |
| baseline_norm | norm-only update (m1·m2·m3 모두 비활성) | 0.3038 | 0.3078 | 130.67 |
| m1_mask_t0p1 | m1: inactive hindcast masking (threshold=0.1) | 0.3040 | 0.3080 | 130.66 |
| m3_skip_0p3 | m3: inactive window skip (min_frac=0.3) | 0.3039 | 0.3079 | 130.67 |
| comb_m1m3_t0p1_f0p3 | m1+m3 결합 | 0.3042 | 0.3084 | 130.66 |

#### 발견한 한계

- catastrophic failure는 줄었지만, backbone을 의미 있게 넘지 못했습니다.
- inactive 구간 제어는 안정성 관리에는 도움을 줬지만 정확도 개선은 거의 없었습니다.
- EMA anchor(m2)는 Murata에서 유의미한 개선 효과가 없어 이후 단계에서는 사용하지 않았습니다.
- direct weight update를 계속 밀어붙이기에는 효율 대비 이득이 작았습니다.

#### 이후 변경

- backbone을 완전히 동결하고, 작은 affine adapter만 업데이트하는 구조로 방향을 전환했습니다.
- 적응 신호도 단순 hindcast 하나가 아니라 `hindcast + temporal consistency + anchor penalty`의 hybrid loss로 재설계했습니다.

### 4.4 V4. Affine adapter scout

이 단계에서는 adapter 축을 `channel-wise`와 `time-wise`로 나눠 비교했고, 짧은 hindcast와 reset guard가 도움이 되는지 확인했습니다. Murata에서 총 6개 핵심 variant를 비교했습니다.

| 방법 | MSE | MAE | sMAPE | Adapt Rate |
| --- | ---: | ---: | ---: | ---: |
| Fed backbone | 0.3032 | 0.3075 | 130.73 | - |
| channel_base | 0.3105 | 0.3102 | 130.86 | 0.821 |
| channel_short_k | 0.3101 | 0.3099 | 130.88 | 0.798 |
| time_base | 0.3098 | 0.3097 | 130.86 | 0.822 |
| time_short_k | 0.3096 | 0.3096 | 130.90 | 0.793 |
| time_short_k_reset_guard | 0.3095 | 0.3096 | 130.90 | 0.790 |

#### 발견한 한계

- affine adapter로 바꾸자 catastrophic failure는 사라졌지만, 절대 성능은 아직 backbone보다 약 2.10% 높은 MSE에 머물렀습니다.
- 다만 같은 affine 계열 안에서는 time-wise가 channel-wise보다 일관되게 나았고, short-k와 reset guard가 개선 방향이라는 점은 분명했습니다.

#### 이후 변경

- Murata 주력 설계를 `time_affine + short k + reset guard`로 고정하고, k와 reset threshold를 본격적으로 sweep했습니다.

### 4.5 V5. Time-affine 주력 설계 튜닝

이 단계는 Murata에서 가장 중요한 전환점이었습니다. short-k와 reset guard를 다시 조정하자 affine 계열 성능이 backbone에 거의 근접했습니다.

| 방법 | 설계 포인트 | MSE | MAE | sMAPE | Adapt Rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Fed backbone | 적응 없음 | 0.3032 | 0.3075 | 130.73 | - |
| time_k00625_reset2p5 | k=0.0625, reset=2.5 | 0.3032 | 0.3075 | 130.72 | 0.778 |
| time_k0125_reset1p5 | k=0.125, reset=1.5 | 0.3033 | 0.3076 | 130.70 | 0.788 |
| time_k0125_reset2p5 | k=0.125, reset=2.5 | 0.3033 | 0.3076 | 130.70 | 0.789 |
| time_k0125_reset4p0 | k=0.125, reset=4.0 | 0.3034 | 0.3076 | 130.70 | 0.791 |

#### 발견한 한계

- 가장 좋은 조합도 Fed backbone 대비 MSE 차이가 거의 없는 수준이었고, 뚜렷한 초과 성능은 아직 확보하지 못했습니다.
- 즉, adapter 전환은 `폭망 방지`에는 성공했지만, `확실한 정확도 우위`까지는 가지 못했습니다.

#### 이후 변경

- 정확도보다 먼저 효율을 보자는 방향으로 실험 목적을 재정의했습니다.
- 동일 성능을 유지하면서 adaptation 횟수를 줄일 수 있는지 hard gate와 selective activation을 실험했습니다.

### 4.6 V6. 효율화 장치와 구조 변형

이 단계에서는 Murata 주력 설계(time-affine control)를 기준으로 `hard gate`, `selective activation`, `horizon-conv`를 비교했습니다.

| 방법 | 핵심 아이디어 | MSE | MAE | sMAPE | Adapt Rate |
| --- | --- | ---: | ---: | ---: | ---: |
| time_control | time-affine 기준선 | 0.3032 | 0.3075 | 130.72 | 0.778 |
| time_gate1p10 | hard gate scale=1.10 | 0.3032 | 0.3075 | 130.71 | 0.177 |
| accept_0p50pct | 0.50% 개선 시만 수용 | 0.3032 | 0.3075 | 130.72 | 0.005 |
| conv_hgate1p10 | horizon-conv + hard gate | 0.3036 | 0.3077 | 130.72 | 0.176 |

#### 발견한 한계

- horizon-conv는 오히려 성능을 떨어뜨렸습니다.
- selective activation은 adaptation을 거의 끊어도 성능이 유지돼, 현재 Murata에서는 많은 업데이트가 실제로는 불필요하다는 신호를 줬습니다.
- hard gate와 acceptance gate는 정확도 개선보다 `연산 효율 확보` 측면의 기여가 더 컸습니다.

#### 정리

- `time_gate1p10`은 기준선 대비 성능을 사실상 유지하면서 adaptation rate를 약 77.2% 줄였습니다.
- `accept_0p50pct`는 adaptation rate를 약 99.4% 줄였지만, 정확도 우위는 뚜렷하지 않았습니다.
- 따라서 현재까지는 `정확도 최적화 버전`보다 `효율 보존형 버전`으로 해석하는 편이 더 적절합니다.

### 4.7 Electricity, Solar 전이 점검

Murata에서 고른 time-affine 계열이 다른 데이터셋에도 통하는지 별도로 점검했습니다.

| 데이터셋 | 방법 | MSE | MAE | sMAPE |
| --- | --- | ---: | ---: | ---: |
| Electricity | backbone | 0.1533 | 0.2467 | 12.56 |
| Electricity | time-affine best | 0.1530 | 0.2464 | 12.55 |
| Solar | backbone | 0.2242 | 0.2567 | 146.00 |
| Solar | time-affine best | 0.2242 | 0.2566 | 146.02 |

#### 해석

- Electricity에서는 약 0.20% 수준의 미세한 MSE 개선이 있었습니다.
- Solar에서는 사실상 개선이 없었습니다.
- 즉, Murata에서 찾은 설계가 범용적으로 강하다고 말하기에는 근거가 아직 부족합니다.

## 5. 현재까지의 결론

### 5.1 확실하게 말할 수 있는 점

1. direct weight TTA와 초기 FED-TTA Loop는 안정성이 부족했고, 그대로는 논문화하기 어렵습니다.
2. backbone 동결 + time-affine adapter 전환은 catastrophic failure를 크게 줄였고, Murata에서는 backbone에 매우 근접한 수준까지 회복했습니다.
3. hard gate와 selective activation은 정확도를 유지하면서 adaptation 빈도를 크게 낮출 수 있다는 점에서 실용적 가치가 있습니다.
4. 3개 데이터셋 공통 drift-gate sweep에서 `gate_1p0`가 no-harm 조건을 만족하며 공통 threshold로 선택되었습니다.

### 5.2 아직 부족한 점

1. 강한 backbone 대비 일관된 정확도 우위가 없습니다.
2. Solar와 Electricity에서 효과가 작아 데이터셋 일반성이 부족합니다.
3. 따라서 현재 스토리는 `정확도 향상형 기법`보다 `안정성/효율 중심의 경량 온라인 적응`에 가깝습니다.

### 5.3 슬라이드용 핵심 메시지

- 초기 질문: FL + TTA + feedback loop를 바로 결합하면 좋아지는가.
- 현재 답: 바로 좋아지지 않았고, direct weight update는 불안정했습니다.
- 핵심 전환: backbone 직접 수정에서 time-affine adapter로 전환했습니다.
- 현재 최종 메시지: Murata에서는 backbone 성능을 거의 보존하면서 adaptation 빈도를 크게 줄이는 구조까지 확보했습니다. 다만 범용 정확도 향상 주장까지는 추가 근거가 더 필요합니다.
- KCC 결론 메시지: 공통 drift-gate(`gate_1p0`)로 no-harm를 유지하면서 Murata/Solar에서 적응 빈도를 유의미하게 줄였습니다.

## 5.4 KCC Drift-Gate Sweep (20260413)

공통 실험 목적은 **평균 성능 개선이 아니라 no-harm + adaptation 비용 절감**을 검증하는 것입니다.

선택 규칙:
- 공통 threshold가 모든 데이터셋에서 상대 MSE 열화 `<= 0.5%`를 만족해야 합니다.
- 그중 평균 adapt rate가 가장 낮은 threshold를 선택합니다.

### 5.4.1 Threshold Sweep 요약

| threshold | murata deg% | electricity deg% | solar deg% | avg adapt |
| --- | ---: | ---: | ---: | ---: |
| 0.3 | 0.021 | -0.097 | 0.003 | 0.846 |
| 0.5 | 0.021 | -0.097 | 0.003 | 0.846 |
| 1.0 | 0.015 | -0.096 | 0.000 | 0.605 |

선택 결과: `gate_1p0`.

### 5.4.2 Main Table (backbone vs control vs gate_1p0)

| dataset | method | MSE | MAE | sMAPE |
| --- | --- | ---: | ---: | ---: |
| murata | backbone | 0.3032 | 0.3075 | 130.73 |
| murata | control | 0.3032 | 0.3075 | 130.72 |
| murata | gate_1p0 | 0.3032 | 0.3076 | 130.72 |
| electricity | backbone | 0.1533 | 0.2467 | 12.56 |
| electricity | control | 0.1532 | 0.2465 | 12.56 |
| electricity | gate_1p0 | 0.1532 | 0.2465 | 12.56 |
| solar | backbone | 0.2242 | 0.2567 | 146.00 |
| solar | control | 0.2242 | 0.2566 | 146.02 |
| solar | gate_1p0 | 0.2242 | 0.2567 | 146.02 |

### 5.4.3 Efficiency Table (adaptation + skips)

| dataset | control adapt | gated adapt | drift skip | rollback skip | param ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| murata | 0.778 | 0.410 | 0.555 | 0.034 | 0.518% |
| electricity | 0.913 | 0.894 | 0.021 | 0.085 | 0.297% |
| solar | 0.847 | 0.510 | 0.414 | 0.076 | 0.346% |

### 5.4.4 Bootstrap CI (per-client, 2k resamples)

| dataset | metric | mean diff | CI low | CI high | n_clients |
| --- | --- | ---: | ---: | ---: | ---: |
| murata | mse | 0.000047 | 0.000028 | 0.000067 | 30 |
| murata | mae | 0.000033 | 0.000018 | 0.000054 | 30 |
| murata | smape | -0.016228 | -0.020545 | -0.012728 | 30 |
| murata | adapt_reduction | 0.472905 | 0.461344 | 0.481631 | 30 |
| electricity | mse | -0.000148 | -0.000183 | -0.000113 | 321 |
| electricity | mae | -0.000137 | -0.000158 | -0.000116 | 321 |
| electricity | smape | -0.003277 | -0.004709 | -0.001898 | 321 |
| electricity | adapt_reduction | 0.021220 | 0.008132 | 0.036421 | 321 |
| solar | mse | 0.000000 | -0.000014 | 0.000014 | 137 |
| solar | mae | 0.000003 | -0.000005 | 0.000010 | 137 |
| solar | smape | 0.022414 | 0.021571 | 0.023249 | 137 |
| solar | adapt_reduction | 0.397797 | 0.396956 | 0.398688 | 137 |

관련 산출물:
- `/home/jylee/DLinear-Season-Trend/docs/kcc_drift_gate_summary.md`

## 6. 비교에서 제외한 탐색

2026년 4월 11일 새벽에 수행한 Murata scope sweep, norm resweep, 일부 lambda/drift 탐색은 Murata scaling 복원 이전 결과가 섞여 있어 본문 주 비교표에서는 제외했습니다. 다만 이 초기 탐색은 `norm-only가 상대적으로 안전하다`, `짧은 hindcast가 유리할 수 있다`, `직접 weight update보다 더 작은 적응 단위가 필요하다`는 방향성을 제공했고, 이후 affine adapter 설계 전환의 근거가 됐습니다.
