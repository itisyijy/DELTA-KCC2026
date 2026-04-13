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
4. 초기 버전은 backbone weight를 직접 수정했고, 후속 버전은 affine/time-wise adapter만 수정합니다.
5. rollback guard, hard gate, acceptance gate로 불안정한 업데이트를 건너뜁니다.
6. loop 모드에서는 수용된 변화량만 서버에 반영하고, anchor와 optimizer state를 함께 갱신합니다.

### 2.3 공통 실험 설정

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

이 단계는 Murata를 대상으로 `직접 weight TTA를 완전히 버릴지, 아니면 제한적으로 살릴 수 있을지`를 확인하는 단계였습니다. 공통 backbone은 FedAvg이며, 학습률을 낮추고 update scope를 `norm`으로 제한했습니다.

| 방법 | 설계 포인트 | MSE | MAE | sMAPE |
| --- | --- | ---: | ---: | ---: |
| Fed backbone | 적응 없음 | 0.3032 | 0.3075 | 130.73 |
| baseline_norm | norm-only update | 0.3038 | 0.3078 | 130.67 |
| m1_mask_t0p1 | inactive hindcast masking | 0.3040 | 0.3080 | 130.66 |
| m3_skip_0p3 | inactive window skip | 0.3039 | 0.3079 | 130.67 |
| comb_m1m3_t0p1_f0p3 | masking + skip 결합 | 0.3042 | 0.3084 | 130.66 |

#### 발견한 한계

- catastrophic failure는 줄었지만, backbone을 의미 있게 넘지 못했습니다.
- inactive 구간 제어는 안정성 관리에는 도움을 줬지만 정확도 개선은 거의 없었습니다.
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

### 5.2 아직 부족한 점

1. 강한 backbone 대비 일관된 정확도 우위가 없습니다.
2. Solar와 Electricity에서 효과가 작아 데이터셋 일반성이 부족합니다.
3. 따라서 현재 스토리는 `정확도 향상형 기법`보다 `안정성/효율 중심의 경량 온라인 적응`에 가깝습니다.

### 5.3 슬라이드용 핵심 메시지

- 초기 질문: FL + TTA + feedback loop를 바로 결합하면 좋아지는가.
- 현재 답: 바로 좋아지지 않았고, direct weight update는 불안정했습니다.
- 핵심 전환: backbone 직접 수정에서 time-affine adapter로 전환했습니다.
- 현재 최종 메시지: Murata에서는 backbone 성능을 거의 보존하면서 adaptation 빈도를 크게 줄이는 구조까지 확보했습니다. 다만 범용 정확도 향상 주장까지는 추가 근거가 더 필요합니다.

## 6. 비교에서 제외한 탐색

2026년 4월 11일 새벽에 수행한 Murata scope sweep, norm resweep, 일부 lambda/drift 탐색은 Murata scaling 복원 이전 결과가 섞여 있어 본문 주 비교표에서는 제외했습니다. 다만 이 초기 탐색은 `norm-only가 상대적으로 안전하다`, `짧은 hindcast가 유리할 수 있다`, `직접 weight update보다 더 작은 적응 단위가 필요하다`는 방향성을 제공했고, 이후 affine adapter 설계 전환의 근거가 됐습니다.
