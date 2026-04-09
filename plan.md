세 가지 수정 사항만 정밀하게 반영한 최종본입니다. 변경된 부분은 **[수정]** 태그로 표시했습니다.

---

# [논문 연구 계획서] DLinear의 성분 분리 기반 Test-Time Adaptation과 연합학습의 유기적 결합을 통한 강건한 시계열 예측

## 1. 연구 배경 및 목적
최근 장기 시계열 예측(LTSF) 분야에서 복잡한 Transformer 계열보다 단순한 1계층 선형 모델(LTSF-Linear)이 우수한 성능을 보인다는 사실이 입증되었다. 하지만 실생활 환경(예: 태양광 발전, 전력망)에서는 기상이변이나 사용자 패턴 변화 등으로 예측 불가능한 도메인 시프트(Domain Shift)가 수시로 발생하며, 프라이버시 문제로 인해 데이터를 중앙 서버에 모아 재학습하기 어렵다.

본 연구는 경량 시계열 모델인 DLinear를 기반으로 연합학습(Federated Learning, FL)과 Test-Time Adaptation(TTA)을 결합한 새로운 온라인 학습 프레임워크를 제안한다. 테스트 타임에 자가 복원(Hindcast)과 동적 성분 페널티를 적용하여 즉각적인 도메인 시프트에 적응하고, 이 결과를 글로벌 서버로 환류하여 모델을 지속 개선하는 **'순환형 학습(FED-TTA Loop)'** 구조를 구축한다.

## 2. 핵심 연구 방법론

### 2.1. 기본 예측 모델 및 업데이트 대상 파라미터
DLinear는 이동 평균 커널(Moving Average Kernel)을 사용하여 원본 시계열을 추세(Trend)와 계절성(Seasonality) 성분으로 분해하며, 각 성분은 독립적인 선형 네트워크($W_{trend}$, $W_{season}$)를 통과하여 합산된다. 본 연구의 TTA 과정에서는 별도의 보조 레이어(예: Batch Normalization) 추가 없이, DLinear의 본연의 가중치인 $W_{trend}$와 $W_{season}$ 자체를 직접 업데이트하여 메모리 오버헤드를 최소화한다.

### 2.2. 연합학습(FL) 구성 및 Non-IID 정량화
각 컬럼(예: 개별 인버터, 가구)을 독립된 클라이언트로 취급한다. 클라이언트 간 데이터 분포의 이질성(Heterogeneity)은 **Wasserstein Distance**로 정량화하여, Non-IID 강도에 따른 제안 기법의 방어력을 입증한다. 로컬 과적합 방지를 위해 글로벌 서버 동기화 주기는 3 Epoch으로 설정한다.

### 2.3. 하이브리드 TTA 손실 함수 설계 [핵심 기여]

기존 TTA 기법(e.g., TENT)이나 연속학습 정규화 기법들은 주로 분류(Classification) 문제에 국한되거나 무거운 연산을 요구한다. 본 연구는 DLinear의 '성분 분리 구조'를 활용해 추세와 계절성을 독립적으로 제어하는 시계열 회귀 최적화 TTA를 제안한다.

**A. 차원 유지 자가 복원 손실 (Shifted-Window Hindcast Loss)**

DLinear의 고정된 입력 차원($L$)과 출력 차원($T$) 구조를 보존하기 위해 이동 윈도우를 활용한다. 현재 시점 $t$를 기준으로, 과거 데이터 $X_{input} = [x_{t-L-k+1}, \dots, x_{t-k}]$를 모델에 입력하여 예측값 $\hat{Y} \in \mathbb{R}^{T}$를 도출한 뒤, 이 중 앞부분 $k$ 스텝을 실제 관측된 최근 데이터 $X_{recent} = [x_{t-k+1}, \dots, x_t]$와 비교한다.

$$L_{recon} = \frac{1}{k} \sum_{i=1}^{k} \| \hat{Y}_{i} - X_{recent, i} \|^2$$

> **설계 제약:** 비교 대상인 $X_{recent}$의 길이 $k$는 반드시 모델의 출력 길이 $T$ 이하여야 한다 ($k \leq T$). 이 조건을 만족하지 않으면 예측값 $\hat{Y}$의 인덱스 범위를 초과하여 연산이 성립하지 않는다. 따라서 Ablation Study에서 $k$의 탐색 범위는 $\{T/4,\ T/2,\ T\}$로 제한한다. (예: $T=96$이면 $k \in \{24, 48, 96\}$)

**B. 연속적 동적 성분 보존 페널티 (Continuous Dynamic Penalty)**

TTA 중 발생하는 치명적 망각(Catastrophic Forgetting)을 막기 위해 두 성분의 가중치 변화를 제한한다.

$$L_{reg} = \lambda_{trend} \| W_{trend}^{TTA} - W_{trend}^{FL} \|^2 + \lambda_{season} \| W_{season}^{TTA} - W_{season}^{FL} \|^2$$

단순 임계치 방식의 불안정성을 극복하고자, 실시간 통계량을 바탕으로 페널티 계수를 **연속적으로 감쇠**시키는 지수 함수를 도입한다. 두 계수는 **표준편차($\sigma_{hist}$)로 정규화된 편차**를 공통 척도로 사용하여 수식의 일관성을 유지한다.

$$\lambda_{trend} = \lambda_0 \cdot \exp\left(-\gamma \cdot \frac{|\mu_{curr} - \mu_{hist}|}{\sigma_{hist}}\right)$$

$$\lambda_{season} = \lambda_0 \cdot \exp\left(-\gamma \cdot \frac{|\sigma_{curr} - \sigma_{hist}|}{\sigma_{hist}}\right)$$

> **수식 일관성 근거:** $\lambda_{trend}$는 평균의 편차를 $\sigma_{hist}$로 나누어 정규화한 z-score 형태를 사용한다. $\lambda_{season}$ 역시 분산이 아닌 **표준편차의 편차**($|\sigma_{curr} - \sigma_{hist}|$)를 동일한 $\sigma_{hist}$로 나누어, 두 계수가 동일한 스케일($[0, \infty)$)에서 작동하도록 통일한다.

**C. 최종 목적 함수:**

$$L_{TTA} = L_{recon} + \alpha L_{reg}$$

### 2.4. 순환형 피드백 루프 및 수렴 안전장치 (Safety Mechanisms)

가중치 변화량(Delta)을 글로벌 모델로 피드백할 때 발생하는 발산 및 Error Accumulation을 방지한다.

- **수렴 보장:** 전송되는 델타 값에 Gradient Clipping을 적용하고, 서버 통합 시 감쇠 계수(Decay Factor)를 적용한다.
- **사전 측정 기반 롤백 (Pre-update Rollback) 및 FL 피드백 연동:** TTA 역전파를 수행하기 **전(Pre-update)**에 측정한 현재 배치의 $L_{recon}$이 과거 누적 이동 평균 $\bar{L}_{recon}$의 3배를 초과할 경우, 해당 배치를 이상치(Outlier)로 간주하여 가중치 업데이트를 Skip하고 가중치를 유지한다. **이 경우 해당 배치의 Delta는 FL 서버 피드백에서도 동시에 제외**하여, 이상 배치로 인한 노이즈가 글로벌 모델 집계에 유입되지 않도록 한다.

## 3. 실험 및 검증 계획

### 3.1. 데이터셋
1. **자체 데이터셋 (Custom):** 15분 단위 데이터.
2. **Solar 데이터셋:** 10분 단위 데이터.
3. **Electricity 데이터셋:** 1시간 단위 데이터.

### 3.2. 비교군 (Baselines)
1. **Centralized DLinear:** 전체 데이터 중앙 집중 학습 (Upper Bound).
2. **FED:** 기본 연합학습 모델.
3. **DLinear-TTA:** 중앙 집중 모델 + $L_{TTA}$.
4. **FED-TTA:** 연합학습 모델 + 일회성 $L_{TTA}$.
5. **FED-TTA Loop (제안 기법):** 피드백을 포함한 온라인 프레임워크.

### 3.3. 평가 지표
- **예측 성능 (Primary):** 클라이언트 간 스케일을 정규화하는 **sMAPE**를 최우선 지표로 하며, MSE, MAE를 병행 측정한다.
- **연산 및 통신 효율성 (Secondary):** Informer 등 무거운 모델 및 PatchTST, TimesNet 등 최신 경량 모델과 비교하여 Communication Cost(전송 파라미터량)와 Adaptation Overhead(FLOPs 및 1스텝 소요 시간)를 측정한다.

### 3.4. 절제 연구 (Ablation Study)
- 자가 복원 예측 길이($k \leq T$ 범위 내) 및 균형 계수($\alpha$) 최적화.
- 동적 페널티 함수의 민감도($\gamma$) 조절에 따른 적응력 비교.
- Wasserstein Distance 기반 Non-IID 강도에 따른 FED-TTA 성능 하락 방어율 측정.

### 하이퍼파라미터 초기값 및 탐색 범위 (참고)

| 파라미터 | 초기값 | 탐색 범위 | 제약 조건 |
|---|---|---|---|
| $k$ (Hindcast 길이) | $T/4$ | $\{T/4,\ T/2,\ T\}$ | $k \leq T$ 필수 |
| $\alpha$ (균형 계수) | 1.0 | $\{0.1,\ 1.0,\ 10.0\}$ | — |
| $\gamma$ (감쇠 민감도) | 1.0 | $\{0.5,\ 1.0,\ 2.0\}$ | — |
| $\lambda_0$ (페널티 기본값) | 1.0 | $\{0.1,\ 1.0\}$ | — |
| Rollback 임계치 | $3\times$ | $\{2\times,\ 3\times,\ 5\times\}$ | — |

## 4. 기대 효과
본 연구는 라벨이 없는 환경에서도 도메인 시프트를 실시간 감지 및 복원하는 하이브리드 TTA를 제안한다. 기존 방식과 뚜렷이 차별화되는 '독립적 성분 제어 및 연속적 감쇠 함수'를 도입하여 강건성을 극대화했다. DLinear 기반의 선형 구조 덕분에 최신 SOTA 경량 시계열 모델(PatchTST 등) 대비 압도적으로 낮은 파라미터 수와 FLOPs를 요구하므로, 자원이 매우 제한된 엣지 컴퓨팅(Edge Computing) 환경에서도 원활한 FL 구동 및 실시간 TTA가 가능하다. 이는 실제 IoT 기반의 확장형 시계열 예측 시스템을 위한 핵심 패러다임이 될 것으로 기대한다.