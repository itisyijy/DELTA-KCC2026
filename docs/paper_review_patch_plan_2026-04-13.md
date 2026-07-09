# `paper.tex` 리뷰 대응 패치안

목표: 추가 대규모 실험 없이 reviewer가 지적한 `재현성`, `claim strength`, `지표 정의`, `기본 인용` 문제를 빠르게 정리한다.

원칙:
- 이번 revision의 중심은 `실험 추가`가 아니라 `기존 결과의 정의 명확화와 서술 조정`이다.
- 구현과 어긋나는 수식 재작성은 피하고, `구현 그대로를 정확히 설명`하는 쪽으로 간다.
- 새 실험이 꼭 필요한 항목처럼 보이는 지점은 `본문 caveat`로 처리한다.

## 1. 우선순위

1. 손실 축/정규화 정의 보강
2. no-harm 관련 문장 톤 다운
3. 지표 정의와 표 캡션 보강
4. 최소 참고문헌 추가
5. proofing 정리

## 2. 구현 기준으로 꼭 밝혀야 할 사실

아래 항목은 코드 기준으로 확인된 내용이다.

- `L_hind`와 `L_cons`는 구현에서 `.mean()`으로 계산된다.
  - 근거: `scripts/tta/loss.py`
  - 즉, 현재 affine loss는 `time x channel x batch` 축 평균 형태로 읽는 것이 가장 정확하다.
- `time_affine`의 파라미터는 `gamma, delta ∈ R^{1 x H x C}`이다.
  - 근거: `scripts/tta/adapter.py`
- 현재 `L_anchor`는 `C`로만 나눈다.
  - 근거: `scripts/tta/loss.py`
  - 따라서 `time_affine`와 `channel_affine`는 자유도 수 기준으로 동일한 regularization scale이 아니다.
- temporal consistency는
  - `y_curr[:, :H-1, :]`와 `y_prev[:, 1:, :]`를 맞춘다.
  - 근거: `scripts/tta/loss.py`
- 테스트 평가 window는 `start += 1`로 이동한다.
  - 근거: `scripts/tta/engine.py`의 `evaluate_client`
  - 즉, 본문에서 `stride=1`을 명시해도 구현과 맞다.
- `deg(%)`는 `100 * (gated - backbone) / backbone`이다.
  - 근거: `scripts/report_kcc_drift_gate_sweep.py`
  - 음수는 backbone 대비 개선이다.
- `parameter ratio`는 `trainable adapter params / total backbone params`이다.
  - 근거: `scripts/report_kcc_drift_gate_sweep.py`
- `sMAPE`는 원 단위 inverse transform 이후
  - `100 * mean(|pred-target| / ((|pred|+|target|)/2 + eps))`
  - 근거: `scripts/utils/metrics.py`
  - 즉, `0-200%` convention이다.

## 3. 섹션별 패치 지시

### A. 인용 추가

위치:
- `paper.tex:82`
- `paper.tex:101`
- `paper.tex:106`
- 문서 끝 `\end{document}` 직전

수정:
- 서론/방법에 최소한 다음 인용을 연결한다.
  - DLinear: Zeng et al., 2023
  - FedAvg: McMahan et al., 2017
  - RevIN: Kim et al., 2022
  - TTA/Test-time training: Sun et al., 2020
  - Tent or 일반 TTA 배경: Wang et al., 2021
  - 가능하면 Murata, Solar, Electricity 데이터셋 출처도 함께 추가한다.

빠른 방법:
- 지금은 `.bib`를 새로 정리하기보다 `thebibliography`를 직접 넣는 방식이 가장 빠르다.

권장 문장 예시:
```tex
장기 시계열 예측에서는 ... DLinear 계열의 강한 선형 backbone은 효율성과 성능을 동시에 제공하지만~\cite{zeng2023dlinear}
```

```tex
\begin{thebibliography}{9}
\bibitem{zeng2023dlinear} ...
\bibitem{mcmahan2017fedavg} ...
\bibitem{kim2022revin} ...
\bibitem{sun2020ttt} ...
\bibitem{wang2021tent} ...
\end{thebibliography}
```

### B. Loss 정의 명확화

위치:
- `paper.tex:148-183`

수정 방향:
- 수식 자체를 구현과 맞는 축 평균으로 명시한다.
- `B, H, C, k`의 의미를 먼저 정의한다.
- `time_affine` 기준 파라미터 shape도 적는다.

권장 삽입 문구:
```tex
이하에서 batch 크기를 $B$, 예측 horizon을 $H$, 출력 채널 수를 $C$, hindcast 길이를 $k$로 둔다.
현재 KCC 비교의 주력 설정은 time-affine이므로 $\hat{Y}_{backbone}, \hat{Y}_{final} \in \mathbb{R}^{B \times H \times C}$이고,
$\gamma, \delta \in \mathbb{R}^{1 \times H \times C}$이다.
별도 언급이 없으면 손실은 batch 평균까지 포함한 평균 제곱오차로 계산한다.
```

`L_hind`는 다음처럼 바꾸는 편이 안전하다.
```tex
\mathcal{L}_{hind}
= \frac{1}{BkC}\sum_{b=1}^{B}\sum_{i=1}^{k}\sum_{c=1}^{C}
\left(\hat{y}^{curr}_{b,i,c}-x^{recent}_{b,i,c}\right)^2.
```

`L_cons`는 다음처럼 바꾸는 편이 안전하다.
```tex
\mathcal{L}_{cons}
= \frac{1}{B(H-1)C}\sum_{b=1}^{B}\sum_{i=1}^{H-1}\sum_{c=1}^{C}
\left(\hat{y}^{curr}_{b,i,c}-\mathrm{sg}\left(\hat{y}^{prev}_{b,i+1,c}\right)\right)^2.
```

`L_anchor`는 구현에 맞추면 다음이 더 정확하다.
```tex
\mathcal{L}_{anchor}
= \frac{1}{C}\left(
\sum_{h=1}^{H}\sum_{c=1}^{C}(\gamma_{h,c}-1)^2
+ \sum_{h=1}^{H}\sum_{c=1}^{C}\delta_{h,c}^2
\right).
```

그리고 바로 아래에 다음 caveat를 추가:
```tex
현재 구현에서는 정규화 분모를 adapter 자유도 전체가 아니라 채널 수 $C$로 두므로,
channel-affine과 time-affine 사이의 regularization strength는 자유도 기준으로 완전히 동일하지 않다.
따라서 adapter mode 간 비교는 구조 차이와 regularization scale 차이가 함께 반영된 결과로 해석해야 한다.
```

### C. Temporal consistency의 stride 가정 명시

위치:
- `paper.tex:172-177` 바로 아래

권장 문장:
```tex
온라인 평가는 연속 테스트 window를 1-step stride로 이동시키며,
위 식은 바로 직전 window의 예측을 한 스텝 시프트해 현재 window와 정렬한다.
```

### D. 평가 지표 정의 보강

위치:
- `paper.tex:233`

현재 문장은 너무 짧다. 아래처럼 구체화:
```tex
평가 지표는 MSE, MAE, sMAPE를 사용했다.
MSE와 MAE는 RevIN 역정규화 직후의 global scale 공간에서 계산한다.
sMAPE는 global scaler까지 완전히 역정규화한 원 단위에서
\[
\mathrm{sMAPE}=100 \cdot \frac{1}{N}\sum_{n=1}^{N}
\frac{|\hat{y}_n-y_n|}{(|\hat{y}_n|+|y_n|)/2+\varepsilon}
\]
로 계산하며, 본문 수치는 0--200\% convention을 따른다.
표의 수치는 모두 클라이언트 평균이다.
```

### E. Threshold 표 캡션과 본문 수정

위치:
- `paper.tex:321-337`

수정:
- 열 제목 `deg(%)`를 `relative MSE change (\%)`로 변경
- 캡션 또는 본문에 `음수는 backbone 대비 개선`을 명시
- `0.3`과 `0.5`가 같은 이유를 한 문장 설명

권장 문장:
```tex
여기서 relative MSE change(\%)는 backbone 대비 상대 변화율이며, 음수는 backbone 대비 개선을 의미한다.
또한 본 sweep에서는 0.3과 0.5가 동일한 aggregate 결과를 보였는데, 이는 적어도 현재 집계 수준에서는 두 threshold가 실질적으로 구분되지 않았음을 시사한다.
```

주의:
- 마지막 문장은 현재 요약 결과가 소수 6자리까지 동일하다는 사실에 근거한 보수적 해석이다.
- per-window gate log까지 확인하지 않았다면 `logged decision이 달랐다/같았다`처럼 단정하지 않는 편이 안전하다.
- 더 강하게 쓰고 싶으면 gate log를 추가 확인한 뒤 `no evaluated window changed decision between 0.3 and 0.5`로 올리면 된다.

### F. no-harm 서술 톤 다운

위치:
- `paper.tex:69`
- `paper.tex:279`
- `paper.tex:341`
- `paper.tex:397`
- `paper.tex:403-405`

핵심 원칙:
- `유지했다`보다는 `near-parity`, `사전 정의한 no-harm 기준 내`, `실질적으로 근접한 수준`으로 낮춘다.

권장 교체 문구:

`paper.tex:279`
```tex
... direct-weight TTA는 세 데이터셋 전반에서 backbone을 일관되게 넘지 못했고, ...
반면 frozen-backbone time-affine은 catastrophic failure를 제거하면서 backbone과 near-parity 수준의 정확도를 보였다.
```

`paper.tex:341`
```tex
표~\ref{tab:kcc-main}은 backbone, 항상 적응하는 기준선, 그리고 \texttt{gate\_1p0}의 최종 비교를 보여준다.
기준선 자체가 이미 backbone과 near-parity 수준의 정확도를 보였고,
\texttt{gate\_1p0} 역시 사전 정의한 no-harm 기준(상대 MSE 열화 $\le 0.5\%$) 안에서 이와 실질적으로 근접한 정확도를 유지하며 selective adaptation을 수행했다.
```

`paper.tex:397`
```tex
\texttt{gate\_1p0}는 세 데이터셋 모두에서 상대 MSE 열화 $\le 0.5\%$를 유지했고,
즉 backbone과 통계적으로 완전히 동일하다고 주장하기보다 사전 허용 열화 범위 내 near-parity를 보였다.
```

`paper.tex:403-405`
```tex
... \texttt{gate\_1p0}가 Murata, Electricity, Solar 모두에서 사전 정의한 no-harm 조건을 만족하는 공통 threshold로 선택되었다.
```

### G. Bootstrap CI 해석 충돌 방지

위치:
- `paper.tex:455-458`

권장 문장:
```tex
평균 차이는 \texttt{gate\_1p0 - backbone}의 raw delta로 계산했다.
따라서 이 표의 bootstrap CI는 차이의 부호와 크기를 보여주며,
본문의 no-harm 판정은 별도로 사전 정의한 상대 MSE 열화 기준($\le 0.5\%$)에 따른다.
```

이 문장을 넣으면 reviewer가 지적한 `CI는 양수인데 왜 유지라고 쓰느냐` 문제를 완화할 수 있다.

### H. 효율 표 정의 보강

위치:
- `paper.tex:371-389`

권장 캡션 수정:
```tex
\caption{공통 gate의 효율 비교. parameter ratio는 trainable adapter parameter 수를 backbone 전체 parameter 수로 나눈 비율이다.}
```

### I. 자잘한 proofing

위치:
- `paper.tex:279`

수정:
- `consistently` -> `일관되게`

## 4. 시간 없을 때 생략해도 되는 것

- multi-seed 추가 실험
- wall-clock/FLOPs 추가 계측
- adapter mode 간 재실험
- 추가 ablation

이 항목들은 논문 본문 한계 문단에 남겨두고 이번 patch에서는 건드리지 않는 편이 낫다.

## 5. reviewer 대응용 한 줄 프레이밍

response letter 또는 rebuttal에 아래 톤을 유지하면 된다.

```text
이번 revision에서는 추가 대규모 재실험보다, 현재 결과가 지지하는 주장 범위에 맞춰 손실 정의, 지표 규약, no-harm 판정 기준, bootstrap CI 해석, 그리고 관련연구 인용을 명시적으로 보강했습니다.
```
