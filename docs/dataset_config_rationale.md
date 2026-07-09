# 데이터셋별 설정 근거 (메모[임6] 답변)

GATE_v2.pdf 메모[임6]: "데이터셋별 입력 길이, 커널 크기가 다른 이유? 변량 개수는?"에 대한 근거 정리.

---

## 1. 입력 길이(seq_len)가 다른 이유

샘플링 간격이 데이터셋마다 다르기 때문에, **물리적 시간 컨텍스트를 동일하게 맞추기 위해** seq_len을 다르게 설정하였다.

| 데이터셋 | 간격 | seq_len | 계산 | 실제 시간 |
|---|---|---|---|---|
| Murata | 15 min | 192 | 192 × 15min | **48시간** |
| Solar | 10 min | 288 | 288 × 10min | **48시간** |
| Electricity | 1 hour | 336 | 336 × 1h | **14일** |

- Murata, Solar: 동일 물리 길이(48시간) 기준으로 정렬
- Electricity: 시간 단위 데이터로 14일(336시간) 컨텍스트 적용

근거 파일:
- `configs/murata/centralized.yaml`
- `configs/solar/centralized.yaml`
- `configs/electricity/centralized.yaml`

---

## 2. 커널 크기(kernel_size)가 다른 이유

DLinear의 moving average 연산에서 trend 성분을 추출할 때 사용되는 평활 윈도우이며,  
**데이터셋별 샘플링 간격의 역수로 스케일하여 물리적 평활 구간을 동일하게 유지**하도록 설계되었다.

| 데이터셋 | 간격 | kernel_size | 계산 | 평활 구간 |
|---|---|---|---|---|
| Murata | 15 min | 49 | 49 × 15min ≈ 735min | **~12시간** |
| Solar | 10 min | 73 | 73 × 10min = 730min | **~12시간** |
| Electricity | 1 hour | 25 | 25 × 1h = 25h | **~1일** |

- Murata, Solar: 약 12시간 단위의 일중 반주기(half-daily) 트렌드 평활
- Electricity: 약 1일 단위 트렌드 평활

근거 파일:
- `docs/report.md` 섹션 2.5.1

---

## 3. 변량(채널) 개수

| 데이터셋 | 변량 수 | 구조 | 비고 |
|---|---|---|---|
| Electricity | **321** | 다변량 (단일 CSV) | 전력 소비 미터 321개 |
| Solar | **137** | 다변량 (단일 CSV) | 태양광 패널 137개 |
| Murata | **30** | 연합학습 구조 (클라이언트 단위) | 클라이언트 30개, 각 단일 채널(`p`: 활성전력) |

근거 파일:
- Electricity: 데이터셋 CSV 헤더 (date + 0~320 + OT = 321 수치형 컬럼)
- Solar: 데이터셋 CSV 헤더 (LocalTime + 0~136 = 137 수치형 컬럼)
- Murata: `datasets/archive/murata_15min_legacy_full/manifest.json` (clients 배열 30개)

---

## 논문 반영 제안

표1에 **변량 개수 열 추가** 및 본문에 입력 길이·커널 크기 설계 근거 1~2문장 추가 필요:

> "입력 길이는 각 데이터셋의 샘플링 간격에 맞춰 물리적 컨텍스트(Murata·Solar 48시간, Electricity 14일)를 동일하게 확보하도록 설정하였으며, 커널 크기는 약 12시간(Murata·Solar) 또는 1일(Electricity)의 트렌드 평활 구간을 유지하도록 간격 역수 비례로 결정하였다."
