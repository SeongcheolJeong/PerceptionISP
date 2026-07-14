# 휴머노이드 VLA / Perception ISP / 센서 퓨전 논의 정리본

**작성 목적:** 다른 작업 공간에서 보고서 초안으로 활용할 수 있도록, 본 채팅에서 논의한 내용을 주제별 기술 문서 형태로 재구성한다.  
**작성일:** 2026-06-17  
**범위:** 실내로봇, 휴머노이드 로봇, 창고·물류·공장 환경, Perception ISP, 48MP/12MP 카메라 전략, VLA/E2E 연산 요구, 센서 퓨전, SoC 선정.

---

## 0. Executive Summary

본 대화의 핵심 결론은 다음이다.

1. **휴머노이드 로봇의 인지 문제는 단순 object detection이 아니다.**  
   이동, 조작, 사람 안전, 언어 grounding, 센서 신뢰도 판단, 실패 후 재관찰이 모두 결합된다.

2. **카메라 해상도는 head camera 기준 상시 12MP급이 적정하고, 48MP는 on-demand detail reserve로 쓰는 것이 합리적이다.**  
   48MP 전체를 상시 VLA/vision encoder에 넣는 구조는 비효율적이다. 일반 운용은 2×2 binning된 12MP stream으로 하고, 글자·라벨·작은 부품·정밀 조작 후보가 있을 때 48MP ROI 또는 snapshot을 사용하는 구조가 적절하다.

3. **Perception ISP의 역할은 vision encoder를 대체하는 것이 아니라, vision encoder와 VLA가 볼 데이터를 줄이고 신뢰도를 알려주는 sensor-aware front-end가 되는 것이다.**  
   Perception ISP는 edge, SNR, saturation, color ratio, blur/MTF confidence, depth invalid, ROI trigger, sensor mode control을 담당해야 한다.

4. **48MP/12MP sensor mode switching이 가능하더라도 Perception ISP의 역할은 사라지지 않는다.**  
   오히려 Perception ISP는 “언제 48MP로 볼지”, “어느 ROI를 고해상도로 볼지”, “현재 12MP 판단이 믿을 만한지”를 결정하는 sensor-attention controller가 되어야 한다.

5. **VLA/E2E 시스템에서 TOPS는 주로 language/VLA backbone, vision encoder, diffusion/action policy에서 크게 소모된다.**  
   카메라 raw/image stream은 TOPS보다 먼저 I/O, ISP throughput, DDR bandwidth, frame buffer를 압박한다. 하지만 이미지가 neural token으로 변환되는 vision encoder 단계부터는 카메라도 TOPS를 크게 소모한다.

6. **Vision encoder 전체를 ISP에 넣는 것은 비추천이다.**  
   본격적인 CNN/ViT/CLIP/DINO/VLA visual encoder는 NPU/GPU/AI accelerator에서 처리하고, ISP 또는 camera-side에서는 작은 ROI trigger, confidence map, sensor-native feature 생성까지만 담당하는 것이 맞다.

7. **창고·물류·공장용 휴머노이드 센서 구성은 카메라 중심에 3D LiDAR, safety scanner, 60GHz radar, wrist vision, tactile/force sensing을 결합하는 구조가 적절하다.**  
   제품형 기준 외부 인지 센서 모듈은 약 22~24개 수준이 적정선이다.

8. **중앙 SoC는 NVIDIA Jetson AGX Thor T5000 또는 산업형 NVIDIA IGX Thor T5000급이 적합하다.**  
   단, safety와 motor control은 Thor에 맡기지 않고 별도 safety MCU / real-time controller로 분리해야 한다.

9. **올바른 시스템 구조는 하이브리드다.**  
   카메라 쪽은 sensor-native feature, confidence, ROI trigger를 생성하는 “똑똑한 센서”가 되고, 중앙 fusion SoC는 multi-sensor fusion, vision encoder, VLA/VLM, policy를 담당한다. 안전 제어는 독립 safety MCU가 담당한다.

---

## 1. 전체 시스템 방향

### 1.1 권장 아키텍처

```text
[Camera / Sensor Modules]
  - RAW sensor, 2×2 binning, HDR, exposure control
  - Perception ISP
  - edge / SNR / saturation / color / blur / depth confidence
  - ROI trigger
  - 48MP mode request
        ↓
[Central Fusion SoC: Thor class]
  - vision encoder
  - multi-camera fusion
  - depth / LiDAR / radar / tactile / proprioception fusion
  - VLA / VLM reasoning
  - diffusion/action policy
  - scene memory
        ↓
[Real-time Controller / Safety MCU]
  - 500Hz~1kHz motor control
  - balance / reflex
  - e-stop / motor inhibit
  - safety scanner / radar / proximity direct path
```

핵심은 다음과 같다.

```text
카메라 쪽:
센서 고유 정보 보존 + ROI/신뢰도 생성

중앙 SoC:
semantic understanding + multi-sensor fusion + VLA/action policy

안전 MCU:
VLA와 독립된 hard safety / stop / motor inhibit
```

### 1.2 피해야 할 구조

```text
피해야 할 구조 1:
모든 카메라 raw full-resolution stream을 중앙 SoC로 보내고,
중앙 SoC가 ISP, perception, fusion, VLA, safety, motor control을 모두 담당

문제:
- bandwidth / DDR / thermal 폭증
- VLA가 low-level pixel 처리까지 떠안음
- safety가 Linux/AI stack에 종속됨
- sensor-native confidence 정보 손실
```

```text
피해야 할 구조 2:
카메라 모듈 안에 full vision encoder나 task-specific detector를 과도하게 넣음

문제:
- VLA 모델 업데이트와 카메라 펌웨어가 강하게 결합됨
- multi-camera / tactile / joint state를 함께 판단할 수 없음
- camera vendor lock-in 위험
- camera마다 AI compute가 중복됨
```

---

## 2. 휴머노이드 로봇이 놓이는 환경과 인지 챌린지

### 2.1 대표 환경

| 환경 | 예시 | 주요 인지 챌린지 |
|---|---|---|
| 가정 / 사무실 / 병원 / 매장 | 거실, 주방, 병실, 회의실, 매장 | 사람·반려동물, 전선, 유리, 거울, 문서/화면 privacy, 조명 변화 |
| 창고 / 물류 | 선반, 박스, 팔레트, 컨베이어 | 라벨/OCR, 바코드, 반복 박스, 작업자/지게차/AMR 공존 |
| 공장 | 금속 부품, 설비, 공구, 조립 라인 | 반사체, 작은 부품, safety zone, 먼지/진동/조명 flicker |
| 주방 / 테이블 / 선반 / 서랍 | 컵, 접시, 도구, 손잡이, 버튼 | 조작, occlusion, 투명/반사/검은 물체, deformable object |
| 계단 / 문 / 엘리베이터 | 문턱, 계단, 손잡이, 엘리베이터 버튼 | 이동+조작 결합, 발 디딤, 문 열림 방향, 버튼/OCR |
| degraded 환경 | 저조도, 역광, 먼지, 렌즈 오염 | low SNR, saturation, blur, rolling shutter, confidence 저하 |

### 2.2 핵심 인지 문제

휴머노이드의 perception은 다음 문제를 동시에 풀어야 한다.

```text
1. 이동 가능한 공간을 이해해야 함
2. 사람과 안전하게 공존해야 함
3. 물체를 보고, 잡고, 조작해야 함
4. 글자/라벨/버튼 같은 작은 detail을 읽어야 함
5. 투명/반사/검은 물체를 다뤄야 함
6. 손/팔/몸에 의한 occlusion을 극복해야 함
7. 카메라, depth, LiDAR, radar, tactile, joint state를 시간동기화해야 함
8. VLA가 언어 명령을 실제 물체와 행동에 정확히 ground해야 함
9. 센서가 틀릴 때 스스로 confidence를 낮추고 다시 봐야 함
```

### 2.3 가장 어려운 object/scene 조합

| 우선순위 | object / scene | 어려운 이유 |
|---:|---|---|
| 1 | 전선, 케이블, 펜, 나사, 커넥터 | 작고 얇으며 12MP/low-res에서 놓치기 쉬움 |
| 2 | 투명 컵, 유리문, 아크릴 파티션 | camera/depth/LiDAR 모두 실패 가능 |
| 3 | 금속, 유광 바닥, shrink wrap 포장재 | specular, glare, ghost, depth invalid |
| 4 | 검은 케이블, 검은 옷, 어두운 가구 | low SNR, low contrast |
| 5 | 사람 손/발/아이/반려동물 | 빠르고 예측 어려움, safety-critical |
| 6 | 버튼, 도어락, 작은 라벨, QR/barcode | 고해상도 ROI/OCR 필요 |
| 7 | 손에 가려진 물체 | manipulation 중 head camera 무력화 |
| 8 | 서랍/선반 안쪽의 어두운 물체 | occlusion + low light + close range |
| 9 | deformable object: 비닐, 천, 종이, 음식 | shape와 상태가 계속 변함 |
| 10 | 렌즈 오염, motion blur, rolling shutter | 시스템이 실패를 모르면 위험 |

---

## 3. 실내로봇 일반 인지 문제

초기 논의에서 실내로봇의 센서 인지 문제를 다음처럼 정리했다.

| 문제 | 실제 장면 | 주 실패 센서 | 결과 |
|---|---|---|---|
| 투명 물체 미검출 | 유리문, 유리 테이블, 투명 컵 | RGB-D, ToF, LiDAR | depth hole, 충돌 |
| 반사면 오인식 | 거울, 유광 바닥, TV 화면 | camera, ToF, LiDAR | fake map, ghost obstacle |
| 검은 물체 문제 | 검은 양말, 검은 전선, 검은 러그 | camera, ToF, cliff sensor | low SNR, miss |
| 작고 낮은 장애물 | 전선, 레고, 펜, 문턱 | camera, LiDAR, bumper | stuck, 흡입, 충돌 |
| dynamic object | 사람 다리, 아이, 반려동물 | camera, LiDAR, RGB-D | SLAM noise, safety risk |
| 얇은 구조물 | 의자 다리, 테이블 다리 | 2D LiDAR, camera | sparse hit, detection miss |
| 상부 돌출물 | 식탁 상판, 서랍, 침대 프레임 | 2D LiDAR | LiDAR는 지나가도 몸체 충돌 |
| 단차/낙하 | 계단, 현관 턱, 매트 끝 | cliff, depth, odometry | false cliff 또는 실제 낙하 miss |
| 조명 문제 | 창가 역광, LED flicker, 야간 | RGB camera, RGB-D | exposure, flicker, blur |
| 반복 구조 | 흰 벽, 긴 복도, 비슷한 문 | visual/LiDAR SLAM | localization drift |
| 센서 오염 | 먼지, 물방울, 지문 | 전 센서 | failure를 시스템이 모름 |
| calibration/time sync | camera+LiDAR+IMU | fusion stack | object 위치 오류 |

실내용 Perception ISP는 이 문제를 해결하기 위해 **작은 물체 edge, depth invalid reason, 저조도 confidence, specular/glass confidence, sensor health**를 후단에 넘기는 쪽이 적합하다.

---

## 4. 색 인지와 센서 QE 시뮬레이션

### 4.1 색은 왜 중요한가

실내로봇과 휴머노이드에서 색은 다음 작업에 중요하다.

| 장면 | 색이 중요한 이유 |
|---|---|
| 바닥 테이프 / 구역 표시 | 노란선, 빨간선, 파란선 navigation cue |
| 충전독 LED / 장비 LED | 상태 구분 |
| 물류 라벨 / 박스 색상 | sorting, picking |
| 음식 / 액체 / 오염물 | 분류와 안전 판단 |
| 사람/손/옷 | 사람 interaction, safety |
| 버튼 / 스위치 / 경고 스티커 | task grounding |
| 유리/금속/반사체 | specular/color shift 자체가 물성 clue |

하지만 목표는 사람 눈 기준의 정확한 색 재현이 아니다.

```text
중요한 목표:
1. color class separability
2. color constancy
3. sensor-to-sensor color domain shift 감소
4. saturation/specular/flicker 상황의 color confidence
```

### 4.2 QE/CFA/SPD 시뮬레이션 필요성

카메라 raw RGB는 다음 요소의 곱적분으로 결정된다.

```text
Raw_R/G/B =
∫ Illuminant_SPD(λ)
  × Object_Reflectance(λ)
  × Lens_Transmittance(λ)
  × IR-cut_Filter(λ)
  × CFA_Response_R/G/B(λ)
  × Sensor_QE(λ)
  dλ
+ noise
```

따라서 같은 물체라도 조명 SPD, 센서 QE, CFA, 렌즈, IR cut, AWB/CCM이 다르면 raw/RGB가 크게 달라진다.

### 4.3 시뮬레이션 수준

| 수준 | 내용 | 용도 |
|---|---|---|
| Level 1 | sRGB augmentation, brightness/color jitter, noise | 빠른 robustness 검토 |
| Level 2 | 조명 SPD, object reflectance, sensor QE×CFA×IR-cut, raw noise, AWB/CCM variation | 센서 후보 비교, 제품 개발 |
| Level 3 | hyperspectral DB, 실제 조명 측정, 센서 spectral sensitivity 측정, lens CRA/color shading, production variation | 양산 전 검증, 산업/의료/고신뢰 |

결론:

```text
단순 장애물 회피에는 full spectral simulation이 필수는 아님.
하지만 LED, 라벨, 바닥선, 오염물, 물체 class처럼 색을 decision variable로 쓰면
QE/CFA/조명 SPD를 포함한 spectral-level simulation이 필요함.
```

### 4.4 Perception ISP color output

Perception ISP는 예쁜 RGB 대신 아래 정보를 후단에 제공해야 한다.

```text
1. log(R/G), log(B/G)
2. opponent color: R-G, B-G, R+B-2G
3. saturation / clipping map
4. AWB gain map
5. illuminant estimate / confidence
6. color reliability map
7. low-light color confidence
8. LED flicker confidence
9. specular highlight map
10. NIR leak / IR contamination confidence
```

---

## 5. TOPS와 연산량 논의

### 5.1 기본 계산식

일반적인 계산식은 다음이다.

```text
Ideal TOPS = Σ(모델 1회 연산량[GOPs] × 실행 Hz × duty ratio) / 1000

Required Peak TOPS = Ideal TOPS / 실제 utilization × safety margin
```

현실적인 실제 utilization은 다음과 같이 낮을 수 있다.

| 작업 | 실제 utilization 감각 |
|---|---:|
| CNN detector / segmentation | 25~60% |
| transformer encoder / fusion | 15~40% |
| autoregressive VLM/VLA | 5~25% |
| diffusion policy | 10~35% |
| 작은 batch, 여러 모델 동시 실행 | 더 낮음 |

따라서 제품 스펙은 보통 다음처럼 잡는다.

```text
Required Peak TOPS ≈ Ideal TOPS × 3~10배
```

### 5.2 실내로봇 perception TOPS 감각

| 시스템 수준 | 권장 TOPS 감각 |
|---|---:|
| 단순 주행 + 장애물 회피 | 3~10 TOPS |
| 로봇청소기 고급형 | 5~15 TOPS |
| 실내 배송/안내로봇 | 15~40 TOPS |
| 가정용 서비스로봇 | 30~70 TOPS |
| 조작 가능한 서비스로봇 | 70~150 TOPS |
| 휴머노이드/VLA 포함 | 150~275+ TOPS, large VLA는 Thor급 |

### 5.3 Perception ISP로 TOPS를 얼마나 줄일 수 있나

기존 RGB DNN 중심 구조와 Perception ISP + ROI 구조를 비교하면 다음과 같다.

```text
기존 RGB DNN 중심 구조:        20~50 TOPS
Perception ISP + RGB 축소 구조: 5~15 TOPS
공격적 ROI/sparse 구조:         2~8 TOPS
특정 기능 전용:                 1~5 TOPS 가능
```

중요한 점은 **RGB 3채널을 1채널로 줄이는 것만으로는 큰 TOPS 감소가 어렵다**는 것이다. 첫 convolution만 줄고 후단 feature channel 연산이 대부분이기 때문이다. 실질적 절감은 다음에서 나온다.

```text
1. full resolution을 저해상도 map으로 줄이기
2. full-frame inference를 ROI/tile inference로 바꾸기
3. 매 frame inference하지 않고 temporal reuse하기
4. 큰 segmentation model을 task-specific small model로 분리하기
5. RGB 대신 edge/SNR/color/depth-confidence feature를 넘기기
```

### 5.4 12MP와 48MP의 기본 TOPS 감각

가정:

```text
12MP = 4000×3000
48MP = 8000×6000
48MP는 12MP보다 pixel 수 4배
```

#### Rule-based / local Perception ISP map

| 처리 수준 | pixel당 연산 | 12MP@20fps | 48MP@20fps |
|---|---:|---:|---:|
| 아주 가벼운 map | 50 ops/px | 0.012 TOPS | 0.048 TOPS |
| 일반 local map | 200 ops/px | 0.048 TOPS | 0.192 TOPS |
| 꽤 무거운 filtering | 1000 ops/px | 0.24 TOPS | 0.96 TOPS |

즉, rule-based/local Perception ISP 자체는 대체로 1 TOPS 이하에서도 가능하다. 실제 병목은 TOPS보다 memory bandwidth와 ISP throughput일 수 있다.

#### Full-frame DNN 감각

640×640 ≈ 0.41MP 기준 작은 detector가 있다고 하면:

| 기준 모델 | 640 입력 1회 | 12MP 1회 | 48MP 1회 |
|---|---:|---:|---:|
| Tiny detector | 6.5 GOPs | 약 190 GOPs | 약 760 GOPs |
| Small detector | 20 GOPs | 약 586 GOPs | 약 2.34 TOPS |
| Medium detector | 70 GOPs | 약 2.05 TOPS | 약 8.2 TOPS |

Tiny detector 30fps 기준 ideal:

| 해상도 | 10fps ideal | 20fps ideal | 30fps ideal |
|---|---:|---:|---:|
| 12MP | 약 1.9 TOPS | 약 3.8 TOPS | 약 5.7 TOPS |
| 48MP | 약 7.6 TOPS | 약 15.2 TOPS | 약 22.8 TOPS |

실제 하드웨어 nominal은 ideal의 3~10배가 필요할 수 있다.

결론:

```text
12MP full-frame DNN: 10~50 TOPS급
48MP full-frame DNN: 40~200 TOPS급
Perception ISP + downscale + ROI 구조: 12MP 상시 5~15 TOPS, 48MP ROI 포함 10~30 TOPS 가능
```

### 5.5 VLA/E2E TOPS 계산

VLA/E2E는 다음 블록으로 나눈다.

```text
Camera / depth / LiDAR / IMU / tactile
        ↓
Perception ISP
        ↓
Image encoder / depth encoder / LiDAR encoder
        ↓
Temporal / spatial fusion
        ↓
VLA or E2E policy
        ↓
Action decoder / diffusion policy / trajectory head
        ↓
Safety filter / controller
```

주기별 감각:

| 블록 | 일반적 주기 |
|---|---:|
| Motor control | 200~1000Hz, MCU/RT CPU |
| IMU / joint state | 100~1000Hz |
| Safety proximity | 30~100Hz |
| Camera perception | 10~30Hz |
| Depth / LiDAR fusion | 10~30Hz |
| E2E visuomotor policy | 10~30Hz |
| VLA high-level reasoning | 1~10Hz |
| Large VLM/VLA semantic reasoning | 0.5~5Hz 가능 |

#### Compact E2E 예시

```text
camera encoder: 30 GOPs × 4 cameras × 15Hz = 1.8 TOPS
Depth/LiDAR: 30 GOPs × 15Hz = 0.45 TOPS
Fusion: 100 GOPs × 10Hz = 1.0 TOPS
Diffusion policy: 50 GOPs × 8 steps × 10Hz = 4.0 TOPS
Safety detector: 20 GOPs × 30Hz = 0.6 TOPS

Ideal total ≈ 7.85 TOPS
Required nominal ≈ 30~50 TOPS
```

#### 3B VLA + manipulation

```text
Ideal equivalent: 약 15~40 TOPS
Required peak: 약 80~250 TOPS급
Memory: 32~64GB 권장
```

#### 7B VLA + multi-camera humanoid

```text
Compact mode: 100~300 TOPS급
3B VLA 실시간: 200~500 TOPS급 equivalent
7B VLA + multi-camera + low latency: 500~1000+ TOPS equivalent
Large VLM/VLA edge reasoning: Thor-class, 1000~2000+ FP4 TFLOPS급
```

### 5.6 Language model과 camera의 자원 압박 차이

질문: “TOPS는 language model에서 주로 먹고, camera image는 TOPS보다 memory에 영향을 준다고 보면 되는가?”

정리:

```text
대체로 맞지만 정확히는 다음처럼 나눠야 한다.

Camera raw path:
I/O, ISP throughput, DDR bandwidth, frame buffer를 먼저 압박

Vision encoder:
이미지를 neural feature/token으로 바꾸는 순간부터 TOPS와 activation memory를 크게 소모

Language/VLA backbone:
TOPS/TFLOPS, memory capacity, memory bandwidth를 모두 크게 소모
```

카메라 raw example:

```text
12MP RAW10 @ 30fps ≈ 3.6Gbps ≈ 450MB/s
48MP RAW10 @ 30fps ≈ 14.4Gbps ≈ 1.8GB/s
```

그러나 실제 DDR traffic은 RAW write/read, ISP intermediate, RGB/YUV, resize/crop, DNN input, logging 등을 거치며 더 커진다.

대형 VLA example:

```text
7B model weight:
FP16 ≈ 14GB
INT8 ≈ 7GB
FP4  ≈ 3.5GB
```

여기에 vision encoder, activation, KV/cache, TensorRT workspace, camera buffers, middleware가 추가된다.

---

## 6. 카메라 해상도와 48MP/12MP 전략

### 6.1 실내로봇 일반 카메라 해상도

| 용도 | 권장 해상도 |
|---|---:|
| 기본 주행/인지 | 1~2MP |
| 전선/양말/장난감 검출 | 2~3MP |
| 사람/반려동물/물체분류/조작 | 3~5MP |
| 라벨/문자/정밀조작 | 5~12MP, ROI용 |

실내로봇에서는 센서 해상도와 DNN 입력 해상도를 분리해야 한다.

```text
센서 입력: 2~5MP 또는 그 이상
Perception ISP output: 320×180~640×360 global feature
DNN 입력: low-res global + ROI crop
```

### 6.2 휴머노이드 head camera

휴머노이드 VLA용 head camera는 **상시 12MP급**이 적정하다.

```text
Head camera:
48MP Quad Bayer sensor
상시 12MP 2×2 binned 20~30fps
필요 시 48MP ROI / snapshot
HFOV 110~140°
HDR, hardware sync, row timestamp
```

이유:

```text
1. wide FOV에서 손, 라벨, 버튼, 작은 부품의 pixel margin 확보
2. 2MP는 scene understanding은 가능하나 OCR/detail에는 부족
3. 48MP 상시는 bandwidth/TOPS/thermal 부담이 과함
4. 12MP binned는 SNR, bandwidth, 실시간성의 균형점
5. 48MP는 high-res reserve로 적합
```

### 6.3 Wrist camera

Wrist camera는 48MP보다 3~5MP global shutter가 적합하다.

```text
Wrist camera:
3~5MP global shutter
30~60fps
close focus 10~80cm
HFOV 70~100°
HDR / good MTF / low latency
```

이유:

```text
1. 손목 카메라는 가까운 물체를 보므로 head camera만큼 높은 MP가 필요하지 않음
2. 조작 중 motion과 occlusion이 많아 global shutter와 latency가 중요
3. close focus와 렌즈 MTF가 실제 성공률에 더 직접적
```

### 6.4 48MP + 2×2 binning을 요구하는 이유

FigureAI류 요구로 논의된 48MP + 2×2 binning의 의도는 다음으로 해석된다.

```text
Full 48MP mode:
작은 글자, 라벨, 버튼, 손끝, 도구, 작은 부품, ROI crop

2×2 binning ≈ 12MP mode:
상시 perception, 저조도 SNR, bandwidth 절감, lower power
```

즉, 48MP는 상시 AI 입력이 아니라 **high-resolution reserve**다.

### 6.5 글자 발견 시 48MP 전환 시나리오

가장 현실적인 구조:

```text
상시:
12MP binned stream으로 global perception

이벤트:
text-like / label / QR / 작은 부품 / 손끝 조작 후보 발견

상세 확인:
48MP ROI crop 또는 snapshot

후처리:
OCR / fine recognition / pose refinement
```

다만 실제 구현은 다음처럼 soft dual-path일 가능성이 크다.

```text
12MP stream → 완전 정지 → 48MP mode 전환
```

보다는:

```text
low-res global stream 유지
+ 필요 시 high-res ROI/snapshot path 활성화
```

### 6.6 하드웨어적 메리트

48MP + 2×2 binning은 48MP peak 하드웨어가 필요하므로 BoM 자체를 줄이는 기술은 아니다. 메리트는 **상시 시스템 부하 감소**다.

| 항목 | 줄어드는가 | 설명 |
|---|---|---|
| sensor cost | 거의 안 줄어듦 | 48MP sensor 필요 |
| lens requirement | 안 줄어듦 | 48MP를 살리려면 고MTF 필요 |
| peak link | 크게 안 줄어듦 | 48MP capture 필요 시 peak 대응 |
| average link | 줄어듦 | 평상시 12MP stream |
| ISP 평균 처리량 | 크게 줄어듦 | 48MP full ISP 상시 회피 |
| DDR bandwidth | 줄어듦 | frame size 1/4 |
| NPU/TOPS | 크게 줄어듦 | full 48MP inference 회피 |
| 전력/발열 | 줄어듦 | average duty 낮음 |
| 저조도 SNR | 좋아짐 | binning으로 effective pixel 증가 |

### 6.7 고객 관점의 메리트

고객 입장에서는 전력 절감이 큰 메리트지만, 더 넓게는 **system resource scheduling의 이득**이다.

```text
평상시:
저전력 / 저발열 / 낮은 bandwidth / 낮은 NPU 점유율

필요 시:
48MP detail / OCR / 라벨 / 손끝 / 작은 물체 인식
```

핵심 표현:

```text
48MP camera is not intended for continuous full-resolution AI inference.
It provides high-resolution reserve for on-demand inspection.
```

---

## 7. 48MP/12MP switching과 Perception ISP의 관계

### 7.1 switching만으로 해결되는 것

48MP/12MP sensor mode switching은 다음을 해결한다.

```text
1. 평상시 bandwidth 절감
2. 평상시 ISP 부하 절감
3. 평상시 NPU 입력 축소
4. 저조도 SNR 개선
5. 필요 시 고해상도 OCR/라벨 확인
6. full-frame 48MP 상시 처리 회피
```

따라서 Perception ISP를 단순히 “해상도를 줄여 TOPS 낮추는 ISP”로 정의하면 메리트가 약해진다.

### 7.2 Perception ISP가 여전히 필요한 이유

스위칭만으로는 다음을 해결하지 못한다.

```text
1. 12MP 상태에서 48MP가 필요한 순간을 어떻게 찾을 것인가?
2. 어떤 ROI를 48MP로 볼 것인가?
3. 12MP 판단이 충분히 믿을 만한가?
4. blur, low SNR, saturation, depth invalid 때문에 재관찰이 필요한가?
5. multi-camera 중 어떤 카메라를 high-res로 올릴 것인가?
6. 48MP activation을 얼마나 자주 허용할 것인가?
```

따라서 Perception ISP의 포지션은 다음으로 바뀌어야 한다.

```text
약한 포지션:
Perception ISP = RGB를 줄여 TOPS를 낮추는 ISP

강한 포지션:
Perception ISP = sensor mode controller + confidence generator + ROI proposer
```

### 7.3 sensor mode switching은 actuator

정리:

```text
Sensor mode switching:
48MP와 12MP 중 어떤 해상도로 볼 수 있게 해주는 하드웨어 기능

Perception ISP:
12MP로 충분한지, 48MP가 필요한지,
어느 ROI를 봐야 하는지,
현재 색/노이즈/포화/blur 때문에 결과를 믿어도 되는지 판단하는 기능
```

즉, 48MP/12MP switching은 Perception ISP를 대체하는 기능이 아니라 **Perception ISP가 제어해야 할 actuator**에 가깝다.

---

## 8. Vision Encoder는 ISP에 포함되는가?

### 8.1 결론

**Vision encoder 전체를 ISP에 포함시키는 것은 비추천이다.**

추천 구조:

```text
RAW sensor
   ↓
Classical ISP + Perception ISP
   ↓
저비용 perception feature / confidence / ROI
   ↓
Vision encoder on NPU/GPU
   ↓
Fusion / VLA / E2E policy
```

### 8.2 기능 구분

| 기능 | 위치 |
|---|---|
| black level, DPC, LSC | ISP |
| demosaic, HDR merge, AWB, CCM | ISP |
| edge / SNR / saturation / color ratio / blur confidence | Perception ISP |
| ROI proposal / small trigger classifier | ISP-side / DSP / small NPU 가능 |
| CNN/ViT image encoder | 중앙 NPU/GPU |
| CLIP/DINO/OpenVLA visual encoder | 중앙 NPU/GPU |
| multi-camera transformer fusion | 중앙 NPU/GPU |
| VLA/VLM backbone | 중앙 NPU/GPU |
| action policy / diffusion head | 중앙 NPU/GPU |

### 8.3 왜 vision encoder를 ISP에 넣으면 위험한가

```text
1. 모델이 자주 바뀜
2. task마다 encoder 구조가 다름
3. precision 요구가 바뀜: FP16, BF16, INT8, FP8, FP4 등
4. attention, normalization, residual 구조가 ISP fixed pipeline과 안 맞음
5. activation memory가 큼
6. multi-camera / temporal fusion과 결합됨
7. VLA/E2E와 end-to-end로 함께 바뀔 가능성이 큼
```

### 8.4 ISP에 넣을 수 있는 micro-encoder

아래 정도는 ISP-side 또는 vision accelerator에 넣을 수 있다.

```text
text-like region detector
thin cable candidate detector
floor boundary trigger
glass/specular risk detector
small obstacle trigger
human/pet proximity trigger
```

이것은 full vision encoder가 아니라 **trigger encoder / micro encoder**다.

---

## 9. ROI Trigger 설계

### 9.1 기본 개념

ROI trigger는 최종 인식기가 아니다. **비싼 고해상도/고정밀 인식을 어디에 쓸지 정하는 attention proposal**이다.

```text
12MP binned / low-res normal stream
        ↓
Perception ISP cheap maps
        ↓
tile별 ROI score 계산
        ↓
temporal persistence / tracking
        ↓
ROI 후보 N개 선택
        ↓
1) 고해상도 crop DNN
2) 48MP snapshot / ROI capture
3) OCR / fine recognition / manipulation refinement
```

### 9.2 ROI trigger 입력

```text
1. RAW luma 또는 Bayer G luma
2. CFA-aware edge map
3. noise / SNR map
4. saturation / clipping map
5. color ratio: log(R/G), log(B/G)
6. local contrast map
7. texture / corner density map
8. temporal difference / motion map
9. blur / MTF confidence
10. depth invalid / specular confidence
```

### 9.3 Tile score 방식

```text
ROI_score =
  w1 × edge_score
+ w2 × thin_object_score
+ w3 × text_like_score
+ w4 × color_event_score
+ w5 × motion_score
+ w6 × low_confidence_score
+ w7 × depth_inconsistency_score
+ w8 × task_prior_score
- w9 × noise_only_penalty
- w10 × saturated_unrecoverable_penalty
```

Top-K tile을 고르고, 주변 tile을 merge한 뒤, bbox expansion과 temporal tracking을 수행한다.

### 9.4 주요 feature

#### Noise-normalized edge

```text
edge_conf = |∇L| / (estimated_noise_sigma + epsilon)
```

단순 gradient가 아니라 **현재 노이즈 수준 대비 edge가 의미 있는지**가 중요하다.

#### Thin-object score

대상:

```text
전선, 케이블, 의자 다리, 테이블 다리, 손가락, 펜, 커넥터, 문턱, 러그 끝
```

계산 개념:

```text
thin_score = elongated_component_score
           × edge_confidence
           × floor_or_hand_proximity_prior
           × temporal_persistence
```

#### Text-like score

글자 후보 특징:

```text
작은 edge 밀집
stroke width 일관성
수평/수직/대각 edge 혼합
local contrast
반복적인 small component
rectangular label/panel prior
```

#### Color event score

```text
color_event_score =
distance(tile_color_ratio, local_background_color_ratio)
× color_confidence
× non_saturation_confidence
```

#### Low-confidence trigger

“뭔가 보인다”뿐 아니라 “현재 잘 안 보이는데 위험할 수 있다”도 ROI trigger가 되어야 한다.

```text
low_confidence_score =
low_SNR + high_blur + high_saturation + depth_invalid + specular + lens_contamination
```

### 9.5 ROI packet

후단 VLA/fusion에 넘길 ROI packet은 bbox만으로 부족하다.

```text
ROI packet:
- bbox: x, y, w, h
- score
- trigger_type
- required_mode: low-res / 12MP crop / 48MP crop / 48MP snapshot
- confidence_reason: edge / text / color / motion / depth_invalid / low_snr / saturation
- estimated_scale
- temporal_age
- world_position, 가능하면
- recommended_exposure, 가능하면
- sensor_mode_request
```

예:

```text
ROI #3
bbox: [1420, 880, 320, 180]
score: 0.87
trigger_type: text_like
required_mode: 48MP crop
reason: high edge density + rectangular component + low OCR confidence
temporal_age: 4 frames
```

### 9.6 ROI trigger 성능 지표

```text
1. trigger recall
2. trigger precision
3. high-res activation rate
4. average ROI count/frame
5. missed critical ROI
6. trigger-to-capture latency
7. energy per successful recognition
8. downstream OCR/grasp/collision improvement
```

핵심 목표:

```text
48MP를 적게 켜면서,
정말 필요한 순간은 놓치지 않는 것
```

---

## 10. Camera-side vs Central Sensor Fusion Chip

### 10.1 결론

정답은 하이브리드다.

```text
Camera-side:
sensor-native perception front-end

Central fusion chip:
multi-sensor fusion + vision encoder + VLA/E2E policy

Safety controller:
real-time control / safety stop / motor reflex
```

### 10.2 Camera-side에 넣어야 할 기능

```text
1. 2×2 binning / subsampling / ROI readout control
2. RAW black level / defect / lens shading
3. HDR exposure source map
4. saturation / clipping map
5. noise / SNR map
6. CFA-aware edge map
7. thin-object candidate
8. blur / MTF confidence
9. color ratio: log(R/G), log(B/G)
10. rolling-shutter row timestamp
11. lens contamination / occlusion score
12. ROI proposal
13. 48MP snapshot / crop trigger
```

### 10.3 Central fusion chip에 남겨야 할 기능

```text
1. vision encoder
2. multi-camera fusion
3. depth / LiDAR / tactile / force fusion
4. temporal memory
5. VLA / VLM reasoning
6. object affordance
7. task-conditioned perception
8. manipulation policy
9. whole-body planning
10. semantic map
11. human interaction
```

### 10.4 판단 기준

| 질문 | Yes이면 camera-side | No이면 central fusion chip |
|---|---|---|
| RAW/Bayer 상태에서만 잘 보이나? | O | |
| pixel streaming 중 line-buffer로 처리 가능한가? | O | |
| 모든 task에 공통으로 유용한가? | O | |
| output이 confidence/ROI/metadata인가? | O | |
| 여러 센서와 task context가 필요한가? | | O |
| 모델이 자주 바뀌는가? | | O |
| VLA와 end-to-end로 학습될 가능성이 큰가? | | O |
| tactile/joint/IMU와 함께 판단해야 하는가? | | O |

---

## 11. LiDAR와 Radar 구성

### 11.1 역할 분담

```text
Camera:
semantic / VLA visual understanding / OCR / 사람 / 물체

Depth:
near-field 3D / manipulation support

LiDAR:
3D geometry / SLAM / occupancy / free-space

Radar:
근거리 사람 존재 / 상대속도 / blind spot / 안전 정지

Tactile / force:
접촉 후 실패 감지와 force control
```

### 11.2 LiDAR 추천

```text
개수:
1개

위치:
upper torso 또는 chest 상단

스펙:
360° horizontal
40~60° 이상 vertical
10~20Hz
30~50m class
0.1m blind zone 이하 선호
Ethernet
PTP sync
IP65 이상 권장
```

용도:

```text
3D occupancy
SLAM 보강
free-space
작업자/팔레트/선반 geometry
카메라 실패 시 geometric fallback
```

### 11.3 Radar 추천

```text
개수:
4개

위치:
front torso
rear torso
left side torso
right side torso

스펙:
60GHz FMCW
0.2~10m normal range
15~25m max class
20~50Hz
range / azimuth / Doppler velocity / track ID
safety MCU에도 직접 연결
```

용도:

```text
작업자 접근
지게차/AMR 접근
후방/측면 blind spot
사람 근접 safety
먼지/저조도/역광에서 camera 보완
```

### 11.4 고급 구성

```text
1× torso 360° 3D LiDAR
4× torso 60GHz radar
2× forearm/wrist short-range radar or ToF
4× foot/drop ToF
2× safety laser scanner
```

---

## 12. 창고·물류·공장용 센서 구성 선정

### 12.1 제품형 추천 센서 구성

| 구분 | 개수 | 권장 스펙 | 주 역할 |
|---|---:|---|---|
| Head RGB stereo | 2 | 48MP Quad Bayer, 상시 12MP 2×2 binning, 20~30fps, HDR, 100~140° HFOV | 사람, 박스, 선반, 라벨 후보, 장면 이해 |
| Wrist RGB camera | 2 | 3~5MP global shutter, 30~60fps, close focus 10~80cm | 피킹, grasp, 손-물체 접촉, 커넥터/부품 |
| Head depth / RGB-D | 1 | 720p~1MP depth, 15~30fps, 0.5~6m | 팔레트, 선반, 박스 거리, 충돌 회피 |
| Wrist short-range depth / ToF | 2 | 0.1~1.5m, 30fps, confidence map | 집기 직전 3D 보정 |
| 360° 3D LiDAR | 1 | 360° H, 40~60°+ V, 10~20Hz, 30~50m급, Ethernet/PTP | 3D occupancy, SLAM, free-space |
| Safety laser scanner | 2 | 전/후방 또는 대각 배치, 270°급, protective field 4~6m+ | 인증 가능한 정지/감속 안전 경로 |
| 60GHz mmWave radar | 4 | 전/후/좌/우, 0.2~10m normal, 20~50Hz, range/velocity/track | 작업자 접근, blind spot, 지게차/AMR 접근 |
| Arm/hand proximity ToF/IR | 4~6 | 팔꿈치/손목/그리퍼 주변, 0.05~1m | 팔 충돌, 손 주변 근접 안전 |
| Foot/drop ToF or depth | 4 | 양발 전/후 또는 발끝/뒤꿈치 | 단차, 팔레트 모서리, 발 디딤 안전 |
| IMU | 3 | torso, head, pelvis 또는 torso/head/wrist | 보행 안정, rolling shutter 보정, 센서 동기 |
| Wrist force-torque | 2 | 6-axis, 200~1000Hz | 삽입, 밀기, 접촉, 집기 실패 감지 |
| Finger tactile | 10 fingertip + palm optional | 50~500Hz | 미끄러짐, 접촉 위치, 파지 안정 |
| Foot pressure / force | 2 | 500~1000Hz | 보행, 균형, 접지 판단 |
| Joint sensing | 전 관절 | encoder/current/torque/temp, 500~1000Hz | 상태 추정, 제어, safety |

외부 인지 센서 모듈만 계산하면 대략:

```text
2 head RGB
+ 2 wrist RGB
+ 1 head depth
+ 2 wrist depth
+ 1 LiDAR
+ 2 safety scanner
+ 4 radar
+ 4~6 arm/hand proximity
+ 4 foot/drop
= 22~24개
```

내부/접촉/관절 센서는 채널 수 기준 50~100+개가 될 수 있다.

### 12.2 배치안

#### Head

```text
2× 48MP/12MP stereo RGB
1× head depth or active stereo
1× head IMU
optional mic array
```

#### Chest / torso

```text
1× 360° 3D LiDAR
4× radar: front/rear/left/right
1× torso IMU
2× safety laser scanner
```

#### Arms / wrists / hands

```text
2× wrist RGB global shutter
2× wrist ToF/depth
2× wrist F/T
4~6× proximity ToF/IR
10× fingertip tactile
```

#### Feet / lower body

```text
4× foot/drop ToF
2× foot pressure sensor
joint encoders/current/torque/temp
```

### 12.3 창고·공장 특화 perception 기능

```text
1. 사람/작업자/지게차/AMR tracking
2. pallet / rack / shelf / box geometry
3. 라벨 / OCR / barcode ROI trigger
4. shrink wrap / 금속 / 반사체 confidence
5. black object / cable / strap detection
6. dynamic safety zone
7. arm sweep collision prediction
8. grasp failure detection via F/T + tactile
9. 48MP high-res inspection trigger
10. sensor health / lens contamination detection
```

---

## 13. SoC / Compute 선정

### 13.1 결론

창고·물류·공장용 휴머노이드의 메인 SoC는 다음이 적합하다.

```text
제품형/산업형 1순위:
NVIDIA IGX Thor T5000 계열
+ Safety MCU
+ camera-side Perception ISP / sensor gateway

개발형/로봇 본체 탑재형 1순위:
NVIDIA Jetson AGX Thor T5000
+ 외부 Safety MCU
+ GMSL/CSI/Ethernet sensor carrier

Cost-down:
Jetson T4000, 단 VLA/카메라/메모리 요구 축소 필요

이전 세대/compact policy:
Jetson AGX Orin 64GB, compact E2E 또는 cloud VLA용
```

### 13.2 SoC 선정 기준

```text
1. VLA / VLM / policy 모델 크기
2. 카메라 수, 해상도, FPS
3. LiDAR / radar / depth / tactile 입력량
4. vision encoder와 fusion 주기
5. 메모리 용량과 bandwidth
6. safety / real-time control 분리 가능성
7. 전력 / 열 설계
8. 소프트웨어 생태계와 양산 지원
```

### 13.3 최소/권장 스펙

| 항목 | 최소선 | 권장선 |
|---|---:|---:|
| AI compute | 1000 FP4 TFLOPS급 | 2000 FP4 TFLOPS급 |
| Memory | 64GB | 128GB |
| Memory bandwidth | 200GB/s 이상 | 270GB/s 이상 |
| Power envelope | 70W급 | 100~130W sustained 설계 |
| CPU | 12-core급 | 14-core급 + RT controller 분리 |
| Sensor I/O | multi-camera + GbE | GMSL/CSI + 25GbE/10GbE + PCIe |
| Vision accelerator | 있으면 좋음 | PVA / camera offload / ISP-side accelerator |
| Safety | 외부 MCU 필수 | IGX safety island + 외부 safety MCU |
| Storage | 1TB NVMe | 2~4TB NVMe for logging |
| Software | Linux + ROS2 | JetPack / Isaac ROS / TensorRT / VLA stack |

### 13.4 Jetson AGX Thor T5000 / IGX Thor T5000을 고르는 이유

```text
1. 7B급 VLA 또는 multi-camera VLA를 edge에서 돌릴 수 있는 compute/memory margin
2. 128GB memory로 camera buffers, VLA weights, KV/cache, TensorRT workspace, logging 동시 처리 가능
3. FP4/FP8/transformer inference 방향과 맞음
4. high-speed sensor ingest 지원
5. Isaac / TensorRT / GR00T / ROS ecosystem과 맞음
6. 물류·공장형 로봇에서 central fusion + policy용으로 적합
```

### 13.5 Safety MCU 분리

Thor가 메인 AI 두뇌라면 safety MCU는 반사 신경이다.

```text
Safety MCU 역할:
- e-stop
- safety laser scanner input
- radar near-zone hazard
- arm proximity
- foot/drop sensor
- joint limit
- force overload
- motor inhibit
- watchdog
```

권장 방향:

```text
AURIX TC4x/TC3xx 또는 동급 safety MCU
별도 real-time motor controller
Thor hang 시에도 motor inhibit 가능
```

### 13.6 Sensor gateway / camera-side SoC

제품형에서는 camera-side 또는 sensor gateway를 추가하는 것이 좋다.

```text
역할:
1. 48MP/12MP mode control
2. Perception ISP map 생성
3. ROI trigger
4. timestamp / sync
5. camera stream compression or crop
6. radar/LiDAR preprocessing
7. sensor health monitoring
```

후보 방향:

| 방식 | 장점 | 단점 |
|---|---|---|
| Custom ISP/FPGA/ASIC | 전력/latency 최적화, RAW 접근성 좋음 | 개발비 큼 |
| TDA4VM급 vision SoC | camera gateway / ROI trigger / low-power vision에 적합 | 메인 VLA용은 아님 |
| Thor 내부 PVA/camera offload | 개발 단순 | camera-side bandwidth 절감은 제한적 |

### 13.7 최종 BOM 구조

```text
Main AI:
NVIDIA IGX Thor T5000 또는 Jetson AGX Thor T5000

Memory:
128GB LPDDR5X class

Storage:
2TB NVMe minimum, 4TB recommended

Safety:
AURIX/safety MCU
certified safety scanner direct path
radar/proximity direct safety path
motor inhibit independent of Thor

Sensor gateway:
Camera-side Perception ISP / FPGA / TDA4VM-class gateway optional

Network:
GMSL/CSI camera aggregator
Ethernet/PTP LiDAR
CAN-FD/EtherCAT/TSN body network
25GbE backbone where needed
```

---

## 14. Perception ISP 제품 포지셔닝

### 14.1 나쁜 포지셔닝

```text
Perception ISP = vision encoder를 ISP 안에 넣는 것
```

문제:

```text
central VLA/SoC와 충돌
모델 업데이트 난이도 증가
camera vendor lock-in 우려
multi-sensor fusion 불가
```

### 14.2 좋은 포지셔닝

```text
Perception ISP = VLA가 볼 필요 없는 pixel을 줄이고,
vision encoder가 놓치기 쉬운 sensor-native signal을 보존하며,
언제 고해상도 ROI를 봐야 하는지 알려주는 sensor-aware front-end
```

### 14.3 고객에게 설명할 핵심 가치

```text
우리는 VLA를 대체하지 않는다.
우리는 VLA가 더 적은 token으로, 더 낮은 전력으로,
더 안정적으로 보게 만드는 smart sensor front-end를 제공한다.
```

### 14.4 핵심 기능 리스트

```text
1. RAW/Bayer 기반 edge/SNR/color/saturation/blur confidence
2. text/label/small-object ROI trigger
3. 48MP/12MP mode switching control
4. high-res activation 최소화
5. central SoC bandwidth / DDR / TOPS 절감
6. sensor health / contamination detection
7. VLA용 ROI packet / confidence packet 생성
8. depth invalid / specular/glass hint 제공
9. rolling-shutter / exposure / timestamp metadata 제공
10. simulation/QE 기반 color confidence calibration
```

---

## 15. 검증 및 테스트셋 제안

### 15.1 Perception ISP / ROI Trigger 검증

| Set | 장면 | Metric |
|---|---|---|
| Small object | 전선, 펜, 나사, 케이블, 검은 물체 | ROI trigger recall, high-res activation rate, miss rate |
| Text/OCR | 박스 라벨, 바코드, QR, 도어락/버튼 | OCR success, trigger precision/recall, latency |
| Reflective/transparent | shrink wrap, 금속, 유리, 아크릴 | depth invalid detection, collision avoidance |
| Low light | 선반 안쪽, 야간, shadow | SNR confidence calibration, object recall |
| Motion/blur | 보행 중 head motion, wrist motion | blur confidence, tracking stability |
| Human safety | 손/발/작업자 접근 | stop latency, false stop, near-miss rate |
| Manipulation occlusion | 손이 물체를 가리는 상황 | pose refinement, grasp success, tactile recovery |
| Sensor degradation | 렌즈 오염, 먼지, 물방울 | contamination detection, fallback trigger |

### 15.2 창고·공장 통합 검증

```text
1. 작업자와 같은 통로 공유
2. 지게차/AMR 접근
3. 팔레트 주변 보행
4. 선반에서 박스 피킹
5. 반사 포장재가 있는 박스 인식
6. 라벨/OCR 후 물체 선택
7. 실패한 grasp 후 tactile 기반 recovery
8. sensor blackout 또는 Thor stall 시 safety MCU 정지
9. 48MP high-res duty cycle 측정
10. VLA token 수와 task success correlation 측정
```

---

## 16. 고객에게 확인해야 할 질문

### 16.1 48MP/12MP 관련

```text
1. 48MP는 continuous stream인가, occasional snapshot/ROI인가?
2. 48MP full frame을 몇 fps로 요구하는가?
3. 48MP에서 full RGB ISP output이 필요한가, RAW capture만 필요한가?
4. 48MP 전체를 perception에 쓰는가, ROI/OCR용인가?
5. 2×2 binning mode와 48MP mode 전환 latency 허용치는 얼마인가?
6. 모드 전환 중 perception blackout을 허용하는가?
7. 48MP capture 시 로봇이 정지 상태인가, 이동 중인가?
8. center/edge에서 필요한 text size와 working distance는 얼마인가?
```

### 16.2 VLA/E2E 관련

```text
1. E2E는 compact visuomotor policy인가, 3B/7B급 foundation VLA인가?
2. VLA update rate는 1Hz, 5Hz, 10Hz, 30Hz 중 어디인가?
3. action horizon은 몇 step인가?
4. diffusion denoising step은 몇 개인가?
5. policy가 joint command를 직접 내는가, latent action을 내는가?
6. safety perception은 VLA와 독립인가?
7. quantization target은 FP16, INT8, FP8, FP4 중 무엇인가?
8. on-device only인가, cloud/hybrid인가?
9. 목표 latency는 50ms, 100ms, 300ms 중 어디인가?
10. battery/thermal budget은 몇 W인가?
```

### 16.3 센서/제품 관련

```text
1. 작업 환경은 창고, 공장, 병원, 매장 중 어디인가?
2. 사람과 같은 공간에서 어느 정도 가까이 작업하는가?
3. 안전 인증 또는 safety scanner 요구가 있는가?
4. 라벨/OCR/barcode가 core task인가?
5. 반사 포장재, 금속, 투명체가 많은가?
6. 저조도/먼지/진동/물방울이 있는가?
7. 고장 시 fail-safe 요구는 무엇인가?
8. sensor logging을 얼마나 오래 저장해야 하는가?
9. 전체 로봇 전력 예산은 얼마인가?
10. 양산 가격과 개발 기간의 우선순위는 무엇인가?
```

---

## 17. 보고서용 핵심 문장 모음

아래 문장들은 보고서의 executive summary 또는 conclusion에 바로 사용할 수 있다.

1. **휴머노이드 VLA 시스템에서 카메라의 역할은 단순 영상 입력이 아니라, high-resolution detail reserve와 sensor confidence를 제공하는 active sensor이다.**

2. **48MP sensor는 상시 48MP AI inference를 위한 것이 아니라, 평상시 12MP binned stream으로 실시간 perception을 수행하고 필요 시 48MP ROI로 OCR·라벨·정밀 조작을 수행하기 위한 dual-mode 전략이다.**

3. **Perception ISP는 vision encoder를 대체하지 않는다. Perception ISP는 vision encoder와 VLA가 볼 token 수를 줄이고, sensor-native confidence를 제공하며, 48MP 고해상도 모드를 언제 사용할지 결정하는 sensor-attention controller이다.**

4. **VLA/E2E에서 카메라 raw stream은 먼저 I/O와 메모리 대역폭을 압박하고, image encoder 단계부터는 TOPS를 압박한다. Language/VLA backbone은 TOPS뿐 아니라 모델 메모리와 memory bandwidth도 크게 요구한다.**

5. **창고·물류·공장형 휴머노이드의 센서 구성은 head 12MP/48MP stereo, wrist 5MP global shutter, depth, 360° 3D LiDAR, safety scanner, 60GHz radar, tactile/force sensing의 결합이 적합하다.**

6. **중앙 SoC는 Thor T5000급이 적합하지만, safety와 motor control은 반드시 독립 MCU/RT controller로 분리해야 한다.**

7. **카메라 쪽은 더 똑똑한 센서가 되어야 하고, 중앙 fusion chip은 로봇의 두뇌가 되어야 한다. 카메라가 두뇌가 되려 하면 과하고, 중앙칩이 모든 raw sensor 문제까지 떠안으면 비효율적이다.**

---

## 18. 최종 추천 아키텍처 한 장 요약

```text
[Head Vision]
2× 48MP Quad Bayer
- normal: 12MP 2×2 binned 30fps
- detail: 48MP ROI/snapshot
- HDR, sync, row timestamp
        │
        ▼
[Camera-side Perception ISP]
- luma / edge / SNR / saturation
- color ratio / blur / MTF confidence
- text/small-object ROI trigger
- 48MP mode request
        │
        ▼
[Thor T5000 / IGX Thor]
- low-res global visual encoder
- high-res ROI encoder
- depth/LiDAR/radar/tactile/proprioception fusion
- VLA / VLM / policy
- scene memory
        │
        ▼
[RT Controller + Safety MCU]
- 1kHz joint/motor loop
- balance/reflex
- safety scanner/radar/proximity direct stop
- motor inhibit independent of Thor
```

창고·공장형 센서 패키지:

```text
Vision:
2× head 48MP/12MP stereo
2× wrist 5MP global shutter

Depth:
1× head RGB-D/stereo depth
2× wrist short-range ToF/depth

Geometry / safety:
1× torso 360° 3D LiDAR
2× safety laser scanner
4× 60GHz radar
4~6× arm/hand proximity
4× foot/drop sensor

Contact / body:
3× IMU
2× wrist force-torque
10× fingertip tactile
2× foot pressure
all joint encoder/current/torque/temp

Compute:
NVIDIA IGX/Jetson Thor T5000 128GB
+ independent safety MCU
+ optional camera-side sensor gateway
```

---

## 19. 결론

본 대화에서 도출된 큰 방향은 다음이다.

```text
휴머노이드 로봇은 고해상도 카메라 하나로 해결되지 않는다.
카메라, depth, LiDAR, radar, tactile, force, joint/IMU sensing을 시간동기화하고,
Perception ISP로 sensor-native feature와 confidence를 보존한 뒤,
중앙 Thor급 SoC에서 compact visual/geometric/contact token으로 fusion해야 한다.

48MP는 상시 처리 대상이 아니라 high-resolution reserve이며,
12MP binned stream이 normal perception loop의 중심이다.

Perception ISP의 핵심 가치는 해상도 축소 자체가 아니라,
VLA가 언제 무엇을 다시 봐야 하는지 결정하고,
센서가 현재 믿을 만한지 알려주는 것이다.

제품형 창고·공장 휴머노이드에서는 Thor T5000급 중앙 AI,
독립 safety MCU,
head 12MP/48MP vision,
wrist global shutter vision,
3D LiDAR,
safety scanner,
60GHz radar,
tactile/force sensing을 결합하는 구조가 가장 합리적이다.
```

