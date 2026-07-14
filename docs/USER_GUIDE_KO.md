# PerceptionISP 사용 매뉴얼

## 1. 목적과 현재 범위

PerceptionISP는 하나의 RAW 입력에서 다음 두 경로를 비교하기 위한
소프트웨어 reference입니다.

- **HumanISP**: 사람이 보기 좋은 RGB를 만드는 고정 baseline.
- **PerceptionISP**: machine-view RGB와 Aux map을 함께 생성하는 경로.

Aux map은 기존 RGB detector에 자동으로 사용되지 않습니다. 학습형
경로에서는 RGB stem과 Aux stem을 첫 feature stage에서 결합하고, 그 뒤의
pretrained backbone과 task head를 사용합니다. 학습하지 않은 Aux map은
diagnostic 또는 proposal-calibration baseline으로만 해석해야 합니다.

## 2. 설치

Python 3.11 이상을 사용합니다.

```bash
git clone https://github.com/SeongcheolJeong/PerceptionISP.git
cd PerceptionISP
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

필요한 기능에 따라 dependency를 추가합니다.

```bash
python -m pip install -e '.[raw]'        # NEF/DNG 등 native RAW
python -m pip install -e '.[ml]'         # detection/segmentation 학습
python -m pip install -e '.[camerae2e]'  # CameraE2E HDF5 dependency
python -m pip install -e '.[all]'        # 위 runtime 기능 전체
```

CameraE2E 자체는 별도 local checkout이며 pip extra에 포함되지 않습니다.
`.[camerae2e]`는 HDF5 bridge 의존성만 추가하므로, 새 환경에서는
CameraE2E checkout도 `python -m pip install -e "$CAMERAE2E_ROOT"`로
설치해야 SciPy, scikit-image 등 simulator 전체 의존성이 준비됩니다.

## 3. 작업 경로

개인 절대경로 대신 아래 환경변수를 사용합니다.

```bash
export PERCEPTION_ISP_DATA=/volume/datasets
export PERCEPTION_ISP_OUTPUT=/volume/perceptionisp-results
export CAMERAE2E_ROOT=/path/to/CameraE2E
```

설정하지 않으면 각각 `./data`, `./reports`를 사용하며 CameraE2E는
비활성 상태입니다. 데이터, 모델, export, report는 Git에 올라가지
않습니다.

## 4. CLI 구조

모든 public workflow는 `perception-isp`에서 시작합니다.

```bash
perception-isp --help
perception-isp isp run --help
perception-isp example suite --help
perception-isp data yolo-aux --help
perception-isp aux export --help
perception-isp train yolo-aux --help
perception-isp evaluate detection --help
perception-isp report rollup --help
```

그룹은 `isp`, `example`, `data`, `aux`, `train`, `evaluate`, `report`입니다.

## 5. ISP smoke test

외부 데이터 없이 synthetic RAW를 생성하고 전체 ISP를 실행합니다.

```bash
perception-isp isp run \
  --width 640 --height 384 \
  --cfa RGGB \
  --demosaic-method edge_aware \
  --output-dir "$PERCEPTION_ISP_OUTPUT/smoke_rggb"
```

주요 산출물은 다음과 같습니다.

- `perception_isp_outputs.npz`: accurate/fast tensor, normalized RAW, Aux maps.
- `summary.json`: channel, latency estimate, provenance, health 정보.
- `vision_rgb.ppm`: PerceptionISP machine-view RGB.
- `human_rgb.ppm`: 해당 run에서 HumanISP 경로가 활성화된 경우의 RGB.

다른 Bayer pattern도 같은 계약으로 실행할 수 있습니다.

```bash
for cfa in RGGB GRBG BGGR GBRG; do
  perception-isp isp run --cfa "$cfa" \
    --output-dir "$PERCEPTION_ISP_OUTPUT/smoke_$cfa"
done
```

## 5.1 실행형 HDR/Metadata 예제 Suite

외부 dataset이나 PyTorch 없이 HDR, sensor metadata, calibration, CFA,
temporal state, DNN 입력 계약을 한 번에 확인할 수 있습니다.

```bash
perception-isp example suite \
  --output-dir "$PERCEPTION_ISP_OUTPUT/perception_isp_example_suite"
```

생성되는 `index.html`은 `Overview`, `HDR`, `Sensor Metadata`,
`Calibration & Optics`, `Temporal`, `DNN Contract`, `CameraE2E` 탭으로
구성됩니다. CFA와 geometry case는 `Calibration & Optics` 탭 안에서 함께
표시됩니다. `summary.json`에는 동일한 PASS/FAIL gate와 metric이 저장되고,
`assets/`에는 HumanISP/PerceptionISP 영상과 Aux map이 저장됩니다.

일부만 빠르게 실행할 수도 있습니다.

```bash
perception-isp example suite --case hdr --case metadata
perception-isp example suite --case dnn-contract
perception-isp example suite --list-cases
```

Metadata 표의 의미는 다음과 같습니다.

- `active processing`: 실제 pixel/map/timing 계산에 사용됩니다.
- `propagated only`: provenance와 sidecar에는 남지만 pixel 계산에는 직접
  사용되지 않습니다.
- `declared but unused`: 계약에는 존재하지만 현재 pipeline block이
  소비하지 않습니다.

현재 6채널·16채널 RGB+Aux tensor에는 공간 Aux map만 포함됩니다. 노출시간,
gain, 온도 같은 scalar metadata는 report와 manifest sidecar에는 기록되지만
DNN input에는 들어가지 않습니다. 이 예제의 PASS는 front-end mechanism
검증이며 detection/segmentation 성능 우월성 근거가 아닙니다.

CameraE2E를 fallback 없이 필수 검증하려면 다음처럼 실행합니다.

```bash
export CAMERAE2E_ROOT=/path/to/CameraE2E
python -m pip install -e "$CAMERAE2E_ROOT"
python -m pip install -e '.[camerae2e]'
perception-isp example suite --with-camerae2e --scene 'uniform ee'
```

`--with-camerae2e`를 지정했는데 직접 실행할 수 없으면 suite는 실패합니다.
현재 CameraE2E bridge의 HDR stack은 하나의 simulated sensor mosaic을
scale하여 만든 synthetic bracket이므로 true multi-capture HDR 근거로
해석하면 안 됩니다.

## 6. CameraE2E 입력

```bash
export CAMERAE2E_ROOT=/path/to/CameraE2E
python -m pip install -e "$CAMERAE2E_ROOT"
perception-isp isp run \
  --camerae2e \
  --scene 'uniform ee' \
  --cfa auto \
  --width 640 --height 384
```

`auto`는 CameraE2E sensor가 제공하는 source CFA를 사용합니다. 명시적인
`RGGB`, `GRBG`, `BGGR`, `GBRG`를 지정하면 target CFA가 되고 source/target
정보와 remosaic 여부를 provenance에 기록합니다. claim용 run에서는
source CFA와 ISP target CFA가 일치하고 remap fraction이 0인지 확인합니다.

RGB annotation dataset을 CameraE2E에 넣는 경우 원 scene의 해상도와
sensor output 해상도를 구분해야 합니다. scene을 먼저 sensor 크기로
줄여서 넣으면 high-information scene 검증이 되지 않습니다.

## 7. HumanISP와 PerceptionISP detection 비교

외부 모델 없이 plumbing만 확인하는 실행입니다.

```bash
perception-isp evaluate detection \
  --source synthetic \
  --count 8 \
  --width 640 --height 384 \
  --output-dir "$PERCEPTION_ISP_OUTPUT/detection-smoke"
```

실제 YOLO 비교 예시는 다음과 같습니다.

```bash
perception-isp evaluate detection \
  --source kitti-dataset \
  --dataset "$PERCEPTION_ISP_DATA/kitti/object" \
  --split training \
  --count 128 \
  --width 640 --height 192 \
  --cfa auto \
  --rgb-detector yolo \
  --rgb-detector-model yolo11n.pt \
  --label-aware \
  --output-dir "$PERCEPTION_ISP_OUTPUT/kitti-detection"
```

HumanISP와 PerceptionISP에 같은 RAW sample, annotation subset, detector
checkpoint, confidence, image size를 사용해야 합니다. 한쪽에만 threshold나
label remap을 적용한 결과는 matched comparison으로 사용하지 않습니다.

## 8. RGB+Aux tensor export

PASCAL RAW native Bayer 예시입니다.

```bash
perception-isp aux export \
  --source pascalraw-dataset \
  --dataset "$PERCEPTION_ISP_DATA/raw_datasets/pascalraw" \
  --pascalraw-manifest "$PERCEPTION_ISP_DATA/raw_datasets/pascalraw/subset.json" \
  --pascalraw-native-raw \
  --count 512 \
  --width 640 --height 384 \
  --cfa auto \
  --channels edge6 \
  --no-preview \
  --output-dir "$PERCEPTION_ISP_OUTPUT/pascalraw-edge6"
```

`edge6`는 RGB와 선택된 edge evidence를 저장합니다. 전체 extended tensor가
필요하지 않은 실험에서 저장 공간과 I/O를 줄일 수 있습니다. 생성된
`manifest.jsonl`이 이후 dataset conversion의 입력입니다.

## 9. YOLO dataset 생성과 split

동일 manifest에서 RGB-only와 RGB+Aux를 각각 생성해야 공정한 paired
comparison이 됩니다.

```bash
perception-isp data yolo-aux \
  --manifest "$PERCEPTION_ISP_OUTPUT/pascalraw-edge6/manifest.jsonl" \
  --channel-mode rgb_only \
  --split-strategy hash \
  --output-dir "$PERCEPTION_ISP_OUTPUT/yolo-rgb"

perception-isp data yolo-aux \
  --manifest "$PERCEPTION_ISP_OUTPUT/pascalraw-edge6/manifest.jsonl" \
  --channel-mode rgb_aux \
  --split-strategy hash \
  --output-dir "$PERCEPTION_ISP_OUTPUT/yolo-rgb-aux"
```

작고 가늘거나 aspect ratio가 큰 object를 train에 더 자주 보이게 하되
validation/test에는 중복시키지 않으려면 resplit 도구를 사용합니다.

```bash
perception-isp data resplit \
  --source "$PERCEPTION_ISP_OUTPUT/yolo-rgb-aux" \
  --destination "$PERCEPTION_ISP_OUTPUT/yolo-rgb-aux-resplit" \
  --seed 17 \
  --eval-fraction 0.2 \
  --hard-repeat 3 \
  --write-split-manifest "$PERCEPTION_ISP_OUTPUT/split-seed17.json"
```

RGB-only dataset에도 같은 split manifest를 재사용합니다.

## 10. Detection 학습

### RGB-only baseline

RGB-only는 표준 Ultralytics 모델과 동일한 학습 budget으로 실행합니다.

```bash
yolo detect train \
  data="$PERCEPTION_ISP_OUTPUT/yolo-rgb/data.yaml" \
  model=yolo11n.pt epochs=30 imgsz=640 batch=8 seed=17
```

### Aux feature warm-up

Aux stem이 RGB teacher의 early feature를 근사하도록 먼저 학습할 수 있습니다.

```bash
perception-isp train feature-distill \
  --data "$PERCEPTION_ISP_OUTPUT/yolo-rgb-aux/data.yaml" \
  --teacher runs/detect/rgb-baseline/weights/best.pt \
  --epochs 5 --batch 8 --seed 17 \
  --project "$PERCEPTION_ISP_OUTPUT/training" \
  --name aux-distill-s17
```

### Gated early-fusion fine-tuning

```bash
perception-isp train yolo-aux \
  --data "$PERCEPTION_ISP_OUTPUT/yolo-rgb-aux/data.yaml" \
  --model runs/detect/rgb-baseline/weights/best.pt \
  --epochs 30 --imgsz 640 --batch 8 --seed 17 \
  --aux-stem-mode gated_norm_sum \
  --aux-stem-init adapter \
  --aux-feature-adapter "$PERCEPTION_ISP_OUTPUT/training/aux-distill-s17/best.pt" \
  --aux-stem-gate-init -2.0 \
  --aux-stem-freeze-rgb-branch \
  --project "$PERCEPTION_ISP_OUTPUT/training" \
  --name rgb-aux-s17
```

권장 순서는 RGB warm start, Aux feature distillation, 작은 gate contribution,
RGB branch freeze, hard-case oversampling입니다. 이후 seed를 최소 3개 이상
반복하고 held-out set의 bootstrap confidence interval을 비교합니다.

## 11. Segmentation

LIS sidecar dataset을 만들 때 RGB image/label과 RAW/Aux image/label의 sample
ID가 정확히 일치해야 합니다.

```bash
perception-isp data lis-rgb-aux \
  --rgb-yolo-root "$PERCEPTION_ISP_DATA/lis/yolo-rgb" \
  --aux-yolo-root "$PERCEPTION_ISP_DATA/lis/yolo-raw" \
  --output-dir "$PERCEPTION_ISP_OUTPUT/lis-rgb-aux" \
  --aux-mode sobel_luma
```

pretrained segmenter로 RAW transform을 먼저 screening할 수 있습니다.

```bash
perception-isp evaluate segmentation \
  --subset-root "$PERCEPTION_ISP_DATA/lis/subset" \
  --model yolo11n-seg.pt \
  --screen-transforms \
  --output-dir "$PERCEPTION_ISP_OUTPUT/lis-screen"
```

scene-truth synthetic mask를 사용한 RGB/RGB+Aux paired training은 다음과
같이 실행합니다.

```bash
perception-isp train segmentation \
  --train-count 512 --val-count 128 \
  --cfas RGGB,GRBG,BGGR,GBRG \
  --psf-sigmas 0.0,0.8,1.6 \
  --adverse lowlight_noise \
  --epochs 20 --seed 17 \
  --model-variant aux_detail_side \
  --rgb-aux-init rgb_preserve_zero_aux \
  --output-dir "$PERCEPTION_ISP_OUTPUT/scene-seg-s17"
```

segmentation에서는 mask IoU만 보지 말고 boundary F1, small/thin-object slice,
low-MTF, low-light, demosaic artifact 조건을 함께 평가합니다.

## 12. 결과 해석

- RGB+Aux의 parameter 수가 더 많다는 사실만으로 개선을 주장할 수 없습니다.
- training metric이 좋아도 held-out metric과 confidence interval이 나쁘면
  일반화 개선이 아닙니다.
- recall 증가와 FP 증가가 동시에 발생하면 operating-point tradeoff입니다.
- box boundary는 실제 contour GT가 아니므로 edge feasibility proxy입니다.
- RGB annotation은 HumanISP appearance에 편향될 수 있습니다. 가능한 경우
  native RAW, scene-truth mask, geometry/renderer truth를 함께 사용합니다.

현재 근거와 금지해야 할 주장은
[EVIDENCE_AND_LIMITATIONS.md](EVIDENCE_AND_LIMITATIONS.md)에 정리되어 있습니다.

## 13. 저장 공간 관리

Git에서 제외되는 기본 디렉터리는 `data`, `reports`, `exports`, `runs`,
`outputs`, `logs`, `models`, `imagegen_report`입니다. dataset archive를
삭제하기 전에는 다음을 확인합니다.

1. 압축이 정상적으로 풀렸는지 확인합니다.
2. manifest가 원본 파일을 archive 내부 경로로 참조하지 않는지 확인합니다.
3. 최소 smoke load를 실행합니다.
4. 삭제 후 다시 받을 수 있는 URL과 checksum을 기록합니다.

`scripts/data/`의 download helper는 free-space guard를 포함하지만 실행 전
대상 경로와 예상 archive 크기를 확인해야 합니다.

## 14. 문제 해결

### `perception-isp`가 보이지 않음

```bash
source .venv/bin/activate
python -m pip install -e .
python -m perception_isp --help
```

### CameraE2E import 실패

`CAMERAE2E_ROOT`가 checkout root인지 확인하고 `${CAMERAE2E_ROOT}/src` 아래에
`pyisetcam`이 존재하는지 확인합니다.

### `rawpy`, `torch`, `ultralytics` 없음

각각 `.[raw]`, `.[ml]` extra를 설치합니다. 기본 설치는 lightweight ISP
smoke test만 보장합니다.

### 이전 checkpoint가 module path 오류로 열리지 않음

v0.2 training/evaluation 모듈은 기존 gated-stem pickle module alias를
등록합니다. 그래도 실패하면 [MIGRATION_V0_2.md](MIGRATION_V0_2.md)의
checkpoint migration 절차를 사용합니다.
