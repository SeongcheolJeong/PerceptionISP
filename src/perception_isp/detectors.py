"""Detector adapters for ISP A/B evaluation.

The pure-numpy detector is only a smoke-test fallback. When ``ultralytics`` and
``torch`` are installed, ``UltralyticsYOLODetector`` provides the real
pretrained detector path.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .eval_types import BoundingBox, Detection, DetectorResult


class DetectorAdapter:
    name = "detector"

    def detect(self, image: Any, *, input_name: str = "image") -> DetectorResult:
        raise NotImplementedError


class NumpyRiskObjectDetector(DetectorAdapter):
    """Small connected-component detector for synthetic smoke tests.

    It is intentionally simple and deterministic. It detects dark road objects
    and bright light blobs from RGB-like images without external dependencies.
    """

    name = "numpy_risk_object_detector"

    def __init__(
        self,
        *,
        min_area_fraction: float = 0.00045,
        max_components: int = 64,
    ) -> None:
        self.min_area_fraction = float(min_area_fraction)
        self.max_components = int(max_components)

    def detect(self, image: Any, *, input_name: str = "image") -> DetectorResult:
        start = time.perf_counter()
        rgb = _as_rgb(image)
        luma = np.mean(rgb[:, :, :3], axis=2)
        rows, cols = luma.shape
        road_mask = _road_roi(rows, cols)
        dark_mask = (luma < np.percentile(luma[road_mask], 18.0)) & road_mask
        bright_mask = (np.max(rgb[:, :, :3], axis=2) > 0.78) & (luma > 0.45)
        red_light_mask = _traffic_light_color_mask(rgb)
        candidate_mask = _morph_close(dark_mask | bright_mask | red_light_mask)
        components = _connected_components(candidate_mask, max_components=self.max_components)
        min_area = max(float(rows * cols) * self.min_area_fraction, 8.0)
        detections: List[Detection] = []
        for comp in components:
            x1, y1, x2, y2, area = comp
            if area < min_area:
                continue
            width = max(x2 - x1, 1)
            height = max(y2 - y1, 1)
            if width > cols * 0.55 or height > rows * 0.65:
                continue
            patch = rgb[y1:y2, x1:x2, :3]
            label = _classify_component(x1, y1, x2, y2, patch, rows, cols)
            score = _component_score(area, rows, cols, label)
            detections.append(
                Detection(
                    BoundingBox((float(x1), float(y1), float(x2), float(y2)), label=label),
                    score=score,
                    metadata={"area": float(area), "fallback": True},
                )
            )
        elapsed = (time.perf_counter() - start) * 1000.0
        return DetectorResult(
            detector_name=self.name,
            input_name=input_name,
            detections=tuple(sorted(detections, key=lambda item: item.score, reverse=True)),
            elapsed_ms=elapsed,
        )


class AuxMapRiskDetector(DetectorAdapter):
    """Risk detector that consumes PerceptionISP auxiliary maps.

    This is not a learned model. It is a deterministic early-warning baseline
    that proves the aux-map path is usable before training a DNN branch.
    """

    name = "aux_map_risk_detector"

    def __init__(self, *, min_area_fraction: float = 0.00035) -> None:
        self.min_area_fraction = float(min_area_fraction)

    def detect(self, image: Any, *, input_name: str = "perception_aux_rgb") -> DetectorResult:
        start = time.perf_counter()
        aux = _as_rgb(image)
        edge = aux[:, :, 0]
        saturation = aux[:, :, 1]
        reliability = aux[:, :, 2]
        rows, cols = edge.shape
        roi = _road_roi(rows, cols) | (saturation > 0.45)
        threshold = max(float(np.percentile(edge[roi], 88.0)) if np.any(roi) else 0.1, 0.12)
        mask = (edge >= threshold) & (reliability > 0.20) & roi
        mask |= (saturation > 0.62)
        mask = _morph_close(mask)
        detections: List[Detection] = []
        min_area = max(float(rows * cols) * self.min_area_fraction, 5.0)
        for x1, y1, x2, y2, area in _connected_components(mask, max_components=96):
            if area < min_area:
                continue
            width = max(x2 - x1, 1)
            height = max(y2 - y1, 1)
            if width > cols * 0.65 or height > rows * 0.75:
                continue
            patch_sat = saturation[y1:y2, x1:x2]
            label = "traffic_light" if float(np.mean(patch_sat)) > 0.35 and y2 < rows * 0.55 else "object"
            detections.append(
                Detection(
                    BoundingBox((float(x1), float(y1), float(x2), float(y2)), label=label),
                    score=float(min(1.0, 0.35 + np.mean(edge[y1:y2, x1:x2]) + 0.3 * np.mean(patch_sat))),
                    metadata={"area": float(area), "uses_aux_maps": True},
                )
            )
        elapsed = (time.perf_counter() - start) * 1000.0
        return DetectorResult(self.name, input_name, tuple(sorted(detections, key=lambda item: item.score, reverse=True)), elapsed)


class UltralyticsYOLODetector(DetectorAdapter):
    """Optional real detector adapter using Ultralytics YOLO."""

    name = "ultralytics_yolo"

    def __init__(self, model_name: str = "yolo11n.pt", *, confidence: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("ultralytics is not installed; install it to run YOLO comparisons") from exc
        self.model_name = str(model_name)
        self.confidence = float(confidence)
        self.model = YOLO(self.model_name)

    def detect(self, image: Any, *, input_name: str = "image") -> DetectorResult:
        start = time.perf_counter()
        rgb = np.clip(_as_rgb(image), 0.0, 1.0)
        uint8 = np.round(rgb * 255.0).astype(np.uint8)
        results = self.model.predict(uint8, conf=self.confidence, verbose=False)
        detections: List[Detection] = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            for coords, score, class_id in zip(xyxy, conf, cls):
                label = str(names.get(int(class_id), int(class_id)))
                detections.append(Detection(BoundingBox(tuple(float(v) for v in coords), label=label), score=float(score)))
        elapsed = (time.perf_counter() - start) * 1000.0
        return DetectorResult(self.name, input_name, tuple(detections), elapsed)


class RGBAuxTorchSmokeDetector(DetectorAdapter):
    """Tiny learned RGB+aux detector used to validate the DNN-facing path.

    The checkpoint predicts objectness and one normalized box. It does not
    learn class labels, so it should not be used for performance claims.
    """

    name = "rgb_aux_torch_smoke_detector"

    def __init__(
        self,
        checkpoint_path: str,
        *,
        confidence: float = 0.10,
        label: str = "object",
        device: str = "auto",
    ) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is not installed; install it to run RGB+aux smoke detector") from exc
        from .aux_dnn import make_aux_smoke_detector_model

        self.torch = torch
        self.checkpoint_path = str(checkpoint_path)
        self.confidence = float(confidence)
        self.label = str(label)
        self.device = _torch_device(torch, device)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        stem_channels = int(checkpoint.get("stem_channels", 16)) if isinstance(checkpoint, Mapping) else 16
        state = checkpoint.get("model_state") if isinstance(checkpoint, Mapping) else checkpoint
        self.model = make_aux_smoke_detector_model(stem_channels=stem_channels)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: Any, *, input_name: str = "perception_rgb_aux_dnn") -> DetectorResult:
        start = time.perf_counter()
        tensor = _as_rgb_aux_chw(image)
        rows, cols = int(tensor.shape[1]), int(tensor.shape[2])
        with self.torch.no_grad():
            x = self.torch.from_numpy(tensor[None, :, :, :]).to(self.device)
            pred = self.model(x).detach().cpu().numpy()[0]
        score = float(1.0 / (1.0 + np.exp(-float(pred[0]))))
        detections: List[Detection] = []
        if score >= self.confidence:
            norm = 1.0 / (1.0 + np.exp(-np.asarray(pred[1:5], dtype=np.float64)))
            x1, x2 = sorted((float(norm[0] * cols), float(norm[2] * cols)))
            y1, y2 = sorted((float(norm[1] * rows), float(norm[3] * rows)))
            if x2 - x1 < 1.0:
                center = 0.5 * (x1 + x2)
                x1, x2 = center - 0.5, center + 0.5
            if y2 - y1 < 1.0:
                center = 0.5 * (y1 + y2)
                y1, y2 = center - 0.5, center + 0.5
            detections.append(
                Detection(
                    BoundingBox(
                        (
                            max(0.0, min(float(cols - 1), x1)),
                            max(0.0, min(float(rows - 1), y1)),
                            max(0.0, min(float(cols), x2)),
                            max(0.0, min(float(rows), y2)),
                        ),
                        label=self.label,
                    ),
                    score=score,
                    metadata={
                        "checkpoint": self.checkpoint_path,
                        "uses_rgb_aux_tensor": True,
                        "class_trained": False,
                    },
                )
            )
        elapsed = (time.perf_counter() - start) * 1000.0
        return DetectorResult(self.name, input_name, tuple(detections), elapsed)


def fuse_rgb_aux_results(
    rgb_result: DetectorResult,
    aux_result: DetectorResult,
    aux_image: Any,
    *,
    input_name: str = "perception_fusion_rgb_aux",
    detector_name: str = "rgb_aux_fusion",
    low_score_threshold: float = 0.42,
    low_support_threshold: float = 0.16,
    score_gain: float = 0.08,
    score_penalty: float = 0.06,
) -> DetectorResult:
    """Fuse RGB detector boxes with PerceptionISP auxiliary-map evidence.

    This is a conservative reference adapter, not a trained detector. It keeps
    RGB detector labels, annotates each detection with aux-map support, and only
    suppresses low-score boxes when aux evidence is also weak.
    """

    start = time.perf_counter()
    aux = _as_rgb(aux_image)
    aux_boxes = tuple(item.box for item in aux_result.detections)
    fused: List[Detection] = []
    for detection in rgb_result.detections:
        support = _aux_support_for_box(detection.box, aux, aux_boxes)
        combined = float(support["combined"])
        rgb_score = float(detection.score)
        suppressed = rgb_score < float(low_score_threshold) and combined < float(low_support_threshold)
        if suppressed:
            continue
        if combined >= 0.35:
            score = rgb_score + float(score_gain) * min((combined - 0.35) / 0.65, 1.0)
        elif rgb_score < 0.55:
            score = rgb_score - float(score_penalty) * min((0.35 - combined) / 0.35, 1.0)
        else:
            score = rgb_score
        metadata = dict(detection.metadata)
        metadata["fusion"] = {
            "rgb_score": rgb_score,
            "aux_support": combined,
            "edge_support": float(support["edge"]),
            "saturation_support": float(support["saturation"]),
            "reliability_support": float(support["reliability"]),
            "aux_box_iou": float(support["aux_box_iou"]),
            "rgb_detector": rgb_result.detector_name,
            "aux_detector": aux_result.detector_name,
        }
        fused.append(
            Detection(
                box=detection.box,
                score=float(np.clip(score, 0.0, 1.0)),
                metadata=metadata,
            )
        )
    elapsed = (time.perf_counter() - start) * 1000.0
    return DetectorResult(
        detector_name=detector_name,
        input_name=input_name,
        detections=tuple(sorted(fused, key=lambda item: item.score, reverse=True)),
        elapsed_ms=float(rgb_result.elapsed_ms + aux_result.elapsed_ms + elapsed),
    )


def detector_from_name(name: str) -> DetectorAdapter:
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"numpy", "fallback", "risk"}:
        return NumpyRiskObjectDetector()
    if normalized in {"aux", "aux_map", "aux_risk"}:
        return AuxMapRiskDetector()
    if normalized in {"yolo", "ultralytics", "yolo11n"}:
        return UltralyticsYOLODetector()
    raise ValueError(f"Unsupported detector {name!r}")


def _as_rgb(image: Any) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("detector image must be HxWx3 or HxW")
    finite = np.nan_to_num(array[:, :, :3], nan=0.0, posinf=1.0, neginf=0.0)
    if float(np.max(finite)) > 1.5:
        finite = finite / 255.0
    return np.clip(finite, 0.0, 1.0)


def _as_rgb_aux_chw(image: Any) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("RGB+aux detector image must be a 6-channel HWC or CHW tensor")
    if array.shape[0] == 6:
        chw = array
    elif array.shape[2] == 6:
        chw = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError("RGB+aux detector image must have six channels")
    return np.nan_to_num(np.clip(chw, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)


def _torch_device(torch: Any, value: str) -> Any:
    requested = str(value or "auto").lower()
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        return torch.device("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _aux_support_for_box(box: BoundingBox, aux: np.ndarray, aux_boxes: Sequence[BoundingBox]) -> Dict[str, float]:
    rows, cols = aux.shape[:2]
    x1, y1, x2, y2 = box.xyxy
    col0 = max(0, min(cols - 1, int(np.floor(x1))))
    col1 = max(col0 + 1, min(cols, int(np.ceil(x2))))
    row0 = max(0, min(rows - 1, int(np.floor(y1))))
    row1 = max(row0 + 1, min(rows, int(np.ceil(y2))))
    patch = aux[row0:row1, col0:col1, :3]
    if patch.size == 0:
        edge = saturation = reliability = 0.0
    else:
        edge_channel = patch[:, :, 0]
        saturation_channel = patch[:, :, 1]
        reliability_channel = patch[:, :, 2]
        edge = 0.65 * float(np.mean(edge_channel)) + 0.35 * float(np.percentile(edge_channel, 90.0))
        saturation = float(np.percentile(saturation_channel, 90.0))
        reliability = float(np.mean(reliability_channel))
    aux_iou = max((_box_iou(box, aux_box) for aux_box in aux_boxes), default=0.0)
    aux_overlap = min(float(aux_iou) / 0.35, 1.0)
    combined = 0.48 * edge + 0.16 * saturation + 0.26 * reliability + 0.10 * aux_overlap
    return {
        "edge": float(np.clip(edge, 0.0, 1.0)),
        "saturation": float(np.clip(saturation, 0.0, 1.0)),
        "reliability": float(np.clip(reliability, 0.0, 1.0)),
        "aux_box_iou": float(np.clip(aux_iou, 0.0, 1.0)),
        "combined": float(np.clip(combined, 0.0, 1.0)),
    }


def _box_iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = a.area + b.area - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _road_roi(rows: int, cols: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, int(rows), dtype=np.float64)[:, None]
    return np.repeat(y > 0.38, int(cols), axis=1)


def _traffic_light_color_mask(rgb: np.ndarray) -> np.ndarray:
    rows = rgb.shape[0]
    y = np.linspace(0.0, 1.0, rows, dtype=np.float64)[:, None]
    top_half = np.repeat(y < 0.58, rgb.shape[1], axis=1)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    return (red > 0.32) & (red > green * 1.18) & (red > blue * 1.18) & top_half


def _morph_close(mask: np.ndarray) -> np.ndarray:
    dilated = _dilate(_dilate(mask))
    return _erode(dilated)


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    out = np.ones_like(mask, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def _connected_components(mask: np.ndarray, *, max_components: int) -> List[Tuple[int, int, int, int, int]]:
    visited = np.zeros(mask.shape, dtype=bool)
    rows, cols = mask.shape
    components: List[Tuple[int, int, int, int, int]] = []
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        min_y = max_y = start_y
        min_x = max_x = start_x
        area = 0
        while queue:
            y, x = queue.popleft()
            area += 1
            min_y, max_y = min(min_y, y), max(max_y, y)
            min_x, max_x = min(min_x, x), max(max_x, x)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        components.append((min_x, min_y, max_x + 1, max_y + 1, area))
        if len(components) >= int(max_components):
            break
    return components


def _classify_component(x1: int, y1: int, x2: int, y2: int, patch: np.ndarray, rows: int, cols: int) -> str:
    mean_rgb = np.mean(patch.reshape(-1, patch.shape[-1]), axis=0)
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    aspect = height / max(width, 1)
    if y2 < rows * 0.58 and mean_rgb[0] > mean_rgb[1] * 1.12 and mean_rgb[0] > mean_rgb[2] * 1.12:
        return "traffic_light"
    if aspect > 1.8 and width < cols * 0.14:
        return "person"
    return "car"


def _component_score(area: int, rows: int, cols: int, label: str) -> float:
    base = min(1.0, 0.35 + float(area) / max(rows * cols * 0.02, 1.0))
    if label == "traffic_light":
        base = min(1.0, base + 0.12)
    return float(base)
