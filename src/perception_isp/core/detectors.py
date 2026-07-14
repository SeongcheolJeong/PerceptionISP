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

from perception_isp.core.aux_dnn import RGB_AUX_CHANNELS, apply_channel_mask
from perception_isp.core.task_types import BoundingBox, Detection, DetectorResult


class DetectorAdapter:
    name = "detector"

    def detect(self, image: Any, *, input_name: str = "image") -> DetectorResult:
        raise NotImplementedError


class LabelMapDetector(DetectorAdapter):
    """Adapter that remaps detector output labels without changing boxes or scores."""

    def __init__(self, detector: DetectorAdapter, label_map: Mapping[str, str]) -> None:
        self.detector = detector
        self.label_map = {str(key): str(value) for key, value in dict(label_map).items()}
        self.name = getattr(detector, "name", "detector")
        for attr in ("channels", "tensor_key", "input_channels", "channel_mode"):
            if hasattr(detector, attr):
                setattr(self, attr, getattr(detector, attr))

    def detect(self, image: Any, *, input_name: str = "image") -> DetectorResult:
        result = self.detector.detect(image, input_name=input_name)
        if not self.label_map:
            return result
        detections = []
        for detection in result.detections:
            original = detection.box.label
            mapped = self.label_map.get(original, original)
            metadata = dict(detection.metadata)
            if mapped != original:
                metadata["original_label"] = original
                metadata["label_mapped"] = True
            detections.append(
                Detection(
                    BoundingBox(detection.box.xyxy, label=mapped),
                    score=detection.score,
                    metadata=metadata,
                )
            )
        return DetectorResult(result.detector_name, result.input_name, tuple(detections), result.elapsed_ms)


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
        from perception_isp.core.aux_dnn import make_aux_smoke_detector_model

        self.torch = torch
        self.checkpoint_path = str(checkpoint_path)
        self.confidence = float(confidence)
        self.label = str(label)
        self.device = _torch_device(torch, device)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        stem_channels = int(checkpoint.get("stem_channels", 16)) if isinstance(checkpoint, Mapping) else 16
        self.channels = tuple(str(value) for value in checkpoint.get("channels", RGB_AUX_CHANNELS)) if isinstance(checkpoint, Mapping) else tuple(RGB_AUX_CHANNELS)
        self.tensor_key = str(checkpoint.get("tensor_key", "rgb_aux_chw")) if isinstance(checkpoint, Mapping) else "rgb_aux_chw"
        self.input_channels = int(checkpoint.get("input_channels", len(self.channels))) if isinstance(checkpoint, Mapping) else len(self.channels)
        state = checkpoint.get("model_state") if isinstance(checkpoint, Mapping) else checkpoint
        self.model = make_aux_smoke_detector_model(stem_channels=stem_channels, in_channels=self.input_channels)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: Any, *, input_name: str = "perception_rgb_aux_dnn") -> DetectorResult:
        start = time.perf_counter()
        tensor = _as_rgb_aux_chw(image, expected_channels=self.input_channels)
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
                        "tensor_key": self.tensor_key,
                        "input_channels": self.input_channels,
                    },
                )
            )
        elapsed = (time.perf_counter() - start) * 1000.0
        return DetectorResult(self.name, input_name, tuple(detections), elapsed)


class RGBAuxTorchDenseDetector(DetectorAdapter):
    """Compact class-aware RGB+aux detector trained on exported tensors."""

    name = "rgb_aux_torch_dense_detector"

    def __init__(
        self,
        checkpoint_path: str,
        *,
        confidence: float = 0.30,
        nms_iou: float = 0.50,
        max_detections: int = 100,
        device: str = "auto",
    ) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is not installed; install it to run RGB+aux dense detector") from exc
        from perception_isp.core.aux_dnn import make_aux_dense_detector_model

        self.torch = torch
        self.checkpoint_path = str(checkpoint_path)
        self.confidence = float(confidence)
        self.nms_iou = float(nms_iou)
        self.max_detections = int(max_detections)
        self.device = _torch_device(torch, device)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("dense detector checkpoint must be a mapping")
        self.box_encoding = str(checkpoint.get("box_encoding", "xyxy"))
        self.class_names = tuple(str(value) for value in checkpoint.get("class_names", ("object",))) or ("object",)
        self.channels = tuple(str(value) for value in checkpoint.get("channels", RGB_AUX_CHANNELS)) or tuple(RGB_AUX_CHANNELS)
        self.tensor_key = str(checkpoint.get("tensor_key", "rgb_aux_chw"))
        self.input_channels = int(checkpoint.get("input_channels", len(self.channels)))
        self.channel_mode = str(checkpoint.get("channel_mode", "rgb_aux"))
        self.channel_mask = tuple(float(value) for value in checkpoint.get("channel_mask", (1.0,) * self.input_channels))
        grid_size = tuple(int(value) for value in checkpoint.get("grid_size", (15, 20)))
        base_channels = int(checkpoint.get("base_channels", 24))
        self.model_architecture = str(checkpoint.get("model_architecture", "early_fusion"))
        self.model = make_aux_dense_detector_model(
            num_classes=len(self.class_names),
            grid_size=(int(grid_size[0]), int(grid_size[1])),
            base_channels=base_channels,
            in_channels=self.input_channels,
            architecture=self.model_architecture,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image: Any, *, input_name: str = "perception_rgb_aux_dnn") -> DetectorResult:
        start = time.perf_counter()
        tensor = _as_rgb_aux_chw(image, expected_channels=self.input_channels)
        rows, cols = int(tensor.shape[1]), int(tensor.shape[2])
        with self.torch.no_grad():
            pred = self._predict_tensor(tensor)
            objectness = self.torch.sigmoid(pred[0]).numpy()
            boxes = self.torch.sigmoid(pred[1:5]).numpy()
            class_scores = self.torch.softmax(pred[5:], dim=0).numpy()

        detections: List[Detection] = []
        grid_rows, grid_cols = objectness.shape
        for row in range(int(grid_rows)):
            for col in range(int(grid_cols)):
                class_index = int(np.argmax(class_scores[:, row, col]))
                score = float(objectness[row, col] * class_scores[class_index, row, col])
                if score < self.confidence:
                    continue
                x1, y1, x2, y2 = _decode_dense_box(
                    boxes[:, row, col],
                    row=row,
                    col=col,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    image_rows=rows,
                    image_cols=cols,
                    encoding=self.box_encoding,
                )
                if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                    continue
                detections.append(
                    Detection(
                        BoundingBox(
                            (
                                max(0.0, min(float(cols - 1), x1)),
                                max(0.0, min(float(rows - 1), y1)),
                                max(0.0, min(float(cols), x2)),
                                max(0.0, min(float(rows), y2)),
                            ),
                            label=self.class_names[class_index],
                        ),
                        score=score,
                        metadata={
                            "checkpoint": self.checkpoint_path,
                            "uses_rgb_aux_tensor": True,
                            "class_trained": True,
                            "channel_mode": self.channel_mode,
                            "model_architecture": self.model_architecture,
                            "grid_cell": [int(row), int(col)],
                            "objectness": float(objectness[row, col]),
                            "class_probability": float(class_scores[class_index, row, col]),
                            "tensor_key": self.tensor_key,
                            "input_channels": self.input_channels,
                        },
                    )
                )
        pruned = _nms_detections(detections, iou_threshold=self.nms_iou, max_detections=self.max_detections)
        elapsed = (time.perf_counter() - start) * 1000.0
        return DetectorResult(self.name, input_name, tuple(pruned), elapsed)

    def _predict_tensor(self, tensor: np.ndarray) -> Any:
        x = apply_channel_mask(self.torch.from_numpy(tensor[None, :, :, :]).to(self.device), self.channel_mask)
        try:
            return self.model(x)[0].detach().cpu()
        except RuntimeError as exc:
            message = str(exc).lower()
            if getattr(self.device, "type", "") == "mps" and "adaptive" in message and "mps" in message:
                self.device = self.torch.device("cpu")
                self.model.to(self.device)
                x = apply_channel_mask(self.torch.from_numpy(tensor[None, :, :, :]).to(self.device), self.channel_mask)
                return self.model(x)[0].detach().cpu()
            raise


def rgb_aux_detector_from_checkpoint(
    checkpoint_path: str,
    *,
    confidence: Optional[float] = None,
    nms_iou: Optional[float] = None,
    max_detections: Optional[int] = None,
    device: str = "auto",
) -> DetectorAdapter:
    """Load the right RGB+aux detector adapter from checkpoint metadata."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is not installed; install it to run RGB+aux detector") from exc
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model_type = str(checkpoint.get("model_type", "")) if isinstance(checkpoint, Mapping) else ""
    if model_type == "rgb_aux_dense_detector_v1":
        kwargs: Dict[str, Any] = {
            "confidence": 0.30 if confidence is None else float(confidence),
            "device": device,
        }
        if nms_iou is not None:
            kwargs["nms_iou"] = float(nms_iou)
        if max_detections is not None:
            kwargs["max_detections"] = int(max_detections)
        return RGBAuxTorchDenseDetector(str(checkpoint_path), **kwargs)
    return RGBAuxTorchSmokeDetector(str(checkpoint_path), confidence=0.10 if confidence is None else float(confidence), device=device)


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
    refine_boxes: bool = False,
    refine_max_shift: float = 0.08,
    refine_max_shift_px: float = 12.0,
    refine_min_edge: float = 0.18,
    refine_min_gain: float = 0.03,
    refine_allow_shrink: bool = False,
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
        box = detection.box
        refinement: Dict[str, Any] = {"enabled": bool(refine_boxes), "changed": False}
        if refine_boxes:
            box, refinement = _refine_box_with_aux_edges(
                detection.box,
                aux,
                max_shift_fraction=float(refine_max_shift),
                max_shift_px=float(refine_max_shift_px),
                min_edge=float(refine_min_edge),
                min_gain=float(refine_min_gain),
                allow_shrink=bool(refine_allow_shrink),
            )
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
            "box_refinement": refinement,
        }
        fused.append(
            Detection(
                box=box,
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


def _as_rgb_aux_chw(image: Any, *, expected_channels: int = 6) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"RGB+aux detector image must be a {expected_channels}-channel HWC or CHW tensor")
    channels = int(expected_channels)
    if array.shape[0] == channels:
        chw = array
    elif array.shape[2] == channels:
        chw = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"RGB+aux detector image must have {channels} channels")
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


def _refine_box_with_aux_edges(
    box: BoundingBox,
    aux: np.ndarray,
    *,
    max_shift_fraction: float,
    max_shift_px: float,
    min_edge: float,
    min_gain: float,
    allow_shrink: bool,
) -> Tuple[BoundingBox, Dict[str, Any]]:
    rows, cols = aux.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box.xyxy]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    x_shift = int(max(1.0, min(float(max_shift_px), width * max(float(max_shift_fraction), 0.0))))
    y_shift = int(max(1.0, min(float(max_shift_px), height * max(float(max_shift_fraction), 0.0))))
    edge = np.asarray(aux[:, :, 0], dtype=np.float64)
    reliability = np.asarray(aux[:, :, 2], dtype=np.float64) if aux.shape[2] >= 3 else np.ones_like(edge)
    evidence = np.clip(edge, 0.0, 1.0) * np.clip(0.5 + 0.5 * reliability, 0.0, 1.0)

    left, left_meta = _snap_vertical_edge(evidence, x1, y1, y2, x_shift, min_edge=min_edge, min_gain=min_gain)
    right, right_meta = _snap_vertical_edge(evidence, x2, y1, y2, x_shift, min_edge=min_edge, min_gain=min_gain)
    top, top_meta = _snap_horizontal_edge(evidence, y1, x1, x2, y_shift, min_edge=min_edge, min_gain=min_gain)
    bottom, bottom_meta = _snap_horizontal_edge(evidence, y2, x1, x2, y_shift, min_edge=min_edge, min_gain=min_gain)

    refined = [
        float(np.clip(left, 0.0, max(float(cols - 1), 0.0))),
        float(np.clip(top, 0.0, max(float(rows - 1), 0.0))),
        float(np.clip(right, 1.0, float(cols))),
        float(np.clip(bottom, 1.0, float(rows))),
    ]
    if not allow_shrink:
        refined = [
            min(refined[0], x1),
            min(refined[1], y1),
            max(refined[2], x2),
            max(refined[3], y2),
        ]
    refined = _constrain_refined_box(tuple(refined), box.xyxy, image_shape=(rows, cols))
    changed = any(abs(float(a) - float(b)) > 0.25 for a, b in zip(refined, box.xyxy))
    metadata = {
        "enabled": True,
        "changed": bool(changed),
        "original_xyxy": [float(value) for value in box.xyxy],
        "refined_xyxy": [float(value) for value in refined],
        "shifts": [float(refined[index] - box.xyxy[index]) for index in range(4)],
        "allow_shrink": bool(allow_shrink),
        "left": left_meta,
        "right": right_meta,
        "top": top_meta,
        "bottom": bottom_meta,
    }
    if not changed:
        return box, metadata
    return BoundingBox(tuple(refined), label=box.label), metadata


def _snap_vertical_edge(
    evidence: np.ndarray,
    boundary_x: float,
    y1: float,
    y2: float,
    max_shift: int,
    *,
    min_edge: float,
    min_gain: float,
) -> Tuple[float, Dict[str, Any]]:
    rows, cols = evidence.shape
    row0, row1 = _trimmed_span(y1, y2, limit=rows)
    col_center = int(np.clip(round(float(boundary_x)), 0, max(cols - 1, 0)))
    col0 = max(0, col_center - int(max_shift))
    col1 = min(cols, col_center + int(max_shift) + 1)
    if row1 <= row0 or col1 <= col0:
        return float(boundary_x), {"snapped": False, "reason": "empty_window"}
    profile = np.mean(evidence[row0:row1, col0:col1], axis=0)
    current_index = int(np.clip(col_center - col0, 0, len(profile) - 1))
    peak_index = int(np.argmax(profile))
    current = float(profile[current_index])
    peak = float(profile[peak_index])
    snapped = bool(peak >= float(min_edge) and peak >= current + float(min_gain))
    snapped_x = float(col0 + peak_index) if snapped else float(boundary_x)
    return snapped_x, {
        "snapped": snapped,
        "current": current,
        "peak": peak,
        "shift": float(snapped_x - float(boundary_x)),
    }


def _snap_horizontal_edge(
    evidence: np.ndarray,
    boundary_y: float,
    x1: float,
    x2: float,
    max_shift: int,
    *,
    min_edge: float,
    min_gain: float,
) -> Tuple[float, Dict[str, Any]]:
    rows, cols = evidence.shape
    col0, col1 = _trimmed_span(x1, x2, limit=cols)
    row_center = int(np.clip(round(float(boundary_y)), 0, max(rows - 1, 0)))
    row0 = max(0, row_center - int(max_shift))
    row1 = min(rows, row_center + int(max_shift) + 1)
    if row1 <= row0 or col1 <= col0:
        return float(boundary_y), {"snapped": False, "reason": "empty_window"}
    profile = np.mean(evidence[row0:row1, col0:col1], axis=1)
    current_index = int(np.clip(row_center - row0, 0, len(profile) - 1))
    peak_index = int(np.argmax(profile))
    current = float(profile[current_index])
    peak = float(profile[peak_index])
    snapped = bool(peak >= float(min_edge) and peak >= current + float(min_gain))
    snapped_y = float(row0 + peak_index) if snapped else float(boundary_y)
    return snapped_y, {
        "snapped": snapped,
        "current": current,
        "peak": peak,
        "shift": float(snapped_y - float(boundary_y)),
    }


def _trimmed_span(start: float, stop: float, *, limit: int) -> Tuple[int, int]:
    lo = float(min(start, stop))
    hi = float(max(start, stop))
    size = max(hi - lo, 1.0)
    trim = 0.12 * size if size >= 12.0 else 0.0
    first = int(np.clip(np.floor(lo + trim), 0, max(limit - 1, 0)))
    last = int(np.clip(np.ceil(hi - trim), first + 1, limit))
    return first, last


def _constrain_refined_box(
    refined_xyxy: Tuple[float, float, float, float],
    original_xyxy: Tuple[float, float, float, float],
    *,
    image_shape: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    rows, cols = image_shape
    ox1, oy1, ox2, oy2 = original_xyxy
    x1, y1, x2, y2 = refined_xyxy
    original_width = max(float(ox2 - ox1), 1.0)
    original_height = max(float(oy2 - oy1), 1.0)
    min_width = max(2.0, 0.75 * original_width)
    max_width = min(float(cols), 1.25 * original_width)
    min_height = max(2.0, 0.75 * original_height)
    max_height = min(float(rows), 1.25 * original_height)
    x1, x2 = _constrain_axis(x1, x2, ox1, ox2, min_size=min_width, max_size=max_width, limit=float(cols))
    y1, y2 = _constrain_axis(y1, y2, oy1, oy2, min_size=min_height, max_size=max_height, limit=float(rows))
    return (float(x1), float(y1), float(x2), float(y2))


def _constrain_axis(
    start: float,
    stop: float,
    original_start: float,
    original_stop: float,
    *,
    min_size: float,
    max_size: float,
    limit: float,
) -> Tuple[float, float]:
    center = 0.5 * (float(start) + float(stop))
    original_center = 0.5 * (float(original_start) + float(original_stop))
    size = float(stop) - float(start)
    if size < float(min_size) or size > float(max_size):
        size = float(np.clip(size, float(min_size), float(max_size)))
        center = original_center
    half = 0.5 * size
    lo = float(np.clip(center - half, 0.0, max(limit - size, 0.0)))
    hi = float(np.clip(lo + size, lo + 1.0, limit))
    return lo, hi


def _box_iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = a.area + b.area - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _nms_detections(
    detections: Sequence[Detection],
    *,
    iou_threshold: float,
    max_detections: int,
) -> Tuple[Detection, ...]:
    kept: List[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if len(kept) >= int(max_detections):
            break
        suppress = False
        for selected in kept:
            if selected.box.label == detection.box.label and _box_iou(selected.box, detection.box) > float(iou_threshold):
                suppress = True
                break
        if not suppress:
            kept.append(detection)
    return tuple(kept)


def _decode_dense_box(
    values: Any,
    *,
    row: int,
    col: int,
    grid_rows: int,
    grid_cols: int,
    image_rows: int,
    image_cols: int,
    encoding: str,
) -> Tuple[float, float, float, float]:
    a, b, c, d = [float(value) for value in values]
    if str(encoding) == "cell_center_size":
        center_x = (float(col) + a) / max(float(grid_cols), 1.0)
        center_y = (float(row) + b) / max(float(grid_rows), 1.0)
        width = c
        height = d
        x1n, x2n = center_x - 0.5 * width, center_x + 0.5 * width
        y1n, y2n = center_y - 0.5 * height, center_y + 0.5 * height
    else:
        x1n, y1n, x2n, y2n = a, b, c, d
        x1n, x2n = sorted((x1n, x2n))
        y1n, y2n = sorted((y1n, y2n))
    return (
        x1n * float(image_cols),
        y1n * float(image_rows),
        x2n * float(image_cols),
        y2n * float(image_rows),
    )


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
