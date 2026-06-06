"""Command line entry point for the Perception ISP reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .camerae2e_bridge import camerae2e_or_synthetic_raw
from .pipeline import PerceptionISPPipeline
from .types import PerceptionISPConfig, json_ready


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run the software Perception ISP reference pipeline.")
    parser.add_argument("--output-dir", default="reports/perception_isp_demo", help="Directory for npz/json/ppm outputs.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--cfa", default="auto", help="CFA pattern: auto, RGGB, GRBG, BGGR, GBRG, RCCB, RGBIR, MONO.")
    parser.add_argument("--camerae2e", action="store_true", help="Try CameraE2E/pyisetcam before synthetic fallback.")
    parser.add_argument("--scene", default="uniform ee", help="CameraE2E scene name when --camerae2e is used.")
    parser.add_argument("--dewarp", action="store_true", help="Enable accurate-path radial dewarp prototype.")
    parser.add_argument("--edge-packets", type=int, default=512)
    parser.add_argument("--demosaic-method", default="edge_aware", choices=["edge_aware", "bilinear"], help="Bayer demosaic method.")
    args = parser.parse_args(argv)

    raw = camerae2e_or_synthetic_raw(
        use_camerae2e=bool(args.camerae2e),
        scene_name=str(args.scene),
        width=int(args.width),
        height=int(args.height),
        cfa_pattern=str(args.cfa),
    )
    config = PerceptionISPConfig(
        accurate_enable_dewarp=bool(args.dewarp),
        max_edge_packets=int(args.edge_packets),
        demosaic_method=str(args.demosaic_method),
    )
    pipeline = PerceptionISPPipeline(config=config)
    result = pipeline.run(raw)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "perception_isp_outputs.npz",
        vision_rgb=result.vision_rgb,
        accurate_tensor=result.accurate.tensor,
        fast_tensor=result.fast.tensor,
        raw_normalized=result.raw_normalized,
        **{f"map_{name}": value for name, value in result.maps.items() if np.asarray(value).ndim <= 2},
    )
    summary = _summary(result)
    (output_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n")
    _write_ppm(output_dir / "vision_rgb.ppm", result.vision_rgb)
    if result.human_rgb is not None:
        _write_ppm(output_dir / "human_rgb.ppm", result.human_rgb)
    print(json.dumps(json_ready(summary), indent=2))
    return 0


def _summary(result: Any) -> Mapping[str, Any]:
    return {
        "accurate_tensor_shape": list(result.accurate.tensor.shape),
        "accurate_channels": list(result.accurate.channels),
        "fast_tensor_shape": list(result.fast.tensor.shape),
        "fast_channels": list(result.fast.channels),
        "fast_roi": list(result.fast.roi),
        "edge_packet_count": len(result.fast.edge_packets),
        "estimated_fast_latency_us": result.fast.estimated_latency_us,
        "metadata": result.metadata,
        "health": result.health,
        "map_names": sorted(result.maps.keys()),
    }


def _write_ppm(path: Path, image: Any) -> None:
    rgb = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("PPM image must be HxWx3")
    payload = np.round(rgb[:, :, :3] * 255.0).astype(np.uint8)
    header = f"P6\n{payload.shape[1]} {payload.shape[0]}\n255\n".encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(payload.tobytes())


if __name__ == "__main__":
    raise SystemExit(main())
