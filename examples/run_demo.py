from __future__ import annotations

from pathlib import Path

from perception_isp import PerceptionISPPipeline
from perception_isp.core.synthetic import make_synthetic_raw


def main() -> None:
    raw = make_synthetic_raw(width=320, height=180, cfa_pattern="RGGB")
    result = PerceptionISPPipeline().run(raw)
    output = Path("reports/perception_isp_demo")
    output.mkdir(parents=True, exist_ok=True)
    print("accurate", result.accurate.tensor.shape, result.accurate.channels)
    print("fast", result.fast.tensor.shape, result.fast.channels)
    print("edge packets", len(result.fast.edge_packets))
    print("health", result.health)


if __name__ == "__main__":
    main()
