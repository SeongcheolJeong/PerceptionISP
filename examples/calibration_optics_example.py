from __future__ import annotations

from perception_isp.core.paths import output_root
from perception_isp.evaluation.example_suite import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--case",
                "calibration",
                "--case",
                "cfa-optics",
                "--output-dir",
                str(output_root() / "example_calibration_optics"),
            ]
        )
    )
