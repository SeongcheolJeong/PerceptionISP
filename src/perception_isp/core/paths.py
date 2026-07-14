"""Environment-aware workspace paths used by public workflows."""

from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    """Return the configured dataset root, defaulting to ``./data``."""

    return _env_path("PERCEPTION_ISP_DATA", "data")


def output_root() -> Path:
    """Return the configured artifact root, defaulting to ``./reports``."""

    return _env_path("PERCEPTION_ISP_OUTPUT", "reports")


def camerae2e_source() -> Path:
    """Return CameraE2E's import directory or a non-existent sentinel path."""

    configured = os.environ.get("CAMERAE2E_ROOT")
    if not configured:
        return Path(".perception-isp/camerae2e-not-configured")
    root = Path(configured).expanduser()
    return root if root.name == "src" else root / "src"


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()
