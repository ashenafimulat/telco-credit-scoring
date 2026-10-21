"""Config + path helpers."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def layer_dir(cfg: dict, layer: str) -> Path:
    d = ROOT / cfg["paths"][layer]
    d.mkdir(parents=True, exist_ok=True)
    return d
