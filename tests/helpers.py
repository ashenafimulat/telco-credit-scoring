"""Shared test helpers: small cached dataset + isolated temp config."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from src.config import load_config
from src.etl import clean_tables
from src.generate_data import generate

_CACHE: dict = {}


def small_tables(n: int = 300, seed: int = 7):
    key = (n, seed)
    if key not in _CACHE:
        cfg = load_config()
        raw = generate(cfg, n_customers=n, seed=seed)
        clean, _ = clean_tables(raw)
        _CACHE[key] = (cfg, raw, clean)
    cfg, raw, clean = _CACHE[key]
    return cfg, {k: v.copy() for k, v in raw.items()}, {k: v.copy() for k, v in clean.items()}


def tmp_cfg():
    cfg = copy.deepcopy(load_config())
    root = Path(tempfile.mkdtemp(prefix="tcs_"))
    cfg["paths"] = {k: str(root / k) for k in cfg["paths"]}
    return cfg
