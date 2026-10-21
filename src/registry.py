"""Tiny local model registry (versioned bundles + metadata) with optional MLflow logging.

    models/registry/<name>/v1/{bundle.joblib, metadata.json}
    models/registry/<name>/latest.json  -> {"version": "v1"}

MLflow is used when installed (params, metrics, metadata artifact); otherwise this registry is the
source of truth. Either way, serving loads from here so a Docker image is self-contained.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib

from .config import ROOT, layer_dir
from .utils import get_logger

log = get_logger("registry")


def _dir(cfg, name) -> Path:
    d = layer_dir(cfg, "registry") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def next_version(cfg, name) -> str:
    nums = [int(p.name[1:]) for p in _dir(cfg, name).glob("v*") if p.name[1:].isdigit()]
    return f"v{max(nums, default=0) + 1}"


def save_bundle(cfg, name, bundle: dict, metadata: dict, set_latest: bool = True) -> str:
    version = next_version(cfg, name)
    d = _dir(cfg, name) / version
    d.mkdir(parents=True)
    joblib.dump(bundle, d / "bundle.joblib")
    metadata = {**metadata, "name": name, "version": version, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    if set_latest:
        set_latest_version(cfg, name, version)
    _log_mlflow(name, version, metadata)
    log.info("registered %s/%s (latest=%s)", name, version, set_latest)
    return version


def set_latest_version(cfg, name, version) -> None:
    (_dir(cfg, name) / "latest.json").write_text(json.dumps({"version": version}))


def resolve(cfg, name, version="latest") -> str:
    if version == "latest":
        p = _dir(cfg, name) / "latest.json"
        if not p.exists():
            raise FileNotFoundError(f"No registered model '{name}'. Run `make train` first.")
        return json.loads(p.read_text())["version"]
    return version


def load_bundle(cfg, name, version="latest") -> tuple[dict, dict]:
    v = resolve(cfg, name, version)
    d = _dir(cfg, name) / v
    return joblib.load(d / "bundle.joblib"), json.loads((d / "metadata.json").read_text())


def load_metadata(cfg, name, version="latest") -> dict:
    v = resolve(cfg, name, version)
    return json.loads((_dir(cfg, name) / v / "metadata.json").read_text())


def update_metadata(cfg, name, version, patch: dict) -> dict:
    v = resolve(cfg, name, version)
    p = _dir(cfg, name) / v / "metadata.json"
    m = {**json.loads(p.read_text()), **patch}
    p.write_text(json.dumps(m, indent=2, default=str))
    return m


def _flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield f"{prefix}{k}", float(v)


def _log_mlflow(name, version, metadata) -> None:
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
        mlflow.set_experiment(name)
        with mlflow.start_run(run_name=version):
            mlflow.log_params({"champion": metadata.get("champion"), "gbm_engine": metadata.get("gbm_engine"),
                               "train_end": metadata["split"]["train_end"], "valid_end": metadata["split"]["valid_end"]})
            for k, v in _flat(metadata.get("metrics", {})):
                mlflow.log_metric(k.replace("/", "_"), v)
            mlflow.log_dict(metadata, "metadata.json")
    except Exception as e:  # tracking must never break training
        log.warning("MLflow logging skipped: %s", e)
