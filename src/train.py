"""Train the WOE scorecard (interpretable baseline) and the GBM challenger; register the bundle.

Out-of-time split by disbursement date (see configs/config.yaml): train < valid < test.
The champion is the GBM only if it beats the scorecard on VALIDATION AUC by >= 0.01; otherwise the
simpler, explainable scorecard wins. The test set is never used for any fitting or selection.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from . import metrics as M
from .config import layer_dir, load_config
from .features import FEATURES
from .gbm import GBMModel
from .registry import save_bundle
from .scorecard import Scorecard
from .scoring import predict_frame
from .utils import get_logger

log = get_logger("train")
MODEL_NAME = "credit_scoring"
CHAMPION_MARGIN = 0.01


def load_features(cfg) -> pd.DataFrame:
    path = layer_dir(cfg, "features") / "features.csv.gz"
    f = pd.read_csv(path, parse_dates=["snapshot_date"])
    return f


def split_frames(f: pd.DataFrame, train_end: str, valid_end: str):
    tr = f[f["snapshot_date"] <= train_end]
    va = f[(f["snapshot_date"] > train_end) & (f["snapshot_date"] <= valid_end)]
    te = f[f["snapshot_date"] > valid_end]
    return tr, va, te


def train(cfg: dict | None = None, feats: pd.DataFrame | None = None, train_end: str | None = None,
          valid_end: str | None = None, set_latest: bool = True, note: str = "initial training") -> tuple[str, dict]:
    cfg = cfg or load_config()
    f = feats if feats is not None else load_features(cfg)
    train_end = train_end or cfg["split"]["train_end"]
    valid_end = valid_end or cfg["split"]["valid_end"]
    tr, va, te = split_frames(f, train_end, valid_end)
    log.info("train=%d valid=%d test=%d", len(tr), len(va), len(te))

    sp = cfg["scorecard"]
    sc = Scorecard(n_bins=sp["n_bins"], min_iv=sp["min_iv"], max_corr=sp["max_corr"], base_score=sp["base_score"],
                   base_odds=sp["base_odds"], pdo=sp["pdo"]).fit(tr, tr["default"], FEATURES)
    gbm = GBMModel(seed=cfg["seed"]).fit(tr, tr["default"], va, va["default"])
    auc_sc = M.auc(va["default"], sc.predict_proba(va))
    auc_gbm = M.auc(va["default"], gbm.predict_proba(va))
    champion = "gbm" if auc_gbm >= auc_sc + CHAMPION_MARGIN else "scorecard"
    log.info("valid AUC scorecard=%.4f gbm=%.4f (%s) -> champion=%s", auc_sc, auc_gbm, gbm.engine_, champion)

    bundle = {"scorecard": sc, "gbm": gbm, "champion": champion, "features": FEATURES}
    pv = predict_frame(bundle, va)
    metadata = {
        "note": note,
        "champion": champion,
        "gbm_engine": gbm.engine_,
        "split": {"train_end": train_end, "valid_end": valid_end},
        "n": {"train": len(tr), "valid": len(va), "test": len(te)},
        "scorecard_features": sc.features,
        "metrics": {"valid": {"scorecard": M.summary(va["default"], pv["pd_scorecard"]),
                              "gbm": M.summary(va["default"], pv["pd_gbm"])}},
        "data_fingerprint": hashlib.sha256(
            pd.util.hash_pandas_object(f[["loan_id", "default"]], index=False).values.tobytes()).hexdigest()[:12],
        "policy": None,
    }
    version = save_bundle(cfg, MODEL_NAME, bundle, metadata, set_latest=set_latest)
    return version, metadata


if __name__ == "__main__":
    train()
