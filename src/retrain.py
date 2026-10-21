"""Scheduled retraining: monitor -> (if triggered) retrain on all matured loans -> champion/challenger.

    python -m src.retrain            # retrain only if monitoring recommends it
    python -m src.retrain --force    # always retrain (what the monthly GitHub Action uses)

The candidate is trained on everything except the last two vintages (which become its validation
window) and is compared with the current model on that same window. It is promoted to `latest`
unless its AUC is materially worse (> 0.005 lower); fresher data keeps bins, coefficients and calibration current. Everything is logged to
reports/retrain_log.json.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import metrics as M
from .config import layer_dir, load_config
from .monitoring import monitor
from .registry import load_bundle, set_latest_version, update_metadata
from .scoring import predict_frame
from .strategy import make_policy
from .train import MODEL_NAME, load_features, train
from .utils import get_logger

log = get_logger("retrain")
PROMOTE_TOLERANCE = 0.005  # refreshed model is promoted unless it is materially worse on the shared holdout


def run(cfg: dict | None = None, force: bool = False) -> dict:
    cfg = cfg or load_config()
    reports = layer_dir(cfg, "reports")
    mon = monitor(cfg)
    entry = {"trigger": mon["reason"] if mon["retrain_recommended"] else ("forced" if force else None),
             "current_version": mon["model_version"], "action": "none"}
    if not (mon["retrain_recommended"] or force):
        log.info("monitoring is healthy; no retrain")
        (reports / "retrain_log.json").write_text(json.dumps(entry, indent=2))
        return entry

    f = load_features(cfg)
    last_month = f["snapshot_date"].max().to_period("M")
    train_end = str((last_month - 2).end_time.normalize().date())
    valid_end = str(f["snapshot_date"].max().date())
    version, meta = train(cfg, f, train_end, valid_end, set_latest=False,
                          note=f"retrain: {entry['trigger']}")
    hold = f[f["snapshot_date"] > train_end]
    cur_bundle, cur_meta = load_bundle(cfg, MODEL_NAME, "latest")
    new_bundle, _ = load_bundle(cfg, MODEL_NAME, version)
    cur = predict_frame(cur_bundle, hold)
    new = predict_frame(new_bundle, hold)
    auc_cur, auc_new = M.auc(hold["default"], cur["pd"]), M.auc(hold["default"], new["pd"])
    entry.update({"candidate_version": version, "holdout_window": f"{train_end} < snapshot <= {valid_end}",
                  "holdout_auc_current": auc_cur, "holdout_auc_candidate": auc_new})

    hv = pd.concat([hold[["default", "fee_etb", "loan_amount"]], new[["pd"]]], axis=1)
    pred_pd = hv["pd"]
    grid = pd.Series(pred_pd).quantile([0.05, 0.995]).round(2).tolist()
    step = cfg["strategy"]["cutoff_grid_step"]
    import numpy as np
    pol = make_policy(hv, np.arange(grid[0], grid[1] + step, step), cfg["strategy"]["lgd"], cfg["strategy"]["cost_of_funds"])
    pol.pop("_sweep")
    update_metadata(cfg, MODEL_NAME, version, {"policy": pol})

    if auc_new >= auc_cur - PROMOTE_TOLERANCE:
        set_latest_version(cfg, MODEL_NAME, version)
        entry["action"] = f"promoted {version}"
        post = monitor(cfg, version=version, write=False)       # sanity check of the new model (report not overwritten)
        entry["post_retrain_monitoring"] = {"retrain_recommended": post["retrain_recommended"],
                                            "statuses": {r["vintage"]: r["status"] for r in post["monthly"]}}
    else:
        entry["action"] = f"kept {cur_meta['version']} (candidate {version} was worse)"
    log.info("%s | holdout AUC current=%.3f candidate=%.3f", entry["action"], auc_cur, auc_new)
    (reports / "retrain_log.json").write_text(json.dumps(entry, indent=2, default=str))
    return entry


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(force=ap.parse_args().force)
