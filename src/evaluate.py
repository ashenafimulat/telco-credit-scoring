"""Evaluation of the registered model: discrimination, calibration, lift, vintage back-test, plots.

Writes reports/metrics.json, reports/predictions.csv.gz and figures. All reported numbers on the
`test` split are strictly out-of-time (loans disbursed after valid_end).
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from . import metrics as M  # noqa: E402
from .config import layer_dir, load_config  # noqa: E402
from .explain import global_importance  # noqa: E402
from .registry import load_bundle  # noqa: E402
from .scoring import predict_frame  # noqa: E402
from .train import MODEL_NAME, load_features  # noqa: E402
from .utils import get_logger, write_table  # noqa: E402

log = get_logger("evaluate")


def label_splits(f: pd.DataFrame, split: dict) -> pd.Series:
    return pd.Series(np.where(f["snapshot_date"] <= split["train_end"], "train",
                              np.where(f["snapshot_date"] <= split["valid_end"], "valid", "test")), index=f.index)


def build_predictions(cfg, version="latest") -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    bundle, meta = load_bundle(cfg, MODEL_NAME, version)
    f = load_features(cfg)
    pred = predict_frame(bundle, f)
    out = pd.concat([f[["loan_id", "customer_id", "snapshot_date", "vintage", "default", "fee_etb", "loan_amount"]], pred], axis=1)
    out["split"] = label_splits(f, meta["split"])
    return f, out, bundle, meta


def run(cfg: dict | None = None, version: str = "latest") -> dict:
    cfg = cfg or load_config()
    reports = layer_dir(cfg, "reports")
    figs = reports / "figures"
    figs.mkdir(exist_ok=True)
    f, pred, bundle, meta = build_predictions(cfg, version)
    write_table(pred, reports / "predictions.csv.gz")

    res: dict = {"champion": meta["champion"], "version": meta["version"], "gbm_engine": meta["gbm_engine"],
                 "models": {}, "vintage_auc": {}}
    for model, col in [("scorecard", "pd_scorecard"), ("gbm", "pd_gbm")]:
        res["models"][model] = {s: M.summary(d["default"], d[col]) for s, d in pred.groupby("split")}
    champ = pred["pd"]
    res["models"]["champion"] = {s: M.summary(d["default"], d["pd"]) for s, d in pred.groupby("split")}
    res["vintage_auc"] = {v: M.auc(d["default"], d["pd"]) for v, d in pred.groupby("vintage")}

    # a retrained model has no later data left: its most recent window (validation) is then the holdout
    hold = "test" if (pred["split"] == "test").any() else "valid"
    res["holdout_split"] = hold
    te = pred[pred["split"] == hold]
    M.calibration_table(te["default"], te["pd"]).to_csv(reports / "calibration_test.csv", index=False)
    M.decile_table(te["default"], te["pd"]).to_csv(reports / "deciles_test.csv", index=False)
    bundle["scorecard"].iv_table().to_csv(reports / "feature_iv.csv", index=False)
    bundle["scorecard"].scorecard_table().to_csv(reports / "scorecard_table.csv", index=False)

    imp, method = global_importance(bundle["gbm"], f.loc[te.index], te["default"], figs / "feature_importance.png", cfg["seed"])
    imp.to_csv(reports / "feature_importance.csv", index=False)
    res["importance_method"] = method

    # ---- plots ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for model, col in [("Scorecard", "pd_scorecard"), ("GBM", "pd_gbm")]:
        fpr, tpr, _ = roc_curve(te["default"], te[col])
        ax[0].plot(fpr, tpr, label=f"{model} AUC={M.auc(te['default'], te[col]):.3f}")
    ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_title("ROC, out-of-time test"); ax[0].legend()
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    s = te.sort_values("pd")
    cg = np.cumsum(1 - s["default"]) / (1 - s["default"]).sum()
    cb = np.cumsum(s["default"]) / s["default"].sum()
    ax[1].plot(np.arange(len(s)) / len(s), cg, label="cum. good"); ax[1].plot(np.arange(len(s)) / len(s), cb, label="cum. bad")
    ax[1].set_title(f"KS = {M.ks(te['default'], te['pd']):.3f} (champion)"); ax[1].legend(); ax[1].set_xlabel("Population sorted by PD")
    cal = M.calibration_table(te["default"], te["pd"])
    ax[2].plot(cal["mean_pd"], cal["observed_bad_rate"], "o-", label="champion"); lim = max(cal["mean_pd"].max(), cal["observed_bad_rate"].max()) * 1.1
    ax[2].plot([0, lim], [0, lim], "k:"); ax[2].set_title("Calibration (deciles)"); ax[2].set_xlabel("Mean predicted PD"); ax[2].set_ylabel("Observed default rate")
    plt.tight_layout(); plt.savefig(figs / "discrimination.png", dpi=130); plt.close()

    va = pd.Series(res["vintage_auc"])
    plt.figure(figsize=(8, 3.6)); plt.plot(va.index, va.values, "o-")
    plt.axvline(list(va.index).index(meta["split"]["train_end"][:7]) + 0.5, color="grey", ls=":")
    plt.axvline(list(va.index).index(meta["split"]["valid_end"][:7]) + 0.5, color="red", ls=":")
    plt.xticks(rotation=45); plt.ylabel("AUC"); plt.title("Back-test: AUC by disbursement vintage (grey=train end, red=valid end)")
    plt.tight_layout(); plt.savefig(figs / "vintage_auc.png", dpi=130); plt.close()

    (reports / "metrics.json").write_text(json.dumps(res, indent=2))
    c = res["models"]["champion"][hold]
    log.info("%s holdout champion=%s AUC=%.3f Gini=%.3f KS=%.3f", hold.upper(), meta["champion"], c["auc"], c["gini"], c["ks"])
    return res


if __name__ == "__main__":
    run()
