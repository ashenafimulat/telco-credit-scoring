"""Model monitoring: score PSI, feature CSI, performance and approval/default trends by vintage.

Reference = development (train) population of the registered model. Each later disbursement month
is compared to it. Status rules (thresholds in configs/config.yaml):

    RED    score PSI >= psi_alert   OR   AUC dropped >= auc_drop_alert vs validation
    AMBER  score PSI >= psi_warn    OR   any feature CSI >= psi_alert  OR  AUC dropped >= half the limit
    GREEN  otherwise

`retrain_recommended` is True when any of the last two months is RED. In production labels lag by the
loan outcome window (30 days here); the simulation has all outcomes, so AUC is available for every month.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import metrics as M  # noqa: E402
from .config import layer_dir, load_config  # noqa: E402
from .evaluate import build_predictions  # noqa: E402
from .features import CATEGORICAL  # noqa: E402
from .utils import get_logger  # noqa: E402

log = get_logger("monitoring")


def status_for(score_psi: float, max_csi: float, auc_drop: float, th: dict) -> str:
    if score_psi >= th["psi_alert"] or (not np.isnan(auc_drop) and auc_drop >= th["auc_drop_alert"]):
        return "RED"
    if score_psi >= th["psi_warn"] or max_csi >= th["psi_alert"] or (not np.isnan(auc_drop) and auc_drop >= th["auc_drop_alert"] / 2):
        return "AMBER"
    return "GREEN"


def monitor(cfg: dict | None = None, version: str = "latest", write: bool = True) -> dict:
    cfg = cfg or load_config()
    th = cfg["monitoring"]
    f, pred, bundle, meta = build_predictions(cfg, version)
    feats = bundle["scorecard"].features
    ref_mask = pred["split"] == "train"
    ref_f, ref_p = f[ref_mask], pred[ref_mask]
    valid_auc = M.auc(pred.loc[pred["split"] == "valid", "default"], pred.loc[pred["split"] == "valid", "pd"])
    cutoff = (meta.get("policy") or {}).get("cutoff_pd")

    rows, csi_rows = [], []
    for vint, idx in pred[pred["split"] != "train"].groupby("vintage").groups.items():
        p, x = pred.loc[idx], f.loc[idx]
        csi = {}
        for c in feats:
            csi[c] = M.psi_categorical(ref_f[c], x[c]) if c in CATEGORICAL else M.psi(ref_f[c], x[c])
        worst = max(csi, key=lambda k: -1 if np.isnan(csi[k]) else csi[k])
        a = M.auc(p["default"], p["pd"])
        spsi = M.psi(ref_p["pd"], p["pd"])
        rows.append({"vintage": vint, "split": p["split"].iloc[0], "loans": len(p), "score_psi": spsi,
                     "max_csi": csi[worst], "max_csi_feature": worst, "auc": a, "auc_drop_vs_valid": valid_auc - a,
                     "default_rate": p["default"].mean(), "mean_pd": p["pd"].mean(),
                     "approval_rate_at_policy": float((p["pd"] <= cutoff).mean()) if cutoff else np.nan,
                     "status": status_for(spsi, csi[worst], valid_auc - a, th)})
        csi_rows += [{"vintage": vint, "feature": c, "csi": v} for c, v in csi.items()]
    mon = pd.DataFrame(rows).sort_values("vintage").reset_index(drop=True)
    csi_df = pd.DataFrame(csi_rows)
    recent = mon.tail(2)
    retrain = bool((recent["status"] == "RED").any())
    reason = None
    if retrain:
        r = recent[recent["status"] == "RED"].iloc[-1]
        reason = (f"{r['vintage']}: score PSI {r['score_psi']:.3f} (alert {th['psi_alert']}), "
                  f"AUC {r['auc']:.3f} vs {valid_auc:.3f} on validation; top drifting feature {r['max_csi_feature']} (CSI {r['max_csi']:.2f})")
    result = {"model_version": meta["version"], "retrain_recommended": retrain, "reason": reason,
              "valid_auc": valid_auc, "monthly": mon.to_dict("records")}

    if write:
        reports = layer_dir(cfg, "reports")
        mon.to_csv(reports / "monitoring_monthly.csv", index=False)
        csi_df.to_csv(reports / "monitoring_csi.csv", index=False)
        (reports / "monitoring.json").write_text(json.dumps(result, indent=2, default=float))
        colors = mon["status"].map({"GREEN": "#4c9f70", "AMBER": "#e0a030", "RED": "#c0392b"})
        fig, ax = plt.subplots(1, 3, figsize=(17, 4))
        ax[0].bar(mon["vintage"], mon["score_psi"], color="#4a78b5")
        ax[0].axhline(th["psi_warn"], color="orange", ls="--"); ax[0].axhline(th["psi_alert"], color="red", ls="--")
        ax[0].set_title("Score PSI vs development sample")
        ax[1].bar(mon["vintage"], mon["max_csi"], color="#4a78b5"); ax[1].axhline(th["psi_alert"], color="red", ls="--")
        ax[1].set_title("Worst feature CSI (log scale)"); ax[1].set_yscale("log")
        ax[2].plot(mon["vintage"], mon["auc"], "-", color="grey"); ax[2].scatter(mon["vintage"], mon["auc"], c=colors, s=80, zorder=3)
        ax[2].axhline(valid_auc, color="grey", ls=":", label="validation AUC")
        ax[2].set_title("AUC by vintage (dot colour = status: green/amber/red)"); ax[2].legend()
        for a_ in ax:
            a_.tick_params(axis="x", rotation=60)
        plt.tight_layout(); plt.savefig(reports / "figures" / "monitoring.png", dpi=130); plt.close()

        top = (csi_df[csi_df["vintage"] == mon["vintage"].iloc[-1]].sort_values("csi", ascending=False).head(8))
        md = [f"# Monitoring report: model {meta['version']} ({meta['champion']})", "",
              f"**Retrain recommended: {'YES' if retrain else 'no'}**" + (f"  \n{reason}" if reason else ""), "",
              "| Vintage | Split | Loans | Score PSI | Max CSI (feature) | AUC | Default rate | Approval @ policy | Status |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in mon.itertuples():
            md.append(f"| {r.vintage} | {r.split} | {r.loans} | {r.score_psi:.3f} | {r.max_csi:.2f} ({r.max_csi_feature}) | {r.auc:.3f} | "
                      f"{r.default_rate:.1%} | {r.approval_rate_at_policy:.0%} | {r.status} |")
        md += ["", f"## Top drifting features in {mon['vintage'].iloc[-1]} (CSI)", "", "| Feature | CSI |", "|---|---|"]
        md += [f"| {r.feature} | {r.csi:.3f} |" for r in top.itertuples()]
        md += ["", "PSI/CSI guide: < 0.10 stable, 0.10-0.25 watch, > 0.25 significant shift."]
        (reports / "monitoring_report.md").write_text("\n".join(md) + "\n")
    log.info("retrain_recommended=%s %s", retrain, reason or "")
    return result


if __name__ == "__main__":
    monitor()
