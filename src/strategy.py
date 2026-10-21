"""Business strategy simulation: cut-off, expected loss, profit and credit-limit tiers.

Economics per loan (all in ETB):  repaid -> +fee;  default -> -(LGD * principal);  every approved
loan costs `cost_of_funds * principal`. The cut-off is CHOSEN ON THE VALIDATION SET and then judged
on the out-of-time TEST set, so the headline numbers are not tuned on the data they are reported on.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import layer_dir, load_config  # noqa: E402
from .registry import update_metadata  # noqa: E402
from .train import MODEL_NAME  # noqa: E402
from .utils import get_logger, read_table  # noqa: E402

log = get_logger("strategy")
TIER_NAMES = ["Tier 1 (lowest risk)", "Tier 2", "Tier 3"]
TIER_LIMIT_MULT = [1.5, 1.0, 0.6]


def loan_profit(d: pd.DataFrame, lgd: float, cof: float) -> pd.Series:
    p = d["loan_amount"]
    return np.where(d["default"] == 1, -lgd * p, d["fee_etb"]) - cof * p


def sweep(d: pd.DataFrame, cutoffs, lgd, cof) -> pd.DataFrame:
    prof = pd.Series(loan_profit(d, lgd, cof), index=d.index)
    rows = []
    for c in cutoffs:
        a = d["pd"] <= c
        n = int(a.sum())
        rows.append({"cutoff_pd": round(float(c), 4), "approval_rate": a.mean(),
                     "default_rate_approved": d.loc[a, "default"].mean() if n else np.nan,
                     "expected_loss_approved": d.loc[a, "pd"].mean() * lgd if n else np.nan,
                     "realized_loss_etb": float((lgd * d.loc[a & (d["default"] == 1), "loan_amount"]).sum()),
                     "net_profit_etb": float(prof[a].sum()),
                     "profit_per_approved_etb": float(prof[a].mean()) if n else np.nan})
    return pd.DataFrame(rows)


def make_policy(va: pd.DataFrame, grid, lgd: float, cof: float) -> dict:
    """Pick the profit-maximising PD cut-off and 3 risk-tier edges from a validation window."""
    sv = sweep(va, grid, lgd, cof)
    cutoff = float(sv.loc[sv["net_profit_etb"].idxmax(), "cutoff_pd"])
    appr = va[va["pd"] <= cutoff]["pd"]
    edges = [float(appr.quantile(1 / 3)), float(appr.quantile(2 / 3))]
    return {"cutoff_pd": cutoff, "tier_pd_edges": edges, "tier_names": TIER_NAMES,
            "tier_limit_mult": TIER_LIMIT_MULT, "chosen_on": "validation", "lgd": lgd, "cost_of_funds": cof,
            "_sweep": sv}


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    reports = layer_dir(cfg, "reports")
    pred = read_table(reports / "predictions.csv.gz")
    if not (pred["split"] == "test").any():
        log.info("No out-of-time test window for this model version (retrained); its policy was set by src.retrain. Skipping.")
        return {}
    lgd, cof = cfg["strategy"]["lgd"], cfg["strategy"]["cost_of_funds"]
    step = cfg["strategy"]["cutoff_grid_step"]
    va, te = pred[pred["split"] == "valid"], pred[pred["split"] == "test"]
    grid = np.arange(round(pred["pd"].quantile(0.05), 2), round(pred["pd"].quantile(0.995), 2) + step, step)

    pol_cfg = make_policy(va, grid, lgd, cof)
    sv, st = pol_cfg.pop("_sweep"), sweep(te, grid, lgd, cof)
    cutoff = pol_cfg["cutoff_pd"]
    log.info("policy cut-off chosen on validation: PD <= %.3f", cutoff)

    base = sweep(te, [te["pd"].max() + 1], lgd, cof).iloc[0]          # approve everyone
    pol = sweep(te, [cutoff], lgd, cof).iloc[0]
    tight = sweep(te, [max(grid.min(), cutoff - 0.03)], lgd, cof).iloc[0]

    # ---- tiers (edges from validation-approved population, applied out-of-time) ----
    edges = pol_cfg["tier_pd_edges"]
    appr = te[te["pd"] <= cutoff].copy()
    appr["tier"] = np.digitize(appr["pd"], edges)
    appr["profit"] = loan_profit(appr, lgd, cof)
    appr["mult"] = np.array(TIER_LIMIT_MULT)[appr["tier"]]
    tiers = appr.groupby("tier").agg(loans=("default", "size"), mean_pd=("pd", "mean"),
                                     default_rate=("default", "mean"), flat_profit_etb=("profit", "sum"),
                                     limit_mult=("mult", "first")).reset_index()
    tiers["tier"] = [TIER_NAMES[i] for i in tiers["tier"]]
    tiers["tiered_profit_etb"] = tiers["flat_profit_etb"] * tiers["limit_mult"]

    sv.assign(split="valid").to_csv(reports / "cutoff_sweep_valid.csv", index=False)
    st.assign(split="test").to_csv(reports / "cutoff_sweep_test.csv", index=False)
    tiers.to_csv(reports / "limit_tiers_test.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    ax[0].plot(st["cutoff_pd"], st["approval_rate"], label="approval rate")
    ax[0].plot(st["cutoff_pd"], st["default_rate_approved"], label="default rate (approved)")
    ax[0].axvline(cutoff, color="k", ls=":"); ax[0].legend(); ax[0].set_xlabel("PD cut-off"); ax[0].set_title("Approval vs default rate (test)")
    ax[1].plot(st["cutoff_pd"], st["net_profit_etb"], label="test"); ax[1].plot(sv["cutoff_pd"], sv["net_profit_etb"], label="valid", alpha=.6)
    ax[1].axvline(cutoff, color="k", ls=":"); ax[1].axhline(0, color="grey", lw=.5); ax[1].legend()
    ax[1].set_xlabel("PD cut-off"); ax[1].set_title("Net profit (ETB) by cut-off"); plt.tight_layout()
    plt.savefig(reports / "figures" / "cutoff_strategy.png", dpi=130); plt.close()

    d_appr = (pol["approval_rate"] - 1) * 100
    summary = {
        "cutoff_pd": cutoff,
        "approve_all": {k: float(base[k]) for k in ["approval_rate", "default_rate_approved", "net_profit_etb", "realized_loss_etb"]},
        "policy": {k: float(pol[k]) for k in ["approval_rate", "default_rate_approved", "net_profit_etb", "realized_loss_etb"]},
        "tiers": tiers.to_dict("records"),
        "headline": (f"Applying a PD cut-off of {cutoff:.2f} (chosen on validation) to the out-of-time test set changes approvals by "
                     f"{d_appr:+.0f} pts, cuts the default rate among approved loans from {base['default_rate_approved']:.1%} to "
                     f"{pol['default_rate_approved']:.1%} and moves net profit from {base['net_profit_etb']:,.0f} to {pol['net_profit_etb']:,.0f} ETB."),
    }
    (reports / "strategy.json").write_text(json.dumps(summary, indent=2, default=float))
    md = ["# Cut-off and credit-limit strategy (out-of-time test set)", "", summary["headline"], "",
          f"Economics: fee on repaid loans, LGD {lgd:.0%} on defaults, cost of funds {cof:.0%} of principal. "
          "Cut-off chosen on the validation set only.", "", "## Cut-off comparison (test)", "",
          "| Policy | Approval | Default rate (approved) | Net profit (ETB) |", "|---|---|---|---|",
          f"| Approve all | {base['approval_rate']:.0%} | {base['default_rate_approved']:.1%} | {base['net_profit_etb']:,.0f} |",
          f"| PD <= {cutoff:.2f} (policy) | {pol['approval_rate']:.0%} | {pol['default_rate_approved']:.1%} | {pol['net_profit_etb']:,.0f} |",
          f"| PD <= {max(grid.min(), cutoff - 0.03):.2f} (tighter) | {tight['approval_rate']:.0%} | {tight['default_rate_approved']:.1%} | {tight['net_profit_etb']:,.0f} |",
          "", "## Limit tiers (approved loans, test)", "", "| Tier | Loans | Mean PD | Default rate | Limit multiplier | Profit flat (ETB) | Profit tiered (ETB) |",
          "|---|---|---|---|---|---|---|"]
    for r in tiers.itertuples():
        md.append(f"| {r.tier} | {r.loans} | {r.mean_pd:.1%} | {r.default_rate:.1%} | x{r.limit_mult} | {r.flat_profit_etb:,.0f} | {r.tiered_profit_etb:,.0f} |")
    (reports / "strategy.md").write_text("\n".join(md) + "\n")

    update_metadata(cfg, MODEL_NAME, "latest", {"policy": pol_cfg})
    log.info(summary["headline"])
    return summary


if __name__ == "__main__":
    run()
