"""Credit-risk metrics: AUC, Gini, KS, Brier, calibration, decile lift, PSI."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score


def auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def gini(y, p) -> float:
    return 2 * auc(y, p) - 1


def ks(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic) if len(np.unique(y)) == 2 else float("nan")


def summary(y, p) -> dict:
    return {"n": int(len(y)), "bad_rate": float(np.mean(y)), "auc": auc(y, p), "gini": gini(y, p),
            "ks": ks(y, p), "brier": float(brier_score_loss(y, p))}


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    t = d.groupby("bin").agg(n=("y", "size"), mean_pd=("p", "mean"), observed_bad_rate=("y", "mean")).reset_index()
    return t


def decile_table(y, p) -> pd.DataFrame:
    """Deciles ordered from riskiest (1) to safest (10) with cumulative capture of bads."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["decile"] = pd.qcut(-d["p"].rank(method="first"), 10, labels=range(1, 11))
    t = d.groupby("decile", observed=True).agg(n=("y", "size"), bads=("y", "sum"), mean_pd=("p", "mean"),
                                               bad_rate=("y", "mean")).reset_index()
    t["cum_bad_capture"] = t["bads"].cumsum() / t["bads"].sum()
    t["lift"] = t["bad_rate"] / d["y"].mean()
    return t


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the reference (expected) sample.
    <0.10 stable, 0.10-0.25 watch, >0.25 significant shift."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    cuts = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0], cuts[-1] = -np.inf, np.inf
    pe = np.clip(np.histogram(e, cuts)[0] / len(e), 1e-6, None)
    pa = np.clip(np.histogram(a, cuts)[0] / len(a), 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_categorical(expected, actual) -> float:
    e, a = pd.Series(expected).fillna("NA"), pd.Series(actual).fillna("NA")
    cats = sorted(set(e) | set(a))
    pe = np.clip(e.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy(), 1e-6, None)
    pa = np.clip(a.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy(), 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))
