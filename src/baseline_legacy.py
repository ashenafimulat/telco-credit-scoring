"""Baseline on the ORIGINAL upstream snapshot dataset (legacy_snapshot/synthetic_data.csv).

Shows what the flat, un-timed data supports (a quick benchmark) and why the project moved to
time-stamped tables: no point-in-time features, no out-of-time test, no repayment history.
Random 70/30 split only - that is all a snapshot table allows.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics as M
from .config import ROOT, layer_dir, load_config

PATH = ROOT / "legacy_snapshot" / "synthetic_data.csv"


def run() -> dict:
    df = pd.read_csv(PATH)
    y = df["default"]
    X = df.drop(columns=[c for c in ["default", "uuid"] if c in df.columns]).select_dtypes("number")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    out = {"rows": len(df), "default_rate": float(y.mean())}
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42).fit(Xtr, ytr)
    out["logistic"] = M.summary(yte, lr.predict_proba(Xte)[:, 1])
    out["gbm"] = M.summary(yte, gb.predict_proba(Xte)[:, 1])
    (layer_dir(load_config(), "reports") / "legacy_baseline.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
