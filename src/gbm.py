"""Gradient-boosting challenger with Platt calibration.

Uses LightGBM when installed (as in requirements.txt) and falls back to scikit-learn's
HistGradientBoostingClassifier, so the pipeline runs anywhere. `engine_` records which one trained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .features import NUMERIC

REGIONS = ["ADDIS_ABABA", "OROMIA", "AMHARA", "TIGRAY", "SNNPR", "SIDAMA", "OTHER", "UNKNOWN"]


def to_matrix(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    M = X[cols].copy()
    if "region" in M:
        M["region"] = M["region"].map({r: i for i, r in enumerate(REGIONS)}).astype(float)
    return M.astype(float)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class GBMModel:
    def __init__(self, seed: int = 42, features: list[str] | None = None):
        self.seed = seed
        self.features = features or (NUMERIC + ["region"])
        self.model = None
        self.platt: LogisticRegression | None = None
        self.engine_ = None
        self.best_iter_ = None

    def _cat_mask(self):
        return [c == "region" for c in self.features]

    def fit(self, Xtr, ytr, Xva, yva) -> "GBMModel":
        A, B = to_matrix(Xtr, self.features), to_matrix(Xva, self.features)
        try:
            import lightgbm as lgb
            self.engine_ = "lightgbm"
            m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                   random_state=self.seed, verbose=-1)
            m.fit(A, ytr, eval_set=[(B, yva)], eval_metric="auc", categorical_feature=["region"],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            self.best_iter_ = int(m.best_iteration_ or m.n_estimators)
            self.model = m
        except ImportError:
            self.engine_ = "sklearn-hist-gbm"
            kw = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=5.0,
                      categorical_features=self._cat_mask(), random_state=self.seed, early_stopping=False)
            probe = HistGradientBoostingClassifier(max_iter=400, **kw).fit(A, ytr)
            ll = [np.mean(-(yva * np.log(np.clip(p[:, 1], 1e-9, 1)) + (1 - yva) * np.log(np.clip(p[:, 0], 1e-9, 1))))
                  for p in probe.staged_predict_proba(B)]
            self.best_iter_ = int(np.argmin(ll)) + 1
            self.model = HistGradientBoostingClassifier(max_iter=self.best_iter_, **kw).fit(A, ytr)
        raw = self.model.predict_proba(B)[:, 1]
        self.platt = LogisticRegression(C=1e6).fit(_logit(raw), yva)  # Platt scaling on validation
        return self

    def predict_raw(self, X):
        return self.model.predict_proba(to_matrix(X, self.features))[:, 1]

    def predict_proba(self, X):
        return self.platt.predict_proba(_logit(self.predict_raw(X)))[:, 1]
