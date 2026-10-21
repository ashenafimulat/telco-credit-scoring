"""Logistic-regression scorecard on Weight-of-Evidence (WOE) features.

Standard industry recipe:  bin -> WOE/IV -> select (IV, correlation, sign) -> logistic regression ->
scale to points (base score at base odds, PDO points to double the odds) -> reason codes.

WOE = ln(%good / %bad), so higher WOE = safer. Because the regression predicts *default*, coefficients
should be negative; a positive coefficient is counter-intuitive and the feature is dropped.
Missing values get their own bin. Fully reproducible, no dependency beyond numpy/pandas/sklearn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import CATEGORICAL, FEATURE_DESCRIPTIONS


RARE = "__RARE__"  # bucket for infrequent categories (cannot collide with a real value such as region 'OTHER')


class WOEBinner:
    def __init__(self, n_bins: int = 8, min_bin_frac: float = 0.05, smoothing: float = 0.5):
        self.n_bins, self.min_bin_frac, self.s = n_bins, min_bin_frac, smoothing
        self.tables: dict[str, pd.DataFrame] = {}
        self.edges: dict[str, np.ndarray] = {}
        self.iv: dict[str, float] = {}

    def _assign(self, col: str, x: pd.Series) -> pd.Series:
        if col in CATEGORICAL:
            lab = x.fillna("MISSING").astype(str)
            return lab.where(lab.isin(self.tables[col].index), RARE)
        b = pd.cut(x, self.edges[col], labels=False, include_lowest=True)
        return b.astype("float").map(lambda v: "MISSING" if np.isnan(v) else int(v)).astype(object)

    def fit(self, X: pd.DataFrame, y: pd.Series, cols: list[str]) -> "WOEBinner":
        y = pd.Series(np.asarray(y), index=X.index)
        n_bad, n_good = y.sum(), (1 - y).sum()
        for col in cols:
            x = X[col]
            if col in CATEGORICAL:
                lab = x.fillna("MISSING").astype(str)
                freq = lab.value_counts(normalize=True)
                keep = freq[freq >= self.min_bin_frac].index
                self.tables[col] = pd.DataFrame(index=pd.Index(list(keep) + [RARE]))
                binned = lab.where(lab.isin(keep), RARE)
            else:
                xv = x.dropna()
                qs = np.unique(np.quantile(xv, np.linspace(0, 1, self.n_bins + 1))) if len(xv) else np.array([0, 1])
                edges = list(qs)
                while len(edges) > 2:  # merge bins smaller than min_bin_frac
                    cnt = np.histogram(xv, edges)[0] / len(xv)
                    i = int(np.argmin(cnt))
                    if cnt[i] >= self.min_bin_frac:
                        break
                    del edges[i + 1 if i < len(cnt) - 1 else i]
                if len(edges) < 2:
                    edges = [float(xv.min()), float(xv.max()) + 1e-9]
                edges[0], edges[-1] = -np.inf, np.inf
                self.edges[col] = np.array(edges)
                self.tables[col] = pd.DataFrame(index=pd.Index(list(range(len(edges) - 1)) + ["MISSING"]))
                binned = self._assign(col, x)
            g = pd.DataFrame({"b": binned, "y": y}).groupby("b")["y"].agg(["size", "sum"])
            t = self.tables[col].join(g).fillna(0)
            t.columns = ["n", "bad"]
            t["good"] = t["n"] - t["bad"]
            pg = (t["good"] + self.s) / (n_good + self.s * len(t))
            pb = (t["bad"] + self.s) / (n_bad + self.s * len(t))
            t["woe"] = np.log(pg / pb)
            t["iv_part"] = (pg - pb) * t["woe"]
            t["bad_rate"] = t["bad"] / t["n"].replace(0, np.nan)
            if col not in CATEGORICAL:
                t["lower"] = [self.edges[col][i] if isinstance(i, int) else np.nan for i in t.index]
                t["upper"] = [self.edges[col][i + 1] if isinstance(i, int) else np.nan for i in t.index]
            self.tables[col] = t
            self.iv[col] = float(t["iv_part"].sum())
        return self

    def transform(self, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = {}
        for col in cols:
            woe = self.tables[col]["woe"]
            out[col] = self._assign(col, X[col]).map(woe).astype(float).to_numpy()
        return pd.DataFrame(out, index=X.index)

    def bins_of(self, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame({c: self._assign(c, X[c]).to_numpy() for c in cols}, index=X.index)


class Scorecard:
    def __init__(self, n_bins=8, min_iv=0.02, max_corr=0.8, base_score=600, base_odds=20, pdo=50, C=1.0):
        self.binner = WOEBinner(n_bins=n_bins)
        self.min_iv, self.max_corr = min_iv, max_corr
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.C = C
        self.features: list[str] = []
        self.lr: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame, y, candidates: list[str]) -> "Scorecard":
        y = np.asarray(y)
        self.binner.fit(X, pd.Series(y, index=X.index), candidates)
        ranked = sorted([c for c in candidates if self.binner.iv[c] >= self.min_iv],
                        key=lambda c: -self.binner.iv[c])
        W = self.binner.transform(X, ranked)
        chosen: list[str] = []
        for c in ranked:  # greedy de-correlation, keeping the higher-IV feature
            if all(abs(W[c].corr(W[k])) < self.max_corr for k in chosen):
                chosen.append(c)
        while True:  # drop wrong-signed features (WOE is good-oriented -> coefficient must be < 0)
            lr = LogisticRegression(C=self.C, max_iter=1000).fit(W[chosen], y)
            wrong = [c for c, b in zip(chosen, lr.coef_[0]) if b > 0]
            if not wrong:
                break
            chosen.remove(min(wrong, key=lambda c: self.binner.iv[c]))
        self.features, self.lr = chosen, lr
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.binner.transform(X, self.features))[:, 1]

    def points(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature points; row sums equal the scorecard score."""
        W = self.binner.transform(X, self.features)
        k = len(self.features)
        b0 = self.lr.intercept_[0]
        pts = {c: -self.factor * (self.lr.coef_[0][i] * W[c]) + (self.offset - self.factor * b0) / k
               for i, c in enumerate(self.features)}
        return pd.DataFrame(pts, index=X.index)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.points(X).sum(axis=1).to_numpy()

    def max_points(self) -> pd.Series:
        k, b0 = len(self.features), self.lr.intercept_[0]
        m = {}
        for i, c in enumerate(self.features):
            best = self.binner.tables[c]["woe"].max()
            m[c] = -self.factor * self.lr.coef_[0][i] * best + (self.offset - self.factor * b0) / k
        return pd.Series(m)

    def reason_codes(self, X: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        """Adverse-action reasons: features that cost the most points versus their best bin."""
        short = self.max_points() - self.points(X)
        out = []
        for i in range(len(X)):
            s = short.iloc[i].sort_values(ascending=False)
            s = s[s > 1e-6].head(top)
            out.append([{"feature": f, "points_lost": round(float(v), 1),
                         "reason": FEATURE_DESCRIPTIONS.get(f, f)} for f, v in s.items()])
        return out

    def scorecard_table(self) -> pd.DataFrame:
        rows = []
        pts = {c: -self.factor * self.lr.coef_[0][i] for i, c in enumerate(self.features)}
        for c in self.features:
            t = self.binner.tables[c].copy()
            t["feature"], t["iv"], t["coef"] = c, self.binner.iv[c], float(self.lr.coef_[0][self.features.index(c)])
            t["points"] = pts[c] * t["woe"]
            rows.append(t.reset_index(names="bin"))
        return pd.concat(rows, ignore_index=True)

    def iv_table(self) -> pd.DataFrame:
        return (pd.DataFrame({"feature": list(self.binner.iv), "iv": list(self.binner.iv.values())})
                .assign(selected=lambda d: d["feature"].isin(self.features)).sort_values("iv", ascending=False))
