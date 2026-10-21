"""Framework-free scoring service used by the FastAPI app (and unit-tested without FastAPI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config
from .features import FEATURES
from .registry import load_bundle
from .scoring import probability_to_credit_score
from .train import MODEL_NAME


class ScoringService:
    def __init__(self, cfg: dict | None = None, version: str = "latest"):
        self.cfg = cfg or load_config()
        self.bundle, self.meta = load_bundle(self.cfg, MODEL_NAME, version)
        self.policy = self.meta.get("policy")

    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows).reindex(columns=FEATURES)
        for c in FEATURES:
            if c == "region":
                df[c] = df[c].fillna("UNKNOWN").astype(str).str.upper()
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def score_batch(self, rows: list[dict]) -> list[dict]:
        X = self._frame(rows)
        sc, gbm = self.bundle["scorecard"], self.bundle["gbm"]
        pd_sc, pd_gbm = sc.predict_proba(X), gbm.predict_proba(X)
        pdv = pd_sc if self.bundle["champion"] == "scorecard" else pd_gbm
        points = sc.score(X)
        reasons = sc.reason_codes(X, top=3)
        out = []
        for i in range(len(X)):
            tier, mult, decision = None, None, "REVIEW"
            if self.policy:
                decision = "APPROVE" if pdv[i] <= self.policy["cutoff_pd"] else "DECLINE"
                if decision == "APPROVE":
                    t = int(np.digitize(pdv[i], self.policy["tier_pd_edges"]))
                    tier, mult = self.policy["tier_names"][t], self.policy["tier_limit_mult"][t]
            out.append({
                "pd": round(float(pdv[i]), 5),
                "decision": decision,
                "scorecard_points": round(float(points[i]), 1),
                "credit_score_200_800": round(float(probability_to_credit_score(pdv[i])), 1),
                "risk_tier": tier,
                "limit_multiplier": mult,
                "reason_codes": reasons[i],
                "model_version": self.meta["version"],
                "champion": self.bundle["champion"],
            })
        return out

    def score(self, features: dict) -> dict:
        return self.score_batch([features])[0]

    def info(self) -> dict:
        return {"model": MODEL_NAME, "version": self.meta["version"], "champion": self.bundle["champion"],
                "gbm_engine": self.meta.get("gbm_engine"), "policy": self.policy,
                "features": FEATURES, "created_at": self.meta.get("created_at")}
