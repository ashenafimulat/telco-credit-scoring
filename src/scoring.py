"""Shared scoring helpers (used by training, evaluation, monitoring and the API)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def probability_to_credit_score(p_default):
    """Original 200-800 mapping from the upstream project (legacy_snapshot/), kept for continuity."""
    p_good = 1 - np.asarray(p_default, dtype=float)
    return np.clip(200 + p_good * 600, 200, 800)


def predict_frame(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """PDs from both models, the champion PD, scorecard points and the legacy 200-800 score."""
    out = pd.DataFrame(index=X.index)
    out["pd_scorecard"] = bundle["scorecard"].predict_proba(X)
    out["pd_gbm"] = bundle["gbm"].predict_proba(X)
    out["pd"] = out["pd_scorecard"] if bundle["champion"] == "scorecard" else out["pd_gbm"]
    out["scorecard_points"] = bundle["scorecard"].score(X)
    out["score_200_800"] = probability_to_credit_score(out["pd"])
    return out
