"""Global + local explanations.

* Scorecard reason codes (points lost vs. best bin) are the adverse-action explanation: always available.
* SHAP (when installed + LightGBM engine) gives mean |SHAP| importance and a beeswarm plot.
* Otherwise permutation importance on the test set is used, and the report says which method ran.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .gbm import GBMModel, to_matrix


def global_importance(gbm: GBMModel, X: pd.DataFrame, y, out_png=None, seed: int = 42) -> tuple[pd.DataFrame, str]:
    M = to_matrix(X, gbm.features)
    try:
        if gbm.engine_ != "lightgbm":
            raise ImportError
        import shap
        sv = shap.TreeExplainer(gbm.model).shap_values(M)
        sv = sv[1] if isinstance(sv, list) else sv
        imp = pd.DataFrame({"feature": gbm.features, "importance": np.abs(sv).mean(0)})
        if out_png:
            import matplotlib.pyplot as plt
            shap.summary_plot(sv, M, show=False, max_display=15)
            plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()
        method = "mean |SHAP| (TreeExplainer)"
    except ImportError:
        r = permutation_importance(gbm.model, M, y, scoring="roc_auc", n_repeats=5, random_state=seed)
        imp = pd.DataFrame({"feature": gbm.features, "importance": r.importances_mean})
        method = "permutation importance (AUC drop) - install shap for SHAP values"
        if out_png:
            import matplotlib.pyplot as plt
            top = imp.sort_values("importance").tail(15)
            plt.figure(figsize=(7, 5)); plt.barh(top["feature"], top["importance"])
            plt.xlabel("AUC drop when permuted (test set)"); plt.title("Top features (GBM)")
            plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()
    return imp.sort_values("importance", ascending=False).reset_index(drop=True), method
