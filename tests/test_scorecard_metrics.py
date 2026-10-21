import numpy as np
import pandas as pd

from src import metrics as M
from src.monitoring import status_for
from src.scorecard import Scorecard


def _toy(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n); noise = rng.normal(size=n); z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.2 - 1.0 * x + 0.5 * z)))).astype(int)
    return pd.DataFrame({"signal": x, "noise": noise, "other": x + rng.normal(size=n), "region": rng.choice(["A", "B"], n)}), y


def test_woe_iv_and_selection():
    X, y = _toy()
    sc = Scorecard(min_iv=0.02, max_corr=0.9).fit(X, y, ["signal", "noise", "other"])
    assert sc.binner.iv["signal"] > 0.15 and sc.binner.iv["noise"] < 0.02
    assert "noise" not in sc.features and "signal" in sc.features
    assert (sc.lr.coef_ < 0).all()


def test_points_add_up_and_order_risk():
    X, y = _toy()
    sc = Scorecard(max_corr=0.9).fit(X, y, ["signal", "other"])
    pts = sc.points(X)
    np.testing.assert_allclose(pts.sum(axis=1), sc.score(X))
    assert np.corrcoef(sc.score(X), sc.predict_proba(X))[0, 1] < -0.9
    odds = lambda p: (1 - p) / p  # noqa: E731
    p0 = sc.predict_proba(X.iloc[:200]); s0 = sc.score(X.iloc[:200])
    i, j = int(np.argmin(s0)), int(np.argmax(s0))
    # PDO: points difference maps to log-odds difference via factor
    np.testing.assert_allclose((s0[j] - s0[i]) / sc.factor, np.log(odds(p0[j])) - np.log(odds(p0[i])), rtol=1e-6)


def test_reason_codes_ordered():
    X, y = _toy()
    sc = Scorecard(max_corr=0.9).fit(X, y, ["signal", "other"])
    r = sc.reason_codes(X.iloc[:5], top=2)
    for row in r:
        lost = [c["points_lost"] for c in row]
        assert lost == sorted(lost, reverse=True)


def test_metrics_and_psi():
    y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([.1, .2, .3, .7, .8, .9])
    assert M.auc(y, p) == 1 and M.gini(y, p) == 1 and M.ks(y, p) == 1
    rng = np.random.default_rng(1)
    a = rng.normal(size=20000)
    assert M.psi(a, rng.normal(size=20000)) < 0.02
    assert M.psi(a, rng.normal(1.0, 1.0, 20000)) > 0.25
    assert M.psi_categorical(["A"] * 50 + ["B"] * 50, ["A"] * 50 + ["B"] * 50) < 1e-6


def test_monitoring_status_rules():
    th = {"psi_warn": 0.1, "psi_alert": 0.25, "auc_drop_alert": 0.03}
    assert status_for(0.02, 0.05, 0.0, th) == "GREEN"
    assert status_for(0.12, 0.05, 0.0, th) == "AMBER"
    assert status_for(0.02, 0.40, 0.0, th) == "AMBER"
    assert status_for(0.30, 0.05, 0.0, th) == "RED"
    assert status_for(0.02, 0.05, 0.04, th) == "RED"
