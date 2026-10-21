"""Leakage tests: features must not change when data from the snapshot onward is perturbed."""
import numpy as np
import pandas as pd

from src.features import FEATURES, build_features
from tests.helpers import small_tables


def _by_loan(f):
    return f.set_index("loan_id")[FEATURES].sort_index()


def test_future_activity_does_not_change_features():
    cfg, _, clean = small_tables()
    base = _by_loan(build_features(clean, cfg))
    L = clean["loans"]
    T = L["disbursement_date"].quantile(0.5).normalize()
    rng = np.random.default_rng(0)
    pert = {k: v.copy() for k, v in clean.items()}
    for tbl, dcol, cols in [("gsm_daily", "date", ["topup_count", "topup_amount", "voice_min", "data_mb", "sms_count"]),
                            ("mpesa_txn", "txn_date", ["amount"]), ("wallet_daily", "date", ["balance_etb"])]:
        m = pert[tbl][dcol] >= T
        for c in cols:
            pert[tbl][c] = pert[tbl][c].astype(float)
            pert[tbl].loc[m, c] = pert[tbl].loc[m, c] * rng.uniform(0.0, 5.0, m.sum())
    pert["mpesa_txn"].loc[pert["mpesa_txn"]["txn_date"] >= T, "direction"] = "IN"
    after = _by_loan(build_features(pert, cfg))
    ids = L.loc[L["disbursement_date"] <= T, "loan_id"]
    pd.testing.assert_frame_equal(base.loc[ids], after.loc[ids])


def test_own_outcome_does_not_change_features():
    cfg, _, clean = small_tables()
    base = _by_loan(build_features(clean, cfg))
    pert = {k: v.copy() for k, v in clean.items()}
    L = pert["loans"]
    # flip outcomes of each customer's FINAL loan only (earlier loans legitimately feed later loans' history)
    is_final = L["disbursement_date"] == L.groupby("customer_id")["disbursement_date"].transform("max")
    last = is_final & (L["disbursement_date"] >= L["disbursement_date"].quantile(0.5))
    flip = last & (L["default_flag"] == 0)
    L.loc[last, "default_flag"] = 1 - L.loc[last, "default_flag"]
    L.loc[last & (L["repaid_date"].notna()), "repaid_date"] = pd.NaT
    L.loc[last, "outcome_date"] = L.loc[last, "disbursement_date"] + pd.Timedelta(days=30)
    after = _by_loan(build_features(pert, cfg))
    ids = L.loc[last, "loan_id"]
    assert flip.any()
    pd.testing.assert_frame_equal(base.loc[ids], after.loc[ids])


def test_known_values_on_handmade_data():
    d0 = pd.Timestamp("2025-01-01")
    day = lambda i: d0 + pd.Timedelta(days=i)  # noqa: E731
    D = 111
    customers = pd.DataFrame({"customer_id": ["A"], "region": ["OROMIA"], "device_type": ["SMARTPHONE"], "tenure_months_at_start": [12]})
    gsm = pd.DataFrame({"customer_id": "A", "date": [day(i) for i in range(D)],
                        "topup_count": [1 if i % 10 == 0 else 0 for i in range(D)], "topup_amount": [20.0 if i % 10 == 0 else 0.0 for i in range(D)],
                        "voice_min": 5.0, "data_mb": 100.0, "sms_count": 2})
    mp = pd.DataFrame({"txn_id": ["t1", "t2", "t3"], "customer_id": "A",
                       "txn_date": [day(99), day(99), day(100)], "txn_type": ["CASH_IN", "CASH_OUT", "CASH_IN"],
                       "direction": ["IN", "OUT", "IN"], "amount": [100.0, 40.0, 1000.0]})
    wal = pd.DataFrame({"customer_id": "A", "date": [day(i) for i in range(D)], "balance_etb": 50.0})
    loans = pd.DataFrame({"loan_id": ["L1"], "customer_id": ["A"], "disbursement_date": [day(100)], "principal_etb": [100],
                          "fee_etb": [10.0], "due_date": [day(115)], "repaid_date": [day(105)], "default_flag": [0],
                          "outcome_date": [day(105)]})
    f = build_features({"customers": customers, "gsm_daily": gsm, "mpesa_txn": mp, "wallet_daily": wal, "loans": loans}).iloc[0]
    assert f["topup_cnt_7d"] == 0 and f["topup_cnt_30d"] == 3 and f["topup_cnt_90d"] == 9   # days 70,80,90 / 10..90
    assert f["mp_inflow_30d"] == 100 and f["mp_outflow_30d"] == 40                          # day-100 txn excluded
    assert abs(f["inflow_outflow_ratio_30d"] - 101 / 41) < 1e-9
    assert f["days_since_last_mpesa_txn"] == 1
    assert f["wallet_bal_mean_30d"] == 50 and f["wallet_bal_std_90d"] == 0
    assert f["prior_loan_cnt"] == 0 and np.isnan(f["last_loan_late"])
