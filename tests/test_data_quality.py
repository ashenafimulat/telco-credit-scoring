import numpy as np

from src import data_quality as dq
from tests.helpers import small_tables


def _failed(rep):
    return {r["name"] for r in rep["results"] if r["status"] == "FAIL"}


def test_clean_data_passes():
    _, _, clean = small_tables()
    assert dq.run_suite(clean, "clean")["counts"]["FAIL"] == 0


def test_raw_data_defects_are_detected():
    _, raw, _ = small_tables()
    failed = _failed(dq.run_suite(raw, "raw"))
    assert "unique[txn_id]" in failed and "non_negative[amount]" in failed


def test_orphans_schema_and_dates_are_detected():
    _, _, clean = small_tables()
    t = {k: v.copy() for k, v in clean.items()}
    t["mpesa_txn"].loc[t["mpesa_txn"].index[:5], "customer_id"] = "C999999"
    t["loans"] = t["loans"].drop(columns=["fee_etb"])
    t["customers"]["device_type"] = t["customers"]["device_type"].replace({"SMARTPHONE": "TABLET"})
    t["wallet_daily"]["balance_etb"] = t["wallet_daily"]["balance_etb"].astype(str)
    failed = _failed(dq.run_suite(t, "bad"))
    assert {"fk[customer_id->customers]", "schema_columns", "valid_values[device_type]", "schema_dtypes"} <= failed


def test_feature_suite_flags_infinite_values():
    from src.features import FEATURES, build_features
    cfg, _, clean = small_tables()
    f = build_features(clean, cfg)
    assert dq.run_feature_suite(f, FEATURES)["counts"]["FAIL"] == 0
    f.loc[f.index[0], "topup_cnt_30d"] = np.inf
    assert "no_infinite_values" in _failed(dq.run_feature_suite(f, FEATURES))
