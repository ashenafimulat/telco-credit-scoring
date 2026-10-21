import pandas as pd

from tests.helpers import small_tables


def test_tables_and_keys():
    cfg, raw, clean = small_tables()
    assert set(raw) == {"customers", "gsm_daily", "mpesa_txn", "wallet_daily", "loans"}
    assert clean["loans"]["loan_id"].is_unique and clean["customers"]["customer_id"].is_unique


def test_no_hidden_drivers_leak():
    _, raw, _ = small_tables()
    forbidden = {"z", "income", "stress", "creditworthiness", "latent", "vol", "irreg"}
    for df in raw.values():
        assert not forbidden & set(df.columns)


def test_default_rate_near_target():
    cfg, _, clean = small_tables(n=600, seed=3)
    r = clean["loans"]["default_flag"].mean()
    assert abs(r - cfg["generator"]["target_default_rate"]) < 0.04


def test_loan_timing_rules():
    cfg, _, clean = small_tables()
    L = clean["loans"].sort_values(["customer_id", "disbursement_date"])
    gap = L.groupby("customer_id")["disbursement_date"].diff().dt.days.dropna()
    assert (gap >= cfg["generator"]["default_after_days"] + 1).all()   # prior loan always resolved
    rep = L[L["repaid_date"].notna()]
    assert (rep["repaid_date"] >= rep["disbursement_date"]).all()
    assert (L.loc[L["default_flag"] == 1, "repaid_date"].isna()).all()
    assert (L["disbursement_date"] >= pd.Timestamp(cfg["generator"]["history_start"]) + pd.Timedelta(days=90)).all()


def test_dirty_raw_data_is_cleaned():
    _, raw, clean = small_tables()
    assert raw["mpesa_txn"]["txn_id"].duplicated().any()
    assert not clean["mpesa_txn"]["txn_id"].duplicated().any()
    assert (clean["mpesa_txn"]["amount"] > 0).all()
