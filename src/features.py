"""Point-in-time feature engineering.

One row per loan, computed as of the disbursement date (the *snapshot*). Every window covers days
strictly BEFORE the snapshot, [snapshot-W, snapshot), and prior-loan features only use loans whose
outcome was already known before the snapshot. Nothing from the loan's own outcome is used.

Implementation: per-customer daily matrices + cumulative sums, so any window sum is two lookups.
tests/test_point_in_time.py proves leakage-freedom by perturbing all data on/after the snapshot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = ["loan_id", "customer_id", "snapshot_date", "vintage", "default", "fee_etb"]

FEATURE_DESCRIPTIONS = {
    "topup_cnt_7d": "Airtime top-ups in the last 7 days",
    "topup_cnt_30d": "Airtime top-ups in the last 30 days",
    "topup_cnt_90d": "Airtime top-ups in the last 90 days",
    "topup_amt_30d": "Airtime top-up value in the last 30 days (ETB)",
    "topup_avg_amt_30d": "Average top-up value in the last 30 days (ETB)",
    "topup_cnt_trend": "Top-ups in last 30 days vs the average of the prior 60 days",
    "topup_weekly_cv_12w": "Irregularity of weekly top-ups over 12 weeks (coefficient of variation)",
    "days_since_last_topup": "Days since the last airtime top-up",
    "gsm_active_days_30d": "Days with any GSM activity in the last 30 days",
    "gsm_active_days_90d": "Days with any GSM activity in the last 90 days",
    "days_since_last_activity": "Days since any GSM activity",
    "voice_min_30d": "Voice minutes in the last 30 days",
    "data_mb_30d": "Mobile data used in the last 30 days (MB)",
    "data_mb_trend": "Data use in last 30 days vs 90-day average",
    "sms_cnt_30d": "SMS sent in the last 30 days",
    "mp_txn_cnt_7d": "M-PESA transactions in the last 7 days",
    "mp_txn_cnt_30d": "M-PESA transactions in the last 30 days",
    "mp_txn_cnt_90d": "M-PESA transactions in the last 90 days",
    "mp_inflow_30d": "Wallet inflows in the last 30 days (ETB)",
    "mp_outflow_30d": "Wallet outflows in the last 30 days (ETB)",
    "inflow_outflow_ratio_30d": "Smoothed wallet inflow/outflow ratio, last 30 days",
    "mp_inflow_90d": "Wallet inflows in the last 90 days (ETB)",
    "mp_inflow_weekly_cv_12w": "Income volatility: weekly inflow coefficient of variation over 12 weeks",
    "mp_cashin_cnt_30d": "Cash-in transactions in the last 30 days",
    "mp_cashout_cnt_30d": "Cash-out transactions in the last 30 days",
    "mp_p2p_send_cnt_30d": "P2P transfers sent in the last 30 days",
    "mp_p2p_recv_cnt_30d": "P2P transfers received in the last 30 days",
    "mp_billpay_cnt_30d": "Bill payments in the last 30 days",
    "mp_merchant_cnt_30d": "Merchant payments in the last 30 days",
    "mp_avg_txn_amt_30d": "Average M-PESA transaction value, last 30 days (ETB)",
    "days_since_last_mpesa_txn": "Days since the last M-PESA transaction",
    "wallet_bal_mean_30d": "Average wallet balance, last 30 days (ETB)",
    "wallet_bal_std_90d": "Wallet balance volatility over 90 days (ETB, population std)",
    "wallet_bal_trend_7v90": "Average balance last 7 days vs last 90 days",
    "wallet_low_bal_days_30d": "Days in the last 30 with wallet balance below 100 ETB",
    "tenure_months": "Months since registration at the snapshot date",
    "is_smartphone": "Customer uses a smartphone (1) or feature phone (0)",
    "region": "Customer region",
    "loan_amount": "Principal of the loan being scored (ETB)",
    "loan_to_inflow_ratio": "Loan principal relative to average monthly wallet inflow (90d)",
    "prior_loan_cnt": "Loans resolved before this snapshot",
    "prior_late_cnt": "Prior loans repaid late or defaulted",
    "prior_default_cnt": "Prior loans that defaulted",
    "last_loan_late": "Was the most recent prior loan late or defaulted (1/0)",
    "days_since_last_loan": "Days since the most recent prior loan was disbursed",
    "avg_prior_repay_lag": "Average days to repay on prior repaid loans",
}
FEATURES = list(FEATURE_DESCRIPTIONS)
CATEGORICAL = ["region"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
CORE_PARITY = ["topup_cnt_7d", "topup_cnt_30d", "topup_cnt_90d", "topup_amt_30d", "voice_min_30d", "data_mb_30d",
               "sms_cnt_30d", "gsm_active_days_30d", "mp_txn_cnt_30d", "mp_inflow_30d", "mp_outflow_30d",
               "inflow_outflow_ratio_30d", "wallet_bal_mean_30d", "wallet_bal_std_90d", "days_since_last_mpesa_txn"]


def _dense(cidx, date0, D, df, date_col, values):
    n = len(cidx)
    ci = cidx.get_indexer(df["customer_id"])
    di = (df[date_col] - date0).dt.days.to_numpy()
    ok = (ci >= 0) & (di >= 0) & (di < D)
    arr = np.zeros((n, D))
    np.add.at(arr, (ci[ok], di[ok]), np.asarray(values, dtype=float)[ok])
    return arr


def _cum(arr):
    cs = np.zeros((arr.shape[0], arr.shape[1] + 1))
    np.cumsum(arr, axis=1, out=cs[:, 1:])
    return cs


def build_features(tables: dict[str, pd.DataFrame], cfg: dict | None = None) -> pd.DataFrame:
    cust = tables["customers"].sort_values("customer_id").reset_index(drop=True)
    gsm, mp, wal, loans = tables["gsm_daily"], tables["mpesa_txn"], tables["wallet_daily"], tables["loans"]
    after = (cfg or {}).get("generator", {}).get("default_after_days", 30)
    cidx = pd.Index(cust["customer_id"])
    n = len(cidx)
    date0 = min(gsm["date"].min(), mp["txn_date"].min(), wal["date"].min())
    date1 = max(gsm["date"].max(), mp["txn_date"].max(), wal["date"].max())
    D = (date1 - date0).days + 1

    loans = loans.sort_values(["disbursement_date", "customer_id"]).reset_index(drop=True)
    lc = cidx.get_indexer(loans["customer_id"])
    ld = (loans["disbursement_date"] - date0).dt.days.to_numpy()
    assert (lc >= 0).all(), "loan with unknown customer"

    def W(cs, a, b):  # sum over days [d-b, d-a)
        return cs[lc, np.maximum(ld - a, 0)] - cs[lc, np.maximum(ld - b, 0)]

    def last_gap(arr):
        idx = np.where(arr > 0, np.arange(D)[None, :], -1)
        last = np.maximum.accumulate(idx, axis=1)[lc, np.clip(ld - 1, 0, D - 1)]
        return np.where(last >= 0, ld - last, np.nan)

    def weekly_cv(cs, weeks=12):
        ws = np.stack([W(cs, 7 * k, 7 * (k + 1)) for k in range(weeks)], axis=1)
        m, s = ws.mean(1), ws.std(1)
        return np.where(m > 0, s / np.where(m > 0, m, 1), np.nan)

    f = pd.DataFrame({"loan_id": loans["loan_id"], "customer_id": loans["customer_id"]})

    # ---- GSM ----
    tc = _dense(cidx, date0, D, gsm, "date", gsm["topup_count"]); cs_tc = _cum(tc)
    cs_ta = _cum(_dense(cidx, date0, D, gsm, "date", gsm["topup_amount"]))
    act = _dense(cidx, date0, D, gsm, "date", np.ones(len(gsm))); cs_act = _cum(act)
    cs_voice = _cum(_dense(cidx, date0, D, gsm, "date", gsm["voice_min"]))
    cs_data = _cum(_dense(cidx, date0, D, gsm, "date", gsm["data_mb"]))
    cs_sms = _cum(_dense(cidx, date0, D, gsm, "date", gsm["sms_count"]))
    f["topup_cnt_7d"], f["topup_cnt_30d"], f["topup_cnt_90d"] = W(cs_tc, 0, 7), W(cs_tc, 0, 30), W(cs_tc, 0, 90)
    f["topup_amt_30d"] = W(cs_ta, 0, 30)
    f["topup_avg_amt_30d"] = np.where(f["topup_cnt_30d"] > 0, f["topup_amt_30d"] / f["topup_cnt_30d"].replace(0, np.nan), np.nan)
    f["topup_cnt_trend"] = f["topup_cnt_30d"] / (W(cs_tc, 30, 90) / 2 + 0.5)
    f["topup_weekly_cv_12w"] = weekly_cv(cs_tc)
    f["days_since_last_topup"] = last_gap(tc)
    f["gsm_active_days_30d"], f["gsm_active_days_90d"] = W(cs_act, 0, 30), W(cs_act, 0, 90)
    f["days_since_last_activity"] = last_gap(act)
    f["voice_min_30d"] = W(cs_voice, 0, 30)
    f["data_mb_30d"] = W(cs_data, 0, 30)
    f["data_mb_trend"] = f["data_mb_30d"] / (W(cs_data, 0, 90) / 3 + 1)
    f["sms_cnt_30d"] = W(cs_sms, 0, 30)
    del tc, act

    # ---- M-PESA ----
    ones = np.ones(len(mp))
    cnt_all = _dense(cidx, date0, D, mp, "txn_date", ones); cs_cnt = _cum(cnt_all)
    is_in = (mp["direction"] == "IN").to_numpy()
    cs_in = _cum(_dense(cidx, date0, D, mp, "txn_date", np.where(is_in, mp["amount"], 0.0)))
    cs_out = _cum(_dense(cidx, date0, D, mp, "txn_date", np.where(~is_in, mp["amount"], 0.0)))
    cs_amt = _cum(_dense(cidx, date0, D, mp, "txn_date", mp["amount"]))
    f["mp_txn_cnt_7d"], f["mp_txn_cnt_30d"], f["mp_txn_cnt_90d"] = W(cs_cnt, 0, 7), W(cs_cnt, 0, 30), W(cs_cnt, 0, 90)
    f["mp_inflow_30d"], f["mp_outflow_30d"] = W(cs_in, 0, 30), W(cs_out, 0, 30)
    f["inflow_outflow_ratio_30d"] = (f["mp_inflow_30d"] + 1) / (f["mp_outflow_30d"] + 1)
    f["mp_inflow_90d"] = W(cs_in, 0, 90)
    f["mp_inflow_weekly_cv_12w"] = weekly_cv(cs_in)
    for col, typ in [("mp_cashin_cnt_30d", "CASH_IN"), ("mp_cashout_cnt_30d", "CASH_OUT"),
                     ("mp_p2p_send_cnt_30d", "P2P_SEND"), ("mp_p2p_recv_cnt_30d", "P2P_RECEIVE"),
                     ("mp_billpay_cnt_30d", "BILL_PAY"), ("mp_merchant_cnt_30d", "MERCHANT")]:
        sel = (mp["txn_type"] == typ).to_numpy()
        f[col] = W(_cum(_dense(cidx, date0, D, mp, "txn_date", sel.astype(float))), 0, 30)
    f["mp_avg_txn_amt_30d"] = np.where(f["mp_txn_cnt_30d"] > 0, W(cs_amt, 0, 30) / f["mp_txn_cnt_30d"].replace(0, np.nan), np.nan)
    f["days_since_last_mpesa_txn"] = last_gap(cnt_all)
    del cnt_all

    # ---- wallet balance ----
    bal = _dense(cidx, date0, D, wal, "date", wal["balance_etb"])
    cs_b, cs_b2 = _cum(bal), _cum(bal**2)
    cs_low = _cum((bal < 100).astype(float) * _dense(cidx, date0, D, wal, "date", np.ones(len(wal))))
    f["wallet_bal_mean_30d"] = W(cs_b, 0, 30) / 30
    m90 = W(cs_b, 0, 90) / 90
    f["wallet_bal_std_90d"] = np.sqrt(np.maximum(W(cs_b2, 0, 90) / 90 - m90**2, 0))
    f["wallet_bal_trend_7v90"] = (W(cs_b, 0, 7) / 7) / (m90 + 1)
    f["wallet_low_bal_days_30d"] = W(cs_low, 0, 30)

    # ---- static + loan ----
    c = cust.iloc[lc].reset_index(drop=True)
    f["tenure_months"] = c["tenure_months_at_start"].to_numpy() + ld / 30.4
    f["is_smartphone"] = (c["device_type"] == "SMARTPHONE").astype(int).to_numpy()
    f["region"] = c["region"].to_numpy()
    f["loan_amount"] = loans["principal_etb"].to_numpy()
    f["loan_to_inflow_ratio"] = f["loan_amount"] / (f["mp_inflow_90d"] / 3 + 1)

    L = loans[["loan_id", "customer_id", "disbursement_date", "due_date", "repaid_date", "default_flag", "outcome_date"]]
    prior = L[["loan_id", "customer_id", "disbursement_date"]].merge(
        L.drop(columns="loan_id").add_suffix("_p"), left_on="customer_id", right_on="customer_id_p")
    # only loans whose outcome was already known strictly before this snapshot
    prior = prior[prior["outcome_date_p"] < prior["disbursement_date"]].copy()
    prior["late_p"] = ((prior["default_flag_p"] == 1) | (prior["repaid_date_p"] > prior["due_date_p"])).astype(int)
    prior["lag_p"] = (prior["repaid_date_p"] - prior["disbursement_date_p"]).dt.days
    prior = prior.sort_values(["loan_id", "disbursement_date_p"])
    g = prior.groupby("loan_id")
    agg = pd.DataFrame({
        "prior_loan_cnt": g.size(),
        "prior_late_cnt": g["late_p"].sum(),
        "prior_default_cnt": g["default_flag_p"].sum(),
        "last_loan_late": g["late_p"].last(),
        "days_since_last_loan": (g["disbursement_date"].first() - g["disbursement_date_p"].max()).dt.days,
        "avg_prior_repay_lag": g["lag_p"].mean(),
    })
    f = f.merge(agg, left_on="loan_id", right_index=True, how="left")
    for col in ["prior_loan_cnt", "prior_late_cnt", "prior_default_cnt"]:
        f[col] = f[col].fillna(0).astype(int)

    f.insert(2, "snapshot_date", loans["disbursement_date"].to_numpy())
    f.insert(3, "vintage", loans["disbursement_date"].dt.to_period("M").astype(str).to_numpy())
    f.insert(4, "default", loans["default_flag"].to_numpy())
    f.insert(5, "fee_etb", loans["fee_etb"].to_numpy())
    f = f[META + FEATURES].replace([np.inf, -np.inf], np.nan)
    return f
