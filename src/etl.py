"""ETL: raw -> clean -> features.

    data/raw      immutable generator / source-system output (contains defects on purpose)
    data/clean    de-duplicated, validated, outlier-capped tables  (+ cleaning log)
    data/features point-in-time model features, one row per loan at disbursement

Every cleaning rule is logged with the number of rows it touched.
"""
from __future__ import annotations

import json

import pandas as pd

from . import data_quality as dq
from .config import layer_dir, load_config
from .features import build_features
from .utils import get_logger, read_layer, write_layer, write_table

log = get_logger("etl")


def clean_tables(raw: dict[str, pd.DataFrame], cap_quantile: float = 0.9995) -> tuple[dict, list[dict]]:
    t = {k: v.copy() for k, v in raw.items()}
    steps: list[dict] = []

    def log_step(table, rule, n):
        steps.append({"table": table, "rule": rule, "rows_affected": int(n)})

    c = t["customers"]
    n = c.duplicated("customer_id").sum(); c = c.drop_duplicates("customer_id"); log_step("customers", "dedupe customer_id", n)
    n = c["region"].isna().sum(); c["region"] = c["region"].fillna("UNKNOWN").str.upper(); log_step("customers", "region null -> UNKNOWN", n)
    t["customers"] = c.reset_index(drop=True)

    g = t["gsm_daily"]
    n = g.duplicated(["customer_id", "date"]).sum(); g = g.drop_duplicates(["customer_id", "date"]); log_step("gsm_daily", "dedupe (customer_id,date)", n)
    cols = ["topup_count", "topup_amount", "voice_min", "data_mb", "sms_count"]
    bad = (g[cols] < 0).any(axis=1); log_step("gsm_daily", "drop negative usage", bad.sum()); g = g[~bad]
    t["gsm_daily"] = g.reset_index(drop=True)

    m = t["mpesa_txn"]
    n = m.duplicated("txn_id").sum(); m = m.drop_duplicates("txn_id"); log_step("mpesa_txn", "dedupe txn_id", n)
    bad = m["amount"] <= 0; log_step("mpesa_txn", "drop non-positive amount", bad.sum()); m = m[~bad].copy()
    caps = m.groupby("txn_type")["amount"].transform(lambda s: s.quantile(cap_quantile))
    over = m["amount"] > caps; log_step("mpesa_txn", f"winsorise amount at p{cap_quantile * 100:.2f} per txn_type", over.sum())
    m.loc[over, "amount"] = caps[over]
    t["mpesa_txn"] = m.reset_index(drop=True)

    w = t["wallet_daily"]
    n = w.duplicated(["customer_id", "date"]).sum(); w = w.drop_duplicates(["customer_id", "date"]); log_step("wallet_daily", "dedupe (customer_id,date)", n)
    bad = w["balance_etb"] < 0; log_step("wallet_daily", "drop negative balance", bad.sum())
    t["wallet_daily"] = w[~bad].reset_index(drop=True)

    L = t["loans"]
    n = L.duplicated("loan_id").sum(); L = L.drop_duplicates("loan_id"); log_step("loans", "dedupe loan_id", n)
    bad = L["repaid_date"].notna() & (L["repaid_date"] < L["disbursement_date"]); log_step("loans", "drop repaid before disbursed", bad.sum())
    t["loans"] = L[~bad].reset_index(drop=True)
    return t, steps


def run(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    raw = read_layer(layer_dir(cfg, "raw"))
    reports = layer_dir(cfg, "reports")

    rep_raw = dq.run_suite(raw, "raw")
    dq.write_report(rep_raw, reports, "dq_raw")
    log.info("DQ raw: %s", rep_raw["counts"])

    clean, steps = clean_tables(raw)
    (reports / "cleaning_log.json").write_text(json.dumps(steps, indent=2))
    rep_clean = dq.run_suite(clean, "clean")
    dq.write_report(rep_clean, reports, "dq_clean")
    log.info("DQ clean: %s", rep_clean["counts"])
    if rep_clean["counts"]["FAIL"]:
        raise SystemExit("Clean layer still has FAIL checks; see reports/dq_clean.md")
    write_layer(clean, layer_dir(cfg, "clean"))

    feats = build_features(clean, cfg)
    from .features import FEATURES
    rep_f = dq.run_feature_suite(feats, FEATURES)
    dq.write_report(rep_f, reports, "dq_features")
    log.info("DQ features: %s", rep_f["counts"])
    write_table(feats, layer_dir(cfg, "features") / "features.csv.gz")
    log.info("features: %s rows x %s cols", *feats.shape)


if __name__ == "__main__":
    run()
