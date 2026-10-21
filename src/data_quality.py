"""Great-Expectations-style data quality checks (dependency-free).

Each check returns a dict {name, table, status: PASS|WARN|FAIL, detail}. `run_suite` runs the suite
for every raw/clean table; `run_feature_suite` covers the model feature table. Reports are written
as JSON + Markdown so they can be attached to a pipeline run or reviewed in a PR.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SCHEMA = {
    "customers": {"customer_id": "object", "region": "object", "device_type": "object",
                  "tenure_months_at_start": "int"},
    "gsm_daily": {"customer_id": "object", "date": "datetime", "topup_count": "int",
                  "topup_amount": "float", "voice_min": "float", "data_mb": "float", "sms_count": "int"},
    "mpesa_txn": {"txn_id": "object", "customer_id": "object", "txn_date": "datetime",
                  "txn_type": "object", "direction": "object", "amount": "float"},
    "wallet_daily": {"customer_id": "object", "date": "datetime", "balance_etb": "float"},
    "loans": {"loan_id": "object", "customer_id": "object", "disbursement_date": "datetime",
              "principal_etb": "int", "fee_etb": "float", "due_date": "datetime",
              "repaid_date": "datetime", "default_flag": "int", "outcome_date": "datetime"},
}
KEYS = {"customers": ["customer_id"], "gsm_daily": ["customer_id", "date"], "mpesa_txn": ["txn_id"],
        "wallet_daily": ["customer_id", "date"], "loans": ["loan_id"]}
NON_NEGATIVE = {"gsm_daily": ["topup_count", "topup_amount", "voice_min", "data_mb", "sms_count"],
                "mpesa_txn": ["amount"], "wallet_daily": ["balance_etb"], "loans": ["principal_etb", "fee_etb"]}
MAX_NULL = {"customers": {"region": 0.01}, "loans": {"repaid_date": 0.30}}
VALID_VALUES = {"mpesa_txn": {"direction": {"IN", "OUT"},
                              "txn_type": {"CASH_IN", "CASH_OUT", "P2P_SEND", "P2P_RECEIVE", "BILL_PAY",
                                           "MERCHANT", "SALARY_IN"}},
                "customers": {"device_type": {"SMARTPHONE", "FEATURE_PHONE"}}}


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    return "object"


def _r(name, table, ok, detail, severity="FAIL"):
    return {"name": name, "table": table, "status": "PASS" if ok else severity, "detail": detail}


def check_schema(df, table, expected):
    """Schema drift: missing / unexpected columns and dtype changes (int is accepted where float expected)."""
    out = []
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    out.append(_r("schema_columns", table, not missing, f"missing={missing} unexpected={extra}"))
    bad = {}
    for c, kind in expected.items():
        if c in df.columns:
            got = _kind(df[c])
            if got != kind and not (kind == "float" and got == "int") and not (kind == "datetime" and df[c].isna().all()):
                bad[c] = f"expected {kind}, got {got}"
    out.append(_r("schema_dtypes", table, not bad, str(bad) if bad else "ok"))
    if extra and not missing:
        out[0]["status"] = "WARN"
    return out


def check_nulls(df, table, max_null):
    out = []
    for c in df.columns:
        frac = float(df[c].isna().mean())
        limit = max_null.get(c, 0.0)
        out.append(_r(f"nulls[{c}]", table, frac <= limit, f"null_frac={frac:.4f} limit={limit}"))
    return out


def check_unique(df, table, keys):
    d = int(df.duplicated(keys).sum())
    return [_r(f"unique[{','.join(keys)}]", table, d == 0, f"duplicate_rows={d}")]


def check_non_negative(df, table, cols):
    out = []
    for c in cols:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            n = int((x < 0).sum()) + int(x.isna().sum() - df[c].isna().sum())  # negatives + non-numeric
            out.append(_r(f"non_negative[{c}]", table, n == 0, f"negative_or_non_numeric_rows={n}"))
    return out


def check_valid_values(df, table, spec):
    out = []
    for c, allowed in spec.items():
        if c in df.columns:
            bad = df[c].dropna()
            bad = bad[~bad.isin(allowed)]
            out.append(_r(f"valid_values[{c}]", table, bad.empty, f"invalid_rows={len(bad)}"))
    return out


def check_outliers(df, table, col, k=3.0, max_frac=0.0005):
    """Flag positive values far above the bulk, measured on a log scale (money is heavy-tailed)."""
    x = np.log1p(df[col][df[col] > 0].dropna())
    if x.empty:
        return []
    q1, q3 = x.quantile([0.25, 0.75])
    frac = float((x > q3 + k * (q3 - q1)).mean())
    return [_r(f"outliers[{col}]", table, frac <= max_frac, f"frac_above_{k}xIQR_log={frac:.5f}", "WARN")]


def check_fk(df, table, col, ref, ref_name):
    n = int((~df[col].isin(ref)).sum())
    return [_r(f"fk[{col}->{ref_name}]", table, n == 0, f"orphan_rows={n}")]


def check_date_order(df, table, early, late):
    d = df[[early, late]].dropna()
    n = int((d[late] < d[early]).sum())
    return [_r(f"date_order[{early}<={late}]", table, n == 0, f"violations={n}")]


def check_completeness(wallet, table, n_customers, n_days):
    expected = n_customers * n_days
    return [_r("grid_completeness", table, len(wallet) >= 0.999 * expected,
               f"rows={len(wallet)} expected~{expected}", "WARN")]


def run_suite(tables: dict[str, pd.DataFrame], stage: str = "raw") -> dict:
    res = []
    for name, df in tables.items():
        res += check_schema(df, name, EXPECTED_SCHEMA[name])
        res += check_nulls(df, name, MAX_NULL.get(name, {}))
        res += check_unique(df, name, KEYS[name])
        res += check_non_negative(df, name, NON_NEGATIVE.get(name, []))
        res += check_valid_values(df, name, VALID_VALUES.get(name, {}))
    cust = tables["customers"]["customer_id"]
    for name in ["gsm_daily", "mpesa_txn", "wallet_daily", "loans"]:
        res += check_fk(tables[name], name, "customer_id", cust, "customers")
    res += check_outliers(tables["mpesa_txn"], "mpesa_txn", "amount")
    res += check_outliers(tables["gsm_daily"], "gsm_daily", "topup_amount")
    res += check_date_order(tables["loans"], "loans", "disbursement_date", "repaid_date")
    res += check_date_order(tables["loans"], "loans", "disbursement_date", "due_date")
    w = tables["wallet_daily"]
    res += check_completeness(w, "wallet_daily", w["customer_id"].nunique(), w["date"].nunique())
    return _summarise(res, stage)


def run_feature_suite(feats: pd.DataFrame, feature_cols: list[str]) -> dict:
    res = []
    res += check_unique(feats, "features", ["loan_id"])
    num = feats[feature_cols].select_dtypes("number")
    inf = int(np.isinf(num.to_numpy(dtype=float)).sum())
    res.append(_r("no_infinite_values", "features", inf == 0, f"inf_cells={inf}"))
    for c in feature_cols:
        frac = float(feats[c].isna().mean())
        res.append(_r(f"nulls[{c}]", "features", frac <= 0.30, f"null_frac={frac:.3f}", "WARN"))
    const = [c for c in num.columns if num[c].nunique() <= 1]
    res.append(_r("no_constant_features", "features", not const, f"constant={const}", "WARN"))
    res.append(_r("label_binary", "features", set(feats["default"].unique()) <= {0, 1}, "default in {0,1}"))
    return _summarise(res, "features")


def _summarise(results: list[dict], stage: str) -> dict:
    counts = {s: sum(r["status"] == s for r in results) for s in ("PASS", "WARN", "FAIL")}
    return {"stage": stage, "counts": counts, "results": results}


def write_report(report: dict, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2))
    c = report["counts"]
    lines = [f"# Data quality report: {report['stage']}", "",
             f"**PASS {c['PASS']} | WARN {c['WARN']} | FAIL {c['FAIL']}**", "",
             "| Status | Table | Check | Detail |", "|---|---|---|---|"]
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    for r in sorted(report["results"], key=lambda r: order[r["status"]]):
        if r["status"] != "PASS" or r["name"].startswith(("unique", "fk", "schema")):
            lines.append(f"| {r['status']} | {r['table']} | `{r['name']}` | {r['detail']} |")
    (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n")
