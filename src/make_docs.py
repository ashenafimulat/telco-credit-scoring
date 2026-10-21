"""Generate docs/ (model card, feature catalog, data dictionary) from the latest pipeline outputs."""
from __future__ import annotations

import json

import pandas as pd

from .config import ROOT, layer_dir, load_config
from .features import CATEGORICAL, FEATURE_DESCRIPTIONS
from .registry import load_metadata
from .train import MODEL_NAME

DATA_DICT = {
    "customers": ("One row per subscriber (static attributes).", {
        "customer_id": "Unique subscriber id", "region": "Region of registration (nullable in raw data)",
        "device_type": "SMARTPHONE or FEATURE_PHONE", "tenure_months_at_start": "Months since registration at history start"}),
    "gsm_daily": ("One row per customer-day with any GSM activity (missing day = no activity).", {
        "customer_id": "FK to customers", "date": "Activity date", "topup_count": "Airtime recharges that day",
        "topup_amount": "Airtime recharge value (ETB)", "voice_min": "Voice minutes (in+out)", "data_mb": "Mobile data (MB)", "sms_count": "SMS sent"}),
    "mpesa_txn": ("One row per mobile-money transaction.", {
        "txn_id": "Unique transaction id", "customer_id": "FK to customers", "txn_date": "Transaction date",
        "txn_type": "CASH_IN, CASH_OUT, P2P_SEND, P2P_RECEIVE, BILL_PAY, MERCHANT, SALARY_IN",
        "direction": "IN (credits wallet) or OUT (debits wallet)", "amount": "Amount (ETB)"}),
    "wallet_daily": ("End-of-day wallet balance for every customer-day (overdraft not modelled; floored at 0).", {
        "customer_id": "FK to customers", "date": "Balance date", "balance_etb": "End-of-day balance (ETB)"}),
    "loans": ("One row per 15-day advance. Target = default_flag.", {
        "loan_id": "Unique loan id", "customer_id": "FK to customers", "disbursement_date": "Snapshot date for scoring",
        "principal_etb": "Amount advanced (ETB)", "fee_etb": "Fee earned if repaid (ETB)", "due_date": "Disbursement + 15 days",
        "repaid_date": "Date fully repaid (null = never repaid)", "default_flag": "1 if not repaid 30 days after disbursement (target)",
        "outcome_date": "Date the outcome became known (repaid_date, or disbursement + 30 days)"}),
}


def main() -> None:
    cfg = load_config()
    reports, docs = layer_dir(cfg, "reports"), ROOT / "docs"
    docs.mkdir(exist_ok=True)
    m = json.loads((reports / "metrics.json").read_text())
    meta = load_metadata(cfg, MODEL_NAME, m["version"])          # the version these metrics were computed for
    rl = reports / "retrain_log.json"
    retrain = json.loads(rl.read_text()) if rl.exists() else None
    st = json.loads((reports / "strategy.json").read_text())
    iv = pd.read_csv(reports / "feature_iv.csv").set_index("feature")

    # ---- data dictionary ----
    out = ["# Data dictionary", "", "All data is synthetic (see `src/generate_data.py`). Currency is ETB-like and illustrative.", ""]
    for t, (desc, cols) in DATA_DICT.items():
        out += [f"## `{t}`", desc, "", "| Column | Description |", "|---|---|"] + [f"| `{c}` | {d} |" for c, d in cols.items()] + [""]
    (docs / "data_dictionary.md").write_text("\n".join(out))

    # ---- feature catalog ----
    rows = ["# Feature catalog", "", "Windows are `[snapshot - W, snapshot)`; prior-loan features use only loans whose outcome was known before the snapshot.",
            "IV = information value on the training window; *selected* = used by the scorecard.", "",
            "| Feature | Type | Description | IV | In scorecard |", "|---|---|---|---|---|"]
    for f, d in FEATURE_DESCRIPTIONS.items():
        ivv = f"{iv.loc[f, 'iv']:.3f}" if f in iv.index else ""
        sel = "yes" if f in iv.index and bool(iv.loc[f, "selected"]) else ""
        rows.append(f"| `{f}` | {'categorical' if f in CATEGORICAL else 'numeric'} | {d} | {ivv} | {sel} |")
    (docs / "feature_catalog.md").write_text("\n".join(rows) + "\n")

    # ---- model card ----
    c, sc, gb = m["models"]["champion"], m["models"]["scorecard"], m["models"]["gbm"]
    fmt = lambda d: f"{d['auc']:.3f} | {d['gini']:.3f} | {d['ks']:.3f} | {d['brier']:.3f}"  # noqa: E731
    card = f"""# Model card: telco credit scoring ({meta['version']})

**Status:** portfolio / demonstration model trained on **synthetic** data. Not for real lending decisions.

## Intended use
Estimate the probability that a prepaid subscriber does not repay a 15-day airtime/cash advance within 30 days
of disbursement (`PD`), to support an approve/decline decision and a credit-limit tier.

## Models
| | Role | Notes |
|---|---|---|
| WOE logistic-regression scorecard | {'**champion**' if m['champion'] == 'scorecard' else 'challenger'} | {len(meta['scorecard_features'])} features after IV (>=0.02), correlation (<0.8) and coefficient-sign screening; points scaled to 600 at 20:1 good:bad odds, 50 points to double the odds; reason codes per applicant |
| Gradient boosting ({m['gbm_engine']}) | {'**champion**' if m['champion'] == 'gbm' else 'challenger'} | Platt-calibrated on the validation set |

Champion rule: the GBM must beat the scorecard by >= 0.01 validation AUC, otherwise the explainable scorecard is used.

## Data and split (by disbursement date, no random shuffling)
| Split | Window | Loans |
|---|---|---|
| Train | up to {meta['split']['train_end']} | {meta['n']['train']:,} |
| Validation | up to {meta['split']['valid_end']} | {meta['n']['valid']:,} |
| Out-of-time test | after {meta['split']['valid_end']} (contains the simulated airtime-promo / salary-delay shock) | {meta['n']['test']:,} |

Point-in-time features: every window ends strictly before the disbursement date; `tests/test_point_in_time.py` proves it.

## Performance
| Model / split | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| Scorecard, validation | {fmt(sc['valid'])} |
| Scorecard, **test** | {fmt(sc['test'])} |
| GBM, validation | {fmt(gb['valid'])} |
| GBM, **test** | {fmt(gb['test'])} |

Test default rate: {c['test']['bad_rate']:.1%} (train: {c['train']['bad_rate']:.1%}). Per-vintage AUC, calibration and decile tables are in `reports/`.

## Decision policy (from `src/strategy.py`)
{st['headline']}

## Limitations and risks
* Synthetic data: relationships are simulated, so absolute performance says nothing about a real operator. The value is the method.
* The simulated shock lowers performance on the test window on purpose; the monitoring report shows the drift and the retraining response.
* No reject inference: outcomes are observed only for approved loans in real life.
* Fairness: no protected attributes are used, but `region` and device type can proxy for them. Review before real use.
* Wallet balances are floored at zero and not exactly consistent with transactions (documented simplification).
* Explanations: scorecard reason codes give adverse-action reasons; permutation importance (or SHAP when installed) explains the GBM.

## Monitoring and retraining
PSI/CSI by vintage, AUC drop vs validation, approval-rate and default-rate trends (`reports/monitoring_report.md`).
Thresholds: PSI 0.10 watch / 0.25 alert; AUC drop 0.03. Monthly scheduled job: `.github/workflows/retrain.yml`.
"""
    if retrain and retrain.get("trigger"):
        card += (f"\n## Retraining history\nTrigger: {retrain['trigger']}  \nCandidate {retrain.get('candidate_version')} vs {retrain['current_version']} on "
                 f"{retrain.get('holdout_window')}: AUC {retrain.get('holdout_auc_candidate', float('nan')):.3f} vs "
                 f"{retrain.get('holdout_auc_current', float('nan')):.3f}. Action: {retrain['action']}.\n")
    (docs / "model_card.md").write_text(card)
    print("docs written")


if __name__ == "__main__":
    main()
