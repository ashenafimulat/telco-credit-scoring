# Model card: telco credit scoring (v1)

**Status:** portfolio / demonstration model trained on **synthetic** data. Not for real lending decisions.

## Intended use
Estimate the probability that a prepaid subscriber does not repay a 15-day airtime/cash advance within 30 days
of disbursement (`PD`), to support an approve/decline decision and a credit-limit tier.

## Models
| | Role | Notes |
|---|---|---|
| WOE logistic-regression scorecard | **champion** | 20 features after IV (>=0.02), correlation (<0.8) and coefficient-sign screening; points scaled to 600 at 20:1 good:bad odds, 50 points to double the odds; reason codes per applicant |
| Gradient boosting (sklearn-hist-gbm) | challenger | Platt-calibrated on the validation set |

Champion rule: the GBM must beat the scorecard by >= 0.01 validation AUC, otherwise the explainable scorecard is used.

## Data and split (by disbursement date, no random shuffling)
| Split | Window | Loans |
|---|---|---|
| Train | up to 2025-10-31 | 11,476 |
| Validation | up to 2025-12-31 | 3,178 |
| Out-of-time test | after 2025-12-31 (contains the simulated airtime-promo / salary-delay shock) | 4,746 |

Point-in-time features: every window ends strictly before the disbursement date; `tests/test_point_in_time.py` proves it.

## Performance
| Model / split | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| Scorecard, validation | 0.733 | 0.465 | 0.364 | 0.088 |
| Scorecard, **test** | 0.695 | 0.390 | 0.279 | 0.125 |
| GBM, validation | 0.731 | 0.462 | 0.351 | 0.088 |
| GBM, **test** | 0.694 | 0.387 | 0.282 | 0.126 |

Test default rate: 15.8% (train: 10.8%). Per-vintage AUC, calibration and decile tables are in `reports/`.

## Decision policy (from `src/strategy.py`)
Applying a PD cut-off of 0.12 (chosen on validation) to the out-of-time test set changes approvals by -36 pts, cuts the default rate among approved loans from 15.8% to 10.2% and moves net profit from -40,954 to -2,946 ETB.

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

## Retraining history
Trigger: 2026-03: score PSI 0.027 (alert 0.25), AUC 0.689 vs 0.733 on validation; top drifting feature topup_avg_amt_30d (CSI 1.95)  
Candidate v2 vs v1 on 2026-01-31 < snapshot <= 2026-03-28: AUC 0.697 vs 0.697. Action: promoted v2.
