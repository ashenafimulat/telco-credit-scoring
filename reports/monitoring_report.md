# Monitoring report: model v1 (scorecard)

**Retrain recommended: YES**  
2026-03: score PSI 0.027 (alert 0.25), AUC 0.689 vs 0.733 on validation; top drifting feature topup_avg_amt_30d (CSI 1.95)

| Vintage | Split | Loans | Score PSI | Max CSI (feature) | AUC | Default rate | Approval @ policy | Status |
|---|---|---|---|---|---|---|---|---|
| 2025-11 | valid | 1615 | 0.008 | 0.18 (prior_late_cnt) | 0.748 | 12.0% | 69% | GREEN |
| 2025-12 | valid | 1563 | 0.009 | 0.25 (prior_late_cnt) | 0.715 | 9.5% | 66% | AMBER |
| 2026-01 | test | 1573 | 0.012 | 0.52 (topup_avg_amt_30d) | 0.690 | 15.1% | 66% | RED |
| 2026-02 | test | 1667 | 0.032 | 1.92 (topup_avg_amt_30d) | 0.704 | 17.0% | 62% | AMBER |
| 2026-03 | test | 1506 | 0.027 | 1.95 (topup_avg_amt_30d) | 0.689 | 15.2% | 64% | RED |

## Top drifting features in 2026-03 (CSI)

| Feature | CSI |
|---|---|
| topup_avg_amt_30d | 1.945 |
| topup_cnt_90d | 0.697 |
| mp_inflow_weekly_cv_12w | 0.409 |
| topup_weekly_cv_12w | 0.400 |
| prior_late_cnt | 0.384 |
| mp_cashin_cnt_30d | 0.315 |
| mp_p2p_recv_cnt_30d | 0.264 |
| topup_cnt_7d | 0.211 |

PSI/CSI guide: < 0.10 stable, 0.10-0.25 watch, > 0.25 significant shift.
