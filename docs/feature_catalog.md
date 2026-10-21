# Feature catalog

Windows are `[snapshot - W, snapshot)`; prior-loan features use only loans whose outcome was known before the snapshot.
IV = information value on the training window; *selected* = used by the scorecard.

| Feature | Type | Description | IV | In scorecard |
|---|---|---|---|---|
| `topup_cnt_7d` | numeric | Airtime top-ups in the last 7 days | 0.052 | yes |
| `topup_cnt_30d` | numeric | Airtime top-ups in the last 30 days | 0.211 |  |
| `topup_cnt_90d` | numeric | Airtime top-ups in the last 90 days | 0.330 | yes |
| `topup_amt_30d` | numeric | Airtime top-up value in the last 30 days (ETB) | 0.289 |  |
| `topup_avg_amt_30d` | numeric | Average top-up value in the last 30 days (ETB) | 0.240 | yes |
| `topup_cnt_trend` | numeric | Top-ups in last 30 days vs the average of the prior 60 days | 0.069 |  |
| `topup_weekly_cv_12w` | numeric | Irregularity of weekly top-ups over 12 weeks (coefficient of variation) | 0.251 | yes |
| `days_since_last_topup` | numeric | Days since the last airtime top-up | 0.124 |  |
| `gsm_active_days_30d` | numeric | Days with any GSM activity in the last 30 days | 0.215 | yes |
| `gsm_active_days_90d` | numeric | Days with any GSM activity in the last 90 days | 0.288 | yes |
| `days_since_last_activity` | numeric | Days since any GSM activity | 0.001 |  |
| `voice_min_30d` | numeric | Voice minutes in the last 30 days | 0.219 | yes |
| `data_mb_30d` | numeric | Mobile data used in the last 30 days (MB) | 0.132 | yes |
| `data_mb_trend` | numeric | Data use in last 30 days vs 90-day average | 0.058 | yes |
| `sms_cnt_30d` | numeric | SMS sent in the last 30 days | 0.364 | yes |
| `mp_txn_cnt_7d` | numeric | M-PESA transactions in the last 7 days | 0.167 | yes |
| `mp_txn_cnt_30d` | numeric | M-PESA transactions in the last 30 days | 0.211 |  |
| `mp_txn_cnt_90d` | numeric | M-PESA transactions in the last 90 days | 0.218 |  |
| `mp_inflow_30d` | numeric | Wallet inflows in the last 30 days (ETB) | 0.235 |  |
| `mp_outflow_30d` | numeric | Wallet outflows in the last 30 days (ETB) | 0.223 |  |
| `inflow_outflow_ratio_30d` | numeric | Smoothed wallet inflow/outflow ratio, last 30 days | 0.057 |  |
| `mp_inflow_90d` | numeric | Wallet inflows in the last 90 days (ETB) | 0.263 |  |
| `mp_inflow_weekly_cv_12w` | numeric | Income volatility: weekly inflow coefficient of variation over 12 weeks | 0.254 | yes |
| `mp_cashin_cnt_30d` | numeric | Cash-in transactions in the last 30 days | 0.233 | yes |
| `mp_cashout_cnt_30d` | numeric | Cash-out transactions in the last 30 days | 0.113 |  |
| `mp_p2p_send_cnt_30d` | numeric | P2P transfers sent in the last 30 days | 0.154 | yes |
| `mp_p2p_recv_cnt_30d` | numeric | P2P transfers received in the last 30 days | 0.164 | yes |
| `mp_billpay_cnt_30d` | numeric | Bill payments in the last 30 days | 0.072 |  |
| `mp_merchant_cnt_30d` | numeric | Merchant payments in the last 30 days | 0.079 |  |
| `mp_avg_txn_amt_30d` | numeric | Average M-PESA transaction value, last 30 days (ETB) | 0.138 | yes |
| `days_since_last_mpesa_txn` | numeric | Days since the last M-PESA transaction | 0.060 |  |
| `wallet_bal_mean_30d` | numeric | Average wallet balance, last 30 days (ETB) | 0.186 | yes |
| `wallet_bal_std_90d` | numeric | Wallet balance volatility over 90 days (ETB, population std) | 0.193 |  |
| `wallet_bal_trend_7v90` | numeric | Average balance last 7 days vs last 90 days | 0.040 |  |
| `wallet_low_bal_days_30d` | numeric | Days in the last 30 with wallet balance below 100 ETB | 0.065 |  |
| `tenure_months` | numeric | Months since registration at the snapshot date | 0.006 |  |
| `is_smartphone` | numeric | Customer uses a smartphone (1) or feature phone (0) | 0.001 |  |
| `region` | categorical | Customer region | 0.004 |  |
| `loan_amount` | numeric | Principal of the loan being scored (ETB) | 0.062 |  |
| `loan_to_inflow_ratio` | numeric | Loan principal relative to average monthly wallet inflow (90d) | 0.165 | yes |
| `prior_loan_cnt` | numeric | Loans resolved before this snapshot | 0.002 |  |
| `prior_late_cnt` | numeric | Prior loans repaid late or defaulted | 0.030 | yes |
| `prior_default_cnt` | numeric | Prior loans that defaulted | 0.001 |  |
| `last_loan_late` | numeric | Was the most recent prior loan late or defaulted (1/0) | 0.000 |  |
| `days_since_last_loan` | numeric | Days since the most recent prior loan was disbursed | 0.001 |  |
| `avg_prior_repay_lag` | numeric | Average days to repay on prior repaid loans | 0.029 | yes |
