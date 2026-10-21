-- DuckDB version of the core point-in-time features (parity-tested against src/features.py).
-- Tables expected: loans, gsm_daily, mpesa_txn, wallet_daily.
-- Every window is [snapshot - W days, snapshot): the join condition `date < disbursement_date`
-- is what prevents leakage. Smoothed ratio (x+1)/(y+1) avoids divide-by-zero.
WITH g AS (
  SELECT l.loan_id,
    COALESCE(SUM(g.topup_count)  FILTER (WHERE g.date >= l.disbursement_date - INTERVAL 7 DAY), 0)  AS topup_cnt_7d,
    COALESCE(SUM(g.topup_count)  FILTER (WHERE g.date >= l.disbursement_date - INTERVAL 30 DAY), 0) AS topup_cnt_30d,
    COALESCE(SUM(g.topup_count), 0)                                                                  AS topup_cnt_90d,
    COALESCE(SUM(g.topup_amount) FILTER (WHERE g.date >= l.disbursement_date - INTERVAL 30 DAY), 0) AS topup_amt_30d,
    COALESCE(SUM(g.voice_min)    FILTER (WHERE g.date >= l.disbursement_date - INTERVAL 30 DAY), 0) AS voice_min_30d,
    COALESCE(SUM(g.data_mb)      FILTER (WHERE g.date >= l.disbursement_date - INTERVAL 30 DAY), 0) AS data_mb_30d,
    COALESCE(SUM(g.sms_count)    FILTER (WHERE g.date >= l.disbursement_date - INTERVAL 30 DAY), 0) AS sms_cnt_30d,
    COUNT(g.date)                FILTER (WHERE g.date >= l.disbursement_date - INTERVAL 30 DAY)     AS gsm_active_days_30d
  FROM loans l
  LEFT JOIN gsm_daily g
    ON g.customer_id = l.customer_id
   AND g.date <  l.disbursement_date
   AND g.date >= l.disbursement_date - INTERVAL 90 DAY
  GROUP BY l.loan_id
),
m AS (
  SELECT l.loan_id,
    COUNT(t.txn_id) FILTER (WHERE t.txn_date >= l.disbursement_date - INTERVAL 30 DAY) AS mp_txn_cnt_30d,
    COALESCE(SUM(t.amount) FILTER (WHERE t.direction = 'IN'  AND t.txn_date >= l.disbursement_date - INTERVAL 30 DAY), 0) AS mp_inflow_30d,
    COALESCE(SUM(t.amount) FILTER (WHERE t.direction = 'OUT' AND t.txn_date >= l.disbursement_date - INTERVAL 30 DAY), 0) AS mp_outflow_30d
  FROM loans l
  LEFT JOIN mpesa_txn t
    ON t.customer_id = l.customer_id
   AND t.txn_date <  l.disbursement_date
   AND t.txn_date >= l.disbursement_date - INTERVAL 30 DAY
  GROUP BY l.loan_id
),
last_txn AS (  -- full history, not windowed
  SELECT l.loan_id,
         DATE_DIFF('day', MAX(t.txn_date), l.disbursement_date) AS days_since_last_mpesa_txn
  FROM loans l
  LEFT JOIN mpesa_txn t ON t.customer_id = l.customer_id AND t.txn_date < l.disbursement_date
  GROUP BY l.loan_id, l.disbursement_date
),
w AS (
  SELECT l.loan_id,
    AVG(w.balance_etb) FILTER (WHERE w.date >= l.disbursement_date - INTERVAL 30 DAY) AS wallet_bal_mean_30d,
    STDDEV_POP(w.balance_etb)                                                         AS wallet_bal_std_90d
  FROM loans l
  LEFT JOIN wallet_daily w
    ON w.customer_id = l.customer_id
   AND w.date <  l.disbursement_date
   AND w.date >= l.disbursement_date - INTERVAL 90 DAY
  GROUP BY l.loan_id
)
SELECT l.loan_id,
       g.topup_cnt_7d, g.topup_cnt_30d, g.topup_cnt_90d, g.topup_amt_30d, g.voice_min_30d, g.data_mb_30d,
       g.sms_cnt_30d, g.gsm_active_days_30d,
       m.mp_txn_cnt_30d, m.mp_inflow_30d, m.mp_outflow_30d,
       (m.mp_inflow_30d + 1) / (m.mp_outflow_30d + 1) AS inflow_outflow_ratio_30d,
       w.wallet_bal_mean_30d, w.wallet_bal_std_90d, x.days_since_last_mpesa_txn
FROM loans l
JOIN g USING (loan_id) JOIN m USING (loan_id) JOIN last_txn x USING (loan_id) JOIN w USING (loan_id)
ORDER BY l.loan_id;
