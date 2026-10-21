# Data quality report: clean

**PASS 65 | WARN 0 | FAIL 0**

| Status | Table | Check | Detail |
|---|---|---|---|
| PASS | customers | `schema_columns` | missing=[] unexpected=[] |
| PASS | customers | `schema_dtypes` | ok |
| PASS | customers | `unique[customer_id]` | duplicate_rows=0 |
| PASS | gsm_daily | `schema_columns` | missing=[] unexpected=[] |
| PASS | gsm_daily | `schema_dtypes` | ok |
| PASS | gsm_daily | `unique[customer_id,date]` | duplicate_rows=0 |
| PASS | mpesa_txn | `schema_columns` | missing=[] unexpected=[] |
| PASS | mpesa_txn | `schema_dtypes` | ok |
| PASS | mpesa_txn | `unique[txn_id]` | duplicate_rows=0 |
| PASS | wallet_daily | `schema_columns` | missing=[] unexpected=[] |
| PASS | wallet_daily | `schema_dtypes` | ok |
| PASS | wallet_daily | `unique[customer_id,date]` | duplicate_rows=0 |
| PASS | loans | `schema_columns` | missing=[] unexpected=[] |
| PASS | loans | `schema_dtypes` | ok |
| PASS | loans | `unique[loan_id]` | duplicate_rows=0 |
| PASS | gsm_daily | `fk[customer_id->customers]` | orphan_rows=0 |
| PASS | mpesa_txn | `fk[customer_id->customers]` | orphan_rows=0 |
| PASS | wallet_daily | `fk[customer_id->customers]` | orphan_rows=0 |
| PASS | loans | `fk[customer_id->customers]` | orphan_rows=0 |
