# Data dictionary

All data is synthetic (see `src/generate_data.py`). Currency is ETB-like and illustrative.

## `customers`
One row per subscriber (static attributes).

| Column | Description |
|---|---|
| `customer_id` | Unique subscriber id |
| `region` | Region of registration (nullable in raw data) |
| `device_type` | SMARTPHONE or FEATURE_PHONE |
| `tenure_months_at_start` | Months since registration at history start |

## `gsm_daily`
One row per customer-day with any GSM activity (missing day = no activity).

| Column | Description |
|---|---|
| `customer_id` | FK to customers |
| `date` | Activity date |
| `topup_count` | Airtime recharges that day |
| `topup_amount` | Airtime recharge value (ETB) |
| `voice_min` | Voice minutes (in+out) |
| `data_mb` | Mobile data (MB) |
| `sms_count` | SMS sent |

## `mpesa_txn`
One row per mobile-money transaction.

| Column | Description |
|---|---|
| `txn_id` | Unique transaction id |
| `customer_id` | FK to customers |
| `txn_date` | Transaction date |
| `txn_type` | CASH_IN, CASH_OUT, P2P_SEND, P2P_RECEIVE, BILL_PAY, MERCHANT, SALARY_IN |
| `direction` | IN (credits wallet) or OUT (debits wallet) |
| `amount` | Amount (ETB) |

## `wallet_daily`
End-of-day wallet balance for every customer-day (overdraft not modelled; floored at 0).

| Column | Description |
|---|---|
| `customer_id` | FK to customers |
| `date` | Balance date |
| `balance_etb` | End-of-day balance (ETB) |

## `loans`
One row per 15-day advance. Target = default_flag.

| Column | Description |
|---|---|
| `loan_id` | Unique loan id |
| `customer_id` | FK to customers |
| `disbursement_date` | Snapshot date for scoring |
| `principal_etb` | Amount advanced (ETB) |
| `fee_etb` | Fee earned if repaid (ETB) |
| `due_date` | Disbursement + 15 days |
| `repaid_date` | Date fully repaid (null = never repaid) |
| `default_flag` | 1 if not repaid 30 days after disbursement (target) |
| `outcome_date` | Date the outcome became known (repaid_date, or disbursement + 30 days) |
