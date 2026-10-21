# 📊 Alternative Credit Scoring using Synthetic Telecom Data

This project simulates a **telecom-based credit advance system** to build and evaluate an **alternative credit scoring model** for prepaid subscribers.

The goal is to predict the probability that a user will **default on a $20 credit advance**, where repayment is defined as making at least one recharge within a 15-day window.

---

## 🎯 Business Problem

In many emerging markets, a large portion of users are **thin-file or unbanked**, making traditional credit bureau scoring ineffective.

Telecom operators can leverage behavioral data to estimate creditworthiness and offer micro-credit products such as:

- $20 airtime/data credit advance
- Automatically granted when balance is low
- Repayment expected within 15 days via recharge activity

---

## 🧠 Target Definition

A user is considered:

- **NOT DEFAULT (0)** → at least one recharge within 15 days  
- **DEFAULT (1)** → no recharge within 15 days after credit advance

This forms a **binary classification problem**.

---

## 📦 Dataset Features

The dataset simulates real-world telecom behavioral signals across four categories:

### 📞 Voice Features
- `outgoing_call_count`
- `incoming_call_count`
- `outgoing_minutes`
- `incoming_minutes`
- `avg_call_duration`
- `outgoing_incoming_ratio`

### 💬 SMS Features
- `sms_sent`
- `sms_received`

### 📡 Data Usage Features
- `download_mb`
- `upload_mb`
- `avg_daily_data_mb`

### 💰 Recharge Features
- `recharge_count_30d`
- `recharge_amount_30d`
- `avg_recharge_amount`

### 📱 Behavioral Features
- `days_since_last_recharge`
- `active_days_30d`
- `num_cell_site_latch_ons`

---

## ⚙️ Data Generation Logic

The dataset is synthetically generated using a **latent creditworthiness model**:

1. Users are assigned a hidden creditworthiness score
2. Telecom usage behavior is simulated based on this score
3. Recharge behavior is modeled as a probabilistic process
4. Default is derived from **repayment behavior within 15 days**

This ensures realistic relationships between:
- Usage intensity
- Financial behavior
- Repayment likelihood

---

Here is the corrected version in clean **Markdown format**:

## 🏗️ Project Structure

```
project/
├── data_generator.py      # Synthetic dataset generation
├── train_model.py         # ML model training
├── scoring.py             # Credit score conversion (200–800)
├── utils.py               # Helper functions
├── notebook.ipynb         # EDA & experimentation
└── README.md
```

---

## 🤖 Modeling Approach

The problem is framed as a **binary classification task**:

### Example models:
- Logistic Regression (baseline / scorecard-style)
- Random Forest
- XGBoost / LightGBM (recommended)

### Output:
- `P(default = 1)` → probability of non-repayment

---

## 💳 Credit Score Mapping

Model probabilities are converted into a **credit score between 200 and 800**:

```python
def probability_to_credit_score(p_default):
    p_good = 1 - p_default
    score = 200 + (p_good * 600)
    return np.clip(score, 200, 800)
