"""Time-stamped synthetic telco credit data generator.

Extends the original snapshot generator (see legacy_snapshot/, Apache-2.0) from one flat table
into five linked, dated tables that look like what a telco lender would hold:

    customers      static attributes
    gsm_daily      daily airtime top-ups, voice, data, SMS (rows only for active days)
    mpesa_txn      mobile-money transactions (cash-in/out, P2P, bill pay, merchant, salary)
    wallet_daily   end-of-day mobile-wallet balance (floored at zero, no overdraft modelled)
    loans          15-day airtime/cash advances with repayment outcome

Hidden drivers (never written to disk): financial stability `z`, income, income volatility,
top-up irregularity, and a persistent monthly "stress" state that depresses top-ups and inflows
and raises default risk. Default is also affected by prior late repayment and loan size relative
to income, plus unobserved noise, so models find signal but not a trivial one.

An optional shock (airtime promo + delayed inflows + a macro default shift) starts at
`generator.shock_start` so monitoring has real drift to detect.

Currency is ETB-like; numbers are illustrative, not calibrated to any operator.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import layer_dir, load_config
from .utils import get_logger, write_layer

log = get_logger("generate_data")

REGIONS = ["ADDIS_ABABA", "OROMIA", "AMHARA", "TIGRAY", "SNNPR", "SIDAMA", "OTHER"]
REGION_P = [0.30, 0.25, 0.15, 0.05, 0.10, 0.05, 0.10]
# type: (direction, base rate per day, mean amount as a fraction of monthly income)
MPESA_TYPES = {
    "CASH_IN": ("IN", 0.10, 0.14),
    "CASH_OUT": ("OUT", 0.12, 0.08),
    "P2P_SEND": ("OUT", 0.08, 0.04),
    "P2P_RECEIVE": ("IN", 0.08, 0.05),
    "BILL_PAY": ("OUT", 0.03, 0.05),
    "MERCHANT": ("OUT", 0.06, 0.02),
}
PRINCIPAL_TIERS = np.array([50, 100, 200, 300, 500])
# default-logit coefficients: kept modest + large unobserved noise so achievable AUC is realistic (~0.70-0.75)
SIG = {"z": 0.55, "stress": 0.8, "vol": 0.25, "irreg": 0.20, "loan": 0.35, "prior": 0.5, "noise": 1.0}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(cfg: dict, n_customers: int | None = None, seed: int | None = None,
             shock: bool = True, dirty_rate: float | None = None) -> dict[str, pd.DataFrame]:
    g = cfg["generator"]
    n = int(n_customers or g["n_customers"])
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)

    dates = pd.date_range(g["history_start"], g["history_end"], freq="D")
    D = len(dates)
    start = dates[0]
    month_idx = ((dates.year - start.year) * 12 + dates.month - start.month).to_numpy()
    M = int(month_idx.max()) + 1
    dom = dates.day.to_numpy()
    dow = dates.dayofweek.to_numpy()

    shock_ts = pd.Timestamp(g["shock_start"]) if (shock and g.get("shock_start")) else None
    sh = g["shock"]
    shock_day = np.zeros(D, bool) if shock_ts is None else np.asarray(dates >= shock_ts)
    f_freq = np.where(shock_day, sh["topup_freq_mult"], 1.0)
    f_amt = np.where(shock_day, sh["topup_amt_mult"], 1.0)
    f_in = np.where(shock_day, sh["inflow_mult"], 1.0)

    # ---------------- hidden customer traits ----------------
    z = rng.standard_normal(n)
    income = np.exp(8.5 + 0.30 * z + 0.5 * rng.standard_normal(n))
    vol = np.clip(0.25 - 0.06 * z + 0.10 * rng.standard_normal(n), 0.05, 0.9)
    irreg = np.clip(0.8 - 0.25 * z + 0.2 * rng.standard_normal(n), 0.15, 2.5)
    salaried = rng.random(n) < _sigmoid(0.2 + 0.6 * z)
    smartphone = rng.random(n) < _sigmoid(0.3 + 0.5 * z + 0.4 * np.log(income / 5000))
    region = rng.choice(REGIONS, n, p=REGION_P)
    tenure0 = rng.integers(1, 60, n)
    cust_ids = np.array([f"C{i + 1:06d}" for i in range(n)])

    # persistent monthly stress state
    p_base = _sigmoid(-2.2 - 0.6 * z)
    stress = np.zeros((n, M), bool)
    stress[:, 0] = rng.random(n) < p_base
    for m in range(1, M):
        stress[:, m] = rng.random(n) < np.clip(p_base + 0.35 * stress[:, m - 1], 0, 0.9)
    stress_d = stress[:, month_idx].astype(float)  # [n, D]

    # ---------------- GSM daily ----------------
    topup_rate = np.clip(0.12 + 0.04 * z + 0.03 * rng.standard_normal(n), 0.03, 0.6)
    amt_mean = np.clip(25 * np.exp(0.25 * z + 0.3 * rng.standard_normal(n)), 5, 300)
    k = 1.0 / irreg
    wk = 1 + 0.15 * (dow >= 5)
    lam = topup_rate[:, None] * wk[None, :] * (1 - 0.45 * stress_d) * f_freq[None, :]
    mult = rng.gamma(shape=k[:, None], scale=1.0 / k[:, None], size=(n, D))
    topup_cnt = rng.poisson(lam * mult)
    topup_amt = topup_cnt * amt_mean[:, None] * f_amt[None, :] * np.exp(0.25 * rng.standard_normal((n, D)))

    voice_scale = np.exp(1.8 + 0.2 * z + 0.3 * rng.standard_normal(n))
    p_voice = np.clip(0.85 - 0.25 * stress_d, 0.05, 1.0)
    voice = np.where(rng.random((n, D)) < p_voice, rng.gamma(1.5, voice_scale[:, None], size=(n, D)), 0.0)
    data_mu = 3.5 + 0.3 * z + 0.5 * rng.standard_normal(n)
    p_data = np.clip(0.75 - 0.25 * stress_d, 0, 1) * smartphone[:, None]
    data = np.where(rng.random((n, D)) < p_data, rng.lognormal(data_mu[:, None], 0.8, size=(n, D)), 0.0)
    sms_rate = np.exp(0.8 + 0.3 * z + 0.3 * rng.standard_normal(n))
    sms = rng.poisson(sms_rate[:, None] * (1 - 0.3 * stress_d))

    active = (topup_cnt > 0) | (voice > 0) | (data > 0) | (sms > 0)
    ci, di = np.nonzero(active)
    gsm = pd.DataFrame({
        "customer_id": cust_ids[ci],
        "date": dates.to_numpy()[di],
        "topup_count": topup_cnt[ci, di],
        "topup_amount": topup_amt[ci, di].round(2),
        "voice_min": voice[ci, di].round(1),
        "data_mb": data[ci, di].round(1),
        "sms_count": sms[ci, di],
    })
    del lam, mult, topup_cnt, topup_amt, voice, data, sms, active

    # ---------------- M-PESA transactions + wallet ----------------
    act = np.exp(0.3 * z + 0.3 * rng.standard_normal(n))
    # salaried customers spend most of their salary, so balances stay roughly stationary over time
    a_in, a_out = np.exp(0.12 * z), np.exp(-0.12 * z) * (1.1 + 1.9 * salaried)
    parts = []
    for name, (direction, rate, frac) in MPESA_TYPES.items():
        base = rate * act * (a_in if direction == "IN" else a_out)
        lam = base[:, None] * np.ones((1, D))
        lam = lam * ((1 - 0.5 * stress_d) * f_in[None, :] if direction == "IN" else (1 - 0.2 * stress_d))
        cnt = rng.poisson(lam).ravel()
        nz = np.flatnonzero(cnt)
        cells = np.repeat(nz, cnt[nz])
        c, d = cells // D, cells % D
        sig = 0.4 + vol[c]
        amt = income[c] * frac * np.exp(sig * rng.standard_normal(len(c)) - sig**2 / 2)
        parts.append((c, d, np.full(len(c), name), np.full(len(c), direction), amt))
    # salary credits on the 26th for salaried customers (some delayed during the shock)
    pay_days = np.flatnonzero(dom == 26)
    sc, sd = np.meshgrid(np.flatnonzero(salaried), pay_days, indexing="ij")
    sc, sd = sc.ravel(), sd.ravel()
    keep = rng.random(len(sc)) < 0.92 * f_in[sd]
    sc, sd = sc[keep], sd[keep]
    parts.append((sc, sd, np.full(len(sc), "SALARY_IN"), np.full(len(sc), "IN"),
                  income[sc] * (1 + 0.1 * rng.standard_normal(len(sc)))))
    c_all = np.concatenate([p[0] for p in parts])
    d_all = np.concatenate([p[1] for p in parts])
    t_all = np.concatenate([p[2] for p in parts])
    dir_all = np.concatenate([p[3] for p in parts])
    a_all = np.concatenate([p[4] for p in parts]).round(2)
    order = np.lexsort((t_all, d_all, c_all))
    c_all, d_all, t_all, dir_all, a_all = (x[order] for x in (c_all, d_all, t_all, dir_all, a_all))
    mpesa = pd.DataFrame({
        "txn_id": [f"T{i + 1:09d}" for i in range(len(c_all))],
        "customer_id": cust_ids[c_all],
        "txn_date": dates.to_numpy()[d_all],
        "txn_type": t_all,
        "direction": dir_all,
        "amount": a_all,
    })
    net = np.zeros((n, D))
    np.add.at(net, (c_all, d_all), np.where(dir_all == "IN", a_all, -a_all))
    cur = income * 0.2 * np.exp(0.5 * rng.standard_normal(n))
    bal = np.empty((n, D))
    for d in range(D):
        cur = np.maximum(0.0, cur + net[:, d])
        bal[:, d] = cur
    wallet = pd.DataFrame({
        "customer_id": np.repeat(cust_ids, D),
        "date": np.tile(dates.to_numpy(), n),
        "balance_etb": bal.ravel().round(2),
    })

    # ---------------- loans ----------------
    vint = pd.period_range(g["first_vintage"], g["last_vintage"], freq="M")
    nv = len(vint)
    p_take = np.clip(g["loan_take_prob"] * np.exp(0.4 * rng.standard_normal(n) - 0.08), 0.05, 0.95)
    U_take, U_bad = rng.random((n, nv)), rng.random((n, nv))
    U_day = rng.integers(1, 29, (n, nv))
    E_lag, eps = rng.exponential(1.0, (n, nv)), rng.standard_normal((n, nv))
    tier_noise = rng.standard_normal((n, nv))
    inc_rank = pd.Series(income).rank(pct=True).to_numpy()
    tier = np.clip(np.floor(inc_rank[:, None] * 5 + 0.8 * tier_noise), 0, 4).astype(int)
    principal = PRINCIPAL_TIERS[tier]
    vol_s = (vol - vol.mean()) / vol.std()
    irr_s = (irreg - irreg.mean()) / irreg.std()
    v_start = np.array([(p.start_time - start).days for p in vint])
    v_mi = np.array([(p.year - start.year) * 12 + p.month - start.month for p in vint])
    day_idx = v_start[None, :] + U_day - 1
    v_shift = np.array([sh["default_logit_shift"] if (shock_ts is not None and p.start_time >= shock_ts) else 0.0
                        for p in vint])
    term, after = g["loan_term_days"], g["default_after_days"]

    def simulate(b0: float):
        last = np.full(n, -10_000)
        prior_late = np.zeros(n)
        out = []
        for j in range(nv):
            take = (U_take[:, j] < p_take) & (day_idx[:, j] - last >= after + 1)
            c = np.flatnonzero(take)
            st = stress[c, v_mi[j]].astype(float)
            logit = (b0 - SIG['z'] * z[c] + SIG['stress'] * st + SIG['vol'] * vol_s[c] + SIG['irreg'] * irr_s[c]
                     + SIG['loan'] * (np.log(principal[c, j] / income[c]) + 3.5) / 0.8
                     + SIG['prior'] * prior_late[c] + v_shift[j] + SIG['noise'] * eps[c, j])
            bad = U_bad[c, j] < _sigmoid(logit)
            lag = np.clip(np.ceil(E_lag[c, j] * (4 + 4 * st + 2 * np.clip(-z[c], 0, None))), 1, after)
            prior_late[c] = (bad | (lag > term)).astype(float)
            last[c] = day_idx[c, j]
            out.append((c, day_idx[c, j], principal[c, j], bad, lag))
        return out

    lo, hi = -8.0, 3.0
    for _ in range(30):  # bisection on the intercept to hit the target default rate
        mid = (lo + hi) / 2
        res = simulate(mid)
        rate = sum(r[3].sum() for r in res) / sum(len(r[3]) for r in res)
        lo, hi = (mid, hi) if rate < g["target_default_rate"] else (lo, mid)
    res = simulate((lo + hi) / 2)
    c = np.concatenate([r[0] for r in res]); dd = np.concatenate([r[1] for r in res])
    pr = np.concatenate([r[2] for r in res]); bad = np.concatenate([r[3] for r in res])
    lag = np.concatenate([r[4] for r in res])
    o = np.lexsort((c, dd))
    c, dd, pr, bad, lag = c[o], dd[o], pr[o], bad[o], lag[o]
    disb = start + pd.to_timedelta(dd, unit="D")
    repaid = pd.Series(disb + pd.to_timedelta(lag, unit="D")).where(~bad)
    loans = pd.DataFrame({
        "loan_id": [f"L{i + 1:07d}" for i in range(len(c))],
        "customer_id": cust_ids[c],
        "disbursement_date": disb,
        "principal_etb": pr,
        "fee_etb": (pr * g["fee_rate"]).round(2),
        "due_date": disb + pd.Timedelta(days=term),
        "repaid_date": repaid.to_numpy(),
        "default_flag": bad.astype(int),
    })
    loans["outcome_date"] = loans["repaid_date"].fillna(loans["disbursement_date"] + pd.Timedelta(days=after))

    customers = pd.DataFrame({
        "customer_id": cust_ids,
        "region": region,
        "device_type": np.where(smartphone, "SMARTPHONE", "FEATURE_PHONE"),
        "tenure_months_at_start": tenure0,
    })
    tables = {"customers": customers, "gsm_daily": gsm, "mpesa_txn": mpesa,
              "wallet_daily": wallet, "loans": loans}
    rate = dirty_rate if dirty_rate is not None else g.get("dirty_rate", 0.0)
    if rate and rate > 0:
        tables = inject_dirt(tables, rate, rng)
    log.info("generated: %s", {k: len(v) for k, v in tables.items()})
    log.info("default rate %.3f", loans["default_flag"].mean())
    return tables


def inject_dirt(t: dict[str, pd.DataFrame], rate: float, rng) -> dict[str, pd.DataFrame]:
    """Realistic raw-data defects so data-quality checks and cleaning have something to do."""
    t = {k: v.copy() for k, v in t.items()}
    mp = t["mpesa_txn"]
    m = len(mp)
    dup = mp.sample(int(m * rate), random_state=int(rng.integers(1e9)))
    neg = rng.choice(m, int(m * rate / 4), replace=False)
    out = rng.choice(m, int(m * rate / 4), replace=False)
    mp.loc[mp.index[neg], "amount"] = -mp.loc[mp.index[neg], "amount"].abs()
    mp.loc[mp.index[out], "amount"] = mp.loc[mp.index[out], "amount"] * 100
    t["mpesa_txn"] = pd.concat([mp, dup], ignore_index=True)
    gs = t["gsm_daily"]
    t["gsm_daily"] = pd.concat([gs, gs.sample(int(len(gs) * rate / 2), random_state=int(rng.integers(1e9)))],
                               ignore_index=True)
    cu = t["customers"]
    cu.loc[cu.sample(max(1, int(len(cu) * 0.005)), random_state=int(rng.integers(1e9))).index, "region"] = np.nan
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-customers", type=int, default=None)
    ap.add_argument("--no-shock", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tables = generate(cfg, n_customers=args.n_customers, shock=not args.no_shock)
    write_layer(tables, layer_dir(cfg, "raw"))


if __name__ == "__main__":
    main()
