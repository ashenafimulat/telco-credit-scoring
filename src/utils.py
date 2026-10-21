"""Small IO helpers shared by all stages."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

DATE_COLS = {
    "customers": [],
    "gsm_daily": ["date"],
    "mpesa_txn": ["txn_date"],
    "wallet_daily": ["date"],
    "loans": ["disbursement_date", "due_date", "repaid_date", "outcome_date"],
}
TABLES = list(DATE_COLS)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip" if str(path).endswith(".gz") else None)


def read_table(path: Path, name: str | None = None) -> pd.DataFrame:
    parse = DATE_COLS.get(name, []) if name else []
    return pd.read_csv(path, parse_dates=parse)


def read_layer(directory: Path) -> dict[str, pd.DataFrame]:
    return {t: read_table(directory / f"{t}.csv.gz", t) for t in TABLES}


def write_layer(tables: dict[str, pd.DataFrame], directory: Path) -> None:
    for name, df in tables.items():
        write_table(df, directory / f"{name}.csv.gz")
