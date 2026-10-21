"""Run sql/features_core.sql in DuckDB. Optional engine: `pip install duckdb`.

The pandas/numpy implementation in features.py builds the full feature set; this module runs the
same core features as plain SQL (what a data team would productionise in a warehouse) and is
parity-tested against it in tests/test_duckdb_parity.py.
"""
from __future__ import annotations

import pandas as pd

from .config import ROOT


def core_features_sql(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect()
    for name in ["loans", "gsm_daily", "mpesa_txn", "wallet_daily"]:
        con.register(name, tables[name])
    return con.execute((ROOT / "sql" / "features_core.sql").read_text()).df()
