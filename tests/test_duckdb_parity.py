import numpy as np
import pytest

from src.features import CORE_PARITY, build_features
from tests.helpers import small_tables


def test_sql_matches_pandas_features():
    pytest.importorskip("duckdb")
    from src.features_duckdb import core_features_sql
    cfg, _, clean = small_tables()
    a = build_features(clean, cfg).set_index("loan_id").sort_index()[CORE_PARITY]
    b = core_features_sql(clean).set_index("loan_id").sort_index()[CORE_PARITY].astype(float)
    assert list(a.index) == list(b.index)
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-6, atol=1e-6, equal_nan=True)
