"""The leakage guarantee, enforced in CI."""

from __future__ import annotations

import pandas as pd
import pytest

from flight_delay.data.build import assert_no_leakage
from flight_delay.schema import (
    BTS_SOURCE_COLUMNS,
    FEATURES,
    LEAKING_COLUMNS,
    TARGET,
)


def test_features_and_leaking_columns_are_disjoint():
    overlap = set(FEATURES) & set(LEAKING_COLUMNS)
    assert not overlap, f"these columns are declared both usable and leaking: {overlap}"


def test_target_is_not_a_feature():
    assert TARGET not in FEATURES


def test_source_columns_carry_no_leakage():
    """We must not even read post-departure columns off disk."""
    leaked = set(BTS_SOURCE_COLUMNS) & set(LEAKING_COLUMNS)
    assert not leaked, f"BTS reader would load leaking columns: {leaked}"


def _frame(**extra) -> pd.DataFrame:
    base = {name: [0] for name in FEATURES}
    base[TARGET] = [0]
    base.update({k: [v] for k, v in extra.items()})
    return pd.DataFrame(base)


def test_assert_no_leakage_accepts_a_clean_frame():
    assert_no_leakage(_frame())


def test_assert_no_leakage_rejects_a_leaked_column():
    with pytest.raises(ValueError, match="post-departure"):
        assert_no_leakage(_frame(DepDelay=12.0))


def test_assert_no_leakage_rejects_a_missing_feature():
    df = _frame().drop(columns=["Distance"])
    with pytest.raises(ValueError, match="missing"):
        assert_no_leakage(df)
