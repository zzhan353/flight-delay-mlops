"""Drift detection behaviour — especially what must *not* raise an alarm."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flight_delay.monitor import PSI_ALERT, compare, psi


def _normal(n=5000, loc=0.0, scale=1.0, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(loc, scale, n))


def test_identical_distributions_have_near_zero_psi():
    assert psi(_normal(seed=1), _normal(seed=2)) < 0.05


def test_shifted_distribution_is_detected():
    assert psi(_normal(loc=0), _normal(loc=2, seed=3)) >= PSI_ALERT


def test_psi_is_finite_when_a_bin_empties():
    """An empty bin must not produce infinity and drown out every other feature."""
    value = psi(_normal(), pd.Series(np.full(5000, 99.0)))
    assert np.isfinite(value)


def test_psi_returns_nan_for_samples_too_small_to_judge():
    assert np.isnan(psi(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])))


def test_compare_reports_label_rate_shift():
    ref = pd.DataFrame({"Distance": _normal(), "ArrDel15": [0] * 2500 + [1] * 2500})
    cur = pd.DataFrame({"Distance": _normal(seed=9), "ArrDel15": [0] * 4000 + [1] * 1000})
    scores = compare(ref, cur)
    assert scores["__label_rate_shift"] < 0  # delays became rarer
