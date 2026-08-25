from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from corrscore import asymmetric_weighted_mean


def test_kappa_one_is_plain_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    severity = np.array([0.1, 0.9, 0.2, 0.8])
    assert asymmetric_weighted_mean(values, severity, kappa=1.0) == pytest.approx(values.mean())


def test_kappa_zero_removes_high_severity_origins_from_average():
    """kappa < 1 down-weights high-severity origins -- at kappa=0 they
    should be excluded from the average entirely."""
    values = np.array([10.0, 10.0, 1000.0, 1000.0])
    severity = np.array([0.1, 0.2, 0.9, 0.8])  # last two are "high severity"
    result = asymmetric_weighted_mean(values, severity, kappa=0.0)
    assert result == pytest.approx(10.0)


def test_high_kappa_dominated_by_high_severity_origins():
    values = np.array([1.0, 1.0, 100.0, 100.0])
    severity = np.array([0.1, 0.2, 0.9, 0.8])
    result = asymmetric_weighted_mean(values, severity, kappa=1000.0)
    assert result == pytest.approx(100.0, rel=0.05)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        asymmetric_weighted_mean([1.0, 2.0], [0.1, 0.2, 0.3])


@given(
    n=st.integers(2, 30),
    kappa=st.floats(0.1, 10.0),
    seed=st.integers(0, 10_000),
)
@settings(max_examples=50, deadline=None)
def test_matches_hand_written_weighted_average(n, kappa, seed):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n)
    severity = rng.uniform(0, 1, n)
    median = np.median(severity)
    weights = np.where(severity >= median, kappa, 1.0)
    expected = float(np.sum(values * weights) / np.sum(weights))
    assert asymmetric_weighted_mean(values, severity, kappa=kappa) == pytest.approx(expected)
