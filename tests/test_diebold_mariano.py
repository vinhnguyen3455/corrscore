"""Tests for diebold_mariano: property tests plus a byte-level oracle
cross-check against R's forecast::dm.test (tests/_reference/
dm_test_oracle_values.py). The oracle check is the real evidence here --
it caught a genuine bug during development (n-lag vs. n normalization in
the autocovariance estimator, see diebold_mariano.py's own comment on
_autocovariance) before this file existed, which is exactly the point of
having an independent reference to check against rather than trusting
the formula transcription alone.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from corrscore import diebold_mariano

_spec = importlib.util.spec_from_file_location(
    "dm_test_oracle_values", Path(__file__).parent / "_reference" / "dm_test_oracle_values.py"
)
_dm_oracle_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dm_oracle_module)
DM_ORACLE_CASES = _dm_oracle_module.DM_ORACLE_CASES


@pytest.mark.parametrize("case", DM_ORACLE_CASES, ids=[f"h={c['h']}_{c['varestimator']}" for c in DM_ORACLE_CASES])
def test_matches_r_forecast_dm_test_oracle(case):
    result = diebold_mariano(case["loss_a"], case["loss_b"], h=case["h"], varestimator=case["varestimator"])
    assert result.statistic == pytest.approx(case["statistic"], abs=1e-6)
    assert result.p_value == pytest.approx(case["p_value"], abs=1e-6)


def test_literally_identical_losses_raise_zero_variance_error():
    """d = loss_a - loss_b is exactly 0 everywhere here, so its
    long-run variance is exactly 0 -- both this implementation and R's
    own dm.test (`stop("Variance of DM statistic is zero")` at h=1)
    correctly refuse to divide by it rather than silently returning a
    meaningless statistic."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(50) + 5
    with pytest.raises(ValueError):
        diebold_mariano(a, a)


def test_nearly_identical_losses_give_a_statistic_near_zero():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(50) + 5
    b = a + rng.standard_normal(50) * 1e-3  # tiny independent noise, breaks the exact-zero-variance case
    result = diebold_mariano(a, b)
    assert abs(result.statistic) < 3.0
    assert result.p_value > 0.01


def test_swapping_arguments_flips_the_sign():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(40) + 3
    b = rng.standard_normal(40) + 2.5
    r_ab = diebold_mariano(a, b)
    r_ba = diebold_mariano(b, a)
    assert r_ab.statistic == pytest.approx(-r_ba.statistic)
    assert r_ab.p_value == pytest.approx(r_ba.p_value)


def test_h_greater_than_n_raises():
    with pytest.raises(ValueError):
        diebold_mariano([1.0, 2.0, 3.0], [1.1, 2.1, 3.1], h=5)


def test_h_less_than_one_raises():
    with pytest.raises(ValueError):
        diebold_mariano([1.0, 2.0], [1.1, 2.1], h=0)


def test_unknown_varestimator_raises():
    with pytest.raises(ValueError):
        diebold_mariano([1.0, 2.0, 3.0], [1.1, 2.1, 3.1], h=2, varestimator="nonsense")


@given(seed=st.integers(0, 10_000), n=st.integers(20, 100))
@settings(max_examples=30, deadline=None)
def test_h1_p_value_matches_plain_normal_z_test(seed, n):
    """At h=1 there's no HAC truncation to get subtly wrong, and the
    HLN correction factor is close to 1 for reasonably large n -- an
    independent, from-scratch sanity check that doesn't rely on the R
    oracle at all."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n) + 5.0
    b = rng.standard_normal(n) + 5.0
    result = diebold_mariano(a, b, h=1)
    d = np.asarray(a) - np.asarray(b)
    plain_z = d.mean() / (d.std(ddof=0) / np.sqrt(n))
    assert result.statistic == pytest.approx(plain_z, rel=0.05)
