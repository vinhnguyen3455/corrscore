"""The Diebold-Mariano test (Diebold & Mariano, 1995), vendored rather
than depending on the small, thinly-adopted `dieboldmariano` PyPI
package.

Deliberately mirrors R's `forecast::dm.test` (Hyndman et al.) formula
by formula -- not because that package is depended on, but because it
is the field's de facto reference implementation and this module's own
test suite cross-checks against its output on fixed synthetic data as a
development-time oracle (`tests/_reference/dm_test_oracle_values.py`),
per this package's "oracle, not dependency" policy. In particular: the
Harvey, Leybourne & Newbold (1997) small-sample correction is always
applied (R's `dm.test` has no toggle for it either) and the p-value
comes from a Student-t distribution with `n - 1` degrees of freedom, not
the plain asymptotic normal.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.stats import t as student_t

__all__ = ["DieboldMarianoResult", "diebold_mariano"]


class DieboldMarianoResult(NamedTuple):
    """Result of `diebold_mariano`.

    Attributes
    ----------
    statistic : float
        The Harvey-Leybourne-Newbold-corrected DM statistic.
    p_value : float
        Two-sided p-value, Student-t with `n - 1` degrees of freedom.
    mean_diff : float
        Mean of `loss_a - loss_b` (uncorrected).
    n : int
    """

    statistic: float
    p_value: float
    mean_diff: float
    n: int


def _autocovariance(d: npt.NDArray[np.float64], lag: int) -> float:
    """Sample autocovariance at `lag`, normalized by the FULL sample
    size `n` (not `n - lag`) at every lag -- the convention R's `acf()`
    uses (and that `forecast::dm.test` relies on for its long-run-
    variance estimate), not the `n - lag`-normalized "unbiased" variant.
    Confirmed to matter here: with `n - lag` normalization this
    function's h=1 oracle cases (lag=0 only, where the two conventions
    coincide) passed, while every h>1 case (which needs lag>0
    autocovariances) silently disagreed with R's own output -- exactly
    how the discrepancy was actually caught."""
    n = d.size
    dbar = d.mean()
    if lag == 0:
        return float(np.mean((d - dbar) ** 2))
    return float(np.sum((d[lag:] - dbar) * (d[:-lag] - dbar)) / n)


def diebold_mariano(
    loss_a: npt.ArrayLike,
    loss_b: npt.ArrayLike,
    h: int = 1,
    varestimator: str = "acf",
) -> DieboldMarianoResult:
    """Diebold-Mariano test on the paired loss differential
    `d_t = loss_a[t] - loss_b[t]`.

    Parameters
    ----------
    loss_a, loss_b : array-like
        Per-origin losses (e.g. `BacktestResult.scores[name]`) -- NOT
        raw forecast errors; unlike R's `dm.test(e1, e2, power=...)`,
        this function does not apply a power transform, since the
        objects being compared here are already non-negative scores
        (energy score, variogram score). Passing already-nonnegative
        losses with R's `power=1` reproduces this function's `d`
        exactly (see the reference-oracle test suite).
    h : int, default=1
        Forecast horizon. Controls the long-run-variance truncation lag
        (`h - 1`) and the Harvey-Leybourne-Newbold correction factor.
    varestimator : {"acf", "bartlett"}, default="acf"
        "acf": unweighted sum of sample autocovariances up to lag
        `h - 1` (R's own default, and the only option when `h == 1`,
        where the two coincide). "bartlett": Bartlett-kernel-weighted
        sum (`1 - k/h` weights), matching R's `varestimator="bartlett"`.

    Returns
    -------
    DieboldMarianoResult
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = d.size
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if h > n:
        raise ValueError(f"h ({h}) cannot exceed the number of observations ({n})")

    gamma0 = _autocovariance(d, 0)
    if varestimator == "acf" or h == 1:
        long_run_var = gamma0 + 2.0 * sum(_autocovariance(d, k) for k in range(1, h))
    elif varestimator == "bartlett":
        long_run_var = gamma0 + 2.0 * sum(
            (1.0 - k / h) * _autocovariance(d, k) for k in range(1, h)
        )
    else:
        raise ValueError(f"Unknown varestimator: {varestimator!r}")
    long_run_var /= n

    if long_run_var <= 0:
        raise ValueError(
            "Estimated long-run variance of the loss differential is <= 0 "
            "(a degenerate or perfectly-periodic differential); the DM "
            "statistic is undefined. R's dm.test falls back to h=1 with a "
            "warning in this situation -- retry with h=1 or varestimator="
            "'bartlett' explicitly rather than silently doing so here."
        )

    dbar = float(d.mean())
    raw_statistic = dbar / np.sqrt(long_run_var)
    correction = np.sqrt((n + 1 - 2 * h + (h / n) * (h - 1)) / n)
    statistic = float(raw_statistic * correction)
    p_value = float(2.0 * student_t.cdf(-abs(statistic), df=n - 1))

    return DieboldMarianoResult(statistic=statistic, p_value=p_value, mean_diff=dbar, n=n)
