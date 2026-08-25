"""Outcome-severity-weighted aggregation, generalized from
`asymmetric_weighted_mean` in `scripts/regime-detection-pfa-filter.py`
(Stage 6) -- a utility surfaced by refactoring that script against this
package (corrscore-package-design.html Sec. 6.5), not part of the
original companion document's Sec. 6 harness scope.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["asymmetric_weighted_mean"]


def asymmetric_weighted_mean(values: npt.ArrayLike, severity: npt.ArrayLike, kappa: float = 1.0) -> float:
    """Weight per-origin `values` (typically per-origin scores from a
    `BacktestResult`) by realized outcome severity: origins at or above
    the median `severity` are weighted `kappa` times as heavily as
    origins below it.

    `severity` should be a non-circular, outcome-derived measure -- e.g.
    the mean absolute off-diagonal correlation of the realized
    ground-truth window (`BacktestResult.severity`, via
    `backtest_zero_overlap`'s `severity_fn`) -- not any model's own
    regime call, to avoid biasing which origins get up-weighted toward
    whichever model is being evaluated.

    `kappa=1.0` (the default) reduces to a plain mean.
    """
    values_arr = np.asarray(values, dtype=float)
    severity_arr = np.asarray(severity, dtype=float)
    if values_arr.shape != severity_arr.shape:
        raise ValueError(
            f"values and severity must have the same shape, got {values_arr.shape} and {severity_arr.shape}"
        )
    median = np.median(severity_arr)
    weights = np.where(severity_arr >= median, kappa, 1.0)
    return float(np.sum(values_arr * weights) / np.sum(weights))
