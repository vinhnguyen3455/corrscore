"""The zero-overlap walk-forward backtest driver.

Design note: an earlier sketch of this API had a `window` parameter
defaulting to `horizon`. Working through the actual mechanics precisely
during implementation surfaced a cleaner, more honestly-scoped contract,
documented here rather than silently substituted. This harness does not,
and cannot, police how much history a caller's `forecast_fn` consults
internally -- that model is opaque to the harness (it might be a
full-history discounted filter, a short trailing window, or anything
else), exactly the same responsibility boundary scikit-learn's
`TimeSeriesSplit` leaves to the caller. What the harness genuinely *can*
and does enforce, unconditionally, is that `ground_truth_fn` is always
called with a start point strictly after the origin
(`origin + 1 + purge_gap`), never a window that reaches back before or
across it. That guards against a real class of bug this package exists
to prevent: computing "ground truth" as a window *ending* near the
origin rather than a window *starting* strictly after it, which silently
leaks estimation-window information into the evaluation. This API makes
that specific mistake structurally impossible to reproduce, since the
caller never controls the start point passed to `ground_truth_fn`.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np
import numpy.typing as npt

from .scoring import matrix_energy_score

ForecastFn = Callable[[int], Mapping[str, Any]]
GroundTruthFn = Callable[[int, int], npt.ArrayLike]
ScoreFn = Callable[[Mapping[str, Any], npt.ArrayLike], float]
SeverityFn = Callable[[npt.NDArray[np.float64]], float]

__all__ = ["BacktestResult", "backtest_zero_overlap"]


class BacktestResult(NamedTuple):
    """Result of `backtest_zero_overlap`.

    Attributes
    ----------
    origins : list of int
        The origins actually scored, in order.
    scores : dict of str -> ndarray
        Per-model, per-origin scores, one array per key of the
        `forecast_fns` mapping passed in, aligned with `origins`.
    severity : ndarray or None
        Per-origin realized-severity values (`severity_fn(y)` at each
        origin), or None if `severity_fn` was not supplied. Intended for
        `corrscore.asymmetric_weighted_mean`.
    horizon : int
    purge_gap : int
    """

    origins: list[int]
    scores: dict[str, npt.NDArray[np.float64]]
    severity: npt.NDArray[np.float64] | None
    horizon: int
    purge_gap: int


def backtest_zero_overlap(
    forecast_fns: Mapping[str, ForecastFn] | ForecastFn,
    ground_truth_fn: GroundTruthFn,
    origins: Sequence[int],
    horizon: int,
    purge_gap: int = 0,
    score_fn: ScoreFn = matrix_energy_score,
    severity_fn: SeverityFn | None = None,
) -> BacktestResult:
    """Score one or more forecasting methods against a proper,
    zero-overlap-by-construction ground truth.

    For each `origin` in `origins`: computes
    `y = ground_truth_fn(origin + 1 + purge_gap, origin + 1 + purge_gap
    + horizon)`, then scores each `forecast_fns[name](origin)` against
    `y` via `score_fn`. `purge_gap=0` (the default) gives zero shared
    days between the forecast origin and the ground-truth window by
    construction; a larger `purge_gap` is a deliberate relaxation the
    caller must opt into explicitly -- never a silent default.

    Parameters
    ----------
    forecast_fns : callable, or dict of str -> callable
        Each callable maps an origin (int) to a forecast dict in the
        shape `matrix_energy_score`/`matrix_variogram_score` expect
        (see `corrscore.scoring`). A single callable is treated as
        `{"model": forecast_fns}`. Sharing one `ground_truth_fn` call
        per origin across every model is deliberate: ground truth is
        model-independent and often the more expensive computation, and
        this shape is exactly what `corrscore.circular_block_bootstrap`,
        `corrscore.diebold_mariano`, and `corrscore.model_confidence_set`
        expect as input (`result.scores[name]`).
    ground_truth_fn : callable
        `(start, end) -> K x K matrix`. Called only with
        `start = origin + 1 + purge_gap`, `end = start + horizon` --
        never anything else. It is the caller's responsibility that
        this function computes a genuinely forward-looking realized
        estimate from `[start, end)`, not a trailing one -- the timing
        guard above prevents overlap, but not a `ground_truth_fn` that
        is itself defined as a trailing window.
    origins : sequence of int
    horizon : int
    purge_gap : int, default=0
    score_fn : callable, default=matrix_energy_score
    severity_fn : callable, optional
        `y -> float`, a per-origin realized-severity summary (e.g. mean
        absolute off-diagonal correlation) for later use with
        `corrscore.asymmetric_weighted_mean`.

    Returns
    -------
    BacktestResult
    """
    if callable(forecast_fns):
        forecast_fns = {"model": forecast_fns}
    if purge_gap < 0:
        raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    names = list(forecast_fns.keys())
    scores: dict[str, list[float]] = {name: [] for name in names}
    severities: list[float] = []
    used_origins: list[int] = []

    for origin in origins:
        start = origin + 1 + purge_gap
        end = start + horizon
        y = np.asarray(ground_truth_fn(start, end), dtype=float)
        for name in names:
            forecast = forecast_fns[name](origin)
            scores[name].append(score_fn(forecast, y))
        if severity_fn is not None:
            severities.append(severity_fn(y))
        used_origins.append(origin)

    scores_arr = {name: np.array(values) for name, values in scores.items()}
    severity_arr = np.array(severities) if severity_fn is not None else None
    return BacktestResult(
        origins=used_origins,
        scores=scores_arr,
        severity=severity_arr,
        horizon=horizon,
        purge_gap=purge_gap,
    )
