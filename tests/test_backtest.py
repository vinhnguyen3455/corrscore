"""Tests for backtest_zero_overlap. The central property to verify is
literally the package's own reason to exist: ground_truth_fn must never
be called with a window that could overlap what a forecast at that
origin used -- see backtest.py's own module docstring for the class of
bug this makes structurally impossible to reproduce."""
from __future__ import annotations

import numpy as np
import pytest

from corrscore import backtest_zero_overlap, matrix_energy_score


def test_ground_truth_fn_always_called_strictly_after_origin():
    calls = []

    def ground_truth(start, end):
        calls.append((start, end))
        return np.eye(3)

    def forecast(origin):
        return {"kind": "point", "Q": np.eye(3)}

    origins = [10, 20, 30]
    backtest_zero_overlap(forecast, ground_truth, origins, horizon=5)

    assert calls == [(11, 16), (21, 26), (31, 36)]
    for origin, (start, end) in zip(origins, calls):
        assert start > origin  # the actual invariant this package sells
        assert end - start == 5


def test_purge_gap_shifts_ground_truth_start_further_out():
    calls = []

    def ground_truth(start, end):
        calls.append((start, end))
        return np.eye(2)

    backtest_zero_overlap(lambda t: {"kind": "point", "Q": np.eye(2)}, ground_truth, [5], horizon=3, purge_gap=2)
    assert calls == [(8, 11)]


def test_forecast_fn_only_ever_receives_the_origin():
    seen_origins = []

    def forecast(origin):
        seen_origins.append(origin)
        return {"kind": "point", "Q": np.eye(2)}

    backtest_zero_overlap(forecast, lambda s, e: np.eye(2), [1, 2, 3], horizon=1)
    assert seen_origins == [1, 2, 3]


def test_single_callable_forecast_fns_is_wrapped_as_model():
    result = backtest_zero_overlap(
        lambda t: {"kind": "point", "Q": np.eye(2)}, lambda s, e: np.eye(2), [1, 2], horizon=1
    )
    assert list(result.scores.keys()) == ["model"]
    assert np.all(result.scores["model"] == 0.0)


def test_multiple_models_share_the_same_ground_truth_call():
    ground_truth_calls = []

    def ground_truth(start, end):
        ground_truth_calls.append((start, end))
        return np.eye(2)

    result = backtest_zero_overlap(
        forecast_fns={
            "a": lambda t: {"kind": "point", "Q": np.eye(2)},
            "b": lambda t: {"kind": "point", "Q": 2 * np.eye(2)},
        },
        ground_truth_fn=ground_truth,
        origins=[1, 2, 3],
        horizon=1,
    )
    assert len(ground_truth_calls) == 3  # not 6 -- shared per origin, not per model
    assert np.all(result.scores["a"] == 0.0)
    assert np.all(result.scores["b"] > 0.0)


def test_severity_fn_is_wired_through():
    result = backtest_zero_overlap(
        lambda t: {"kind": "point", "Q": np.eye(2)},
        lambda s, e: np.full((2, 2), 0.3) + 0.7 * np.eye(2),
        [1, 2],
        horizon=1,
        severity_fn=lambda y: float(y[0, 1]),
    )
    assert result.severity is not None
    assert np.allclose(result.severity, 0.3)


def test_severity_is_none_when_not_requested():
    result = backtest_zero_overlap(lambda t: {"kind": "point", "Q": np.eye(2)}, lambda s, e: np.eye(2), [1], horizon=1)
    assert result.severity is None


@pytest.mark.parametrize("bad_horizon", [0, -1])
def test_invalid_horizon_raises(bad_horizon):
    with pytest.raises(ValueError):
        backtest_zero_overlap(lambda t: {"kind": "point", "Q": np.eye(2)}, lambda s, e: np.eye(2), [1], horizon=bad_horizon)


def test_negative_purge_gap_raises():
    with pytest.raises(ValueError):
        backtest_zero_overlap(
            lambda t: {"kind": "point", "Q": np.eye(2)}, lambda s, e: np.eye(2), [1], horizon=1, purge_gap=-1
        )


def test_custom_score_fn_is_used():
    calls = []

    def fake_score(forecast, y):
        calls.append((forecast, y))
        return 42.0

    result = backtest_zero_overlap(
        lambda t: {"kind": "point", "Q": np.eye(2)}, lambda s, e: np.eye(2), [1, 2], horizon=1, score_fn=fake_score
    )
    assert np.all(result.scores["model"] == 42.0)
    assert len(calls) == 2
