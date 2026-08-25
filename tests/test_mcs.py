"""Tests for model_confidence_set: property tests plus a verdict-level
(not byte-exact) cross-check against R's MCS::MCSprocedure -- the
honest bar this specific vendored implementation gets, per mcs.py's own
module docstring: this package's range-statistic elimination algorithm
is its own reasonable operationalization of Hansen, Lunde & Nason
(2011), not a line-by-line port, so agreement is checked on which
models survive (an unambiguous verdict by construction on the fixture
below), not on matching statistics/p-values exactly.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from corrscore import model_confidence_set

_spec = importlib.util.spec_from_file_location(
    "mcs_oracle_case", Path(__file__).parent / "_reference" / "mcs_oracle_case.py"
)
_mcs_oracle_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mcs_oracle_module)
MCS_LOSS_MATRIX = _mcs_oracle_module.MCS_LOSS_MATRIX
MCS_ORACLE_SURVIVORS = _mcs_oracle_module.MCS_ORACLE_SURVIVORS


def test_matches_r_mcsprocedure_verdict_on_fixed_case():
    result = model_confidence_set(MCS_LOSS_MATRIX, alpha=0.10, block_len=5, n_boot=2000, seed=7)
    assert set(result.survivors) == MCS_ORACLE_SURVIVORS


def test_identical_loss_models_all_survive():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(100) + 3.0
    scores = {"a": base, "b": base, "c": base}
    result = model_confidence_set(scores, alpha=0.10, block_len=5, n_boot=500, seed=0)
    assert set(result.survivors) == {"a", "b", "c"}
    assert result.eliminated == []
    assert result.final_p_value == 1.0


def test_one_clearly_dominant_model_isolates_the_rest():
    rng = np.random.default_rng(1)
    n = 200
    scores = {
        "best": rng.standard_normal(n) * 0.2 + 1.0,
        "worst": rng.standard_normal(n) * 0.2 + 5.0,
        "worse": rng.standard_normal(n) * 0.2 + 4.0,
    }
    result = model_confidence_set(scores, alpha=0.10, block_len=5, n_boot=1000, seed=1)
    assert result.survivors == ["best"]
    eliminated_names = {name for name, _ in result.eliminated}
    assert eliminated_names == {"worst", "worse"}


def test_at_least_two_models_required():
    with pytest.raises(ValueError):
        model_confidence_set({"only_one": np.array([1.0, 2.0, 3.0])})


def test_eliminated_order_is_worst_first():
    rng = np.random.default_rng(2)
    n = 150
    scores = {
        "good": rng.standard_normal(n) * 0.1 + 1.0,
        "bad": rng.standard_normal(n) * 0.1 + 2.0,
        "terrible": rng.standard_normal(n) * 0.1 + 5.0,
    }
    result = model_confidence_set(scores, alpha=0.10, block_len=5, n_boot=1000, seed=2)
    assert result.eliminated[0][0] == "terrible"
