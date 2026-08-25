from __future__ import annotations

import numpy as np

from corrscore import circular_block_bootstrap


def test_identical_series_gives_zero_observed_difference_and_no_significance():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(80)
    results = circular_block_bootstrap(a, a, block_lengths=[1, 5], n_boot=500, seed=0)
    for r in results.values():
        assert r.obs == 0.0
        assert not r.significant


def test_large_systematic_difference_is_detected_significant():
    rng = np.random.default_rng(1)
    a = 5.0 + rng.standard_normal(100) * 0.2
    b = rng.standard_normal(100) * 0.2
    results = circular_block_bootstrap(a, b, block_lengths=[1, 3, 6], n_boot=1000, seed=1)
    for block_len, r in results.items():
        assert r.significant, block_len
        assert r.p_value < 0.01, block_len
        assert r.ci_lo > 0, block_len


def test_confidence_interval_contains_observed_mean():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(60)
    b = rng.standard_normal(60)
    results = circular_block_bootstrap(a, b, block_lengths=[4], n_boot=1000, seed=2)
    r = results[4]
    assert r.ci_lo <= r.obs <= r.ci_hi


def test_results_keyed_by_every_requested_block_length():
    a = np.arange(20, dtype=float)
    b = np.arange(20, dtype=float) * 0.5
    results = circular_block_bootstrap(a, b, block_lengths=[1, 2, 5, 10], n_boot=200, seed=3)
    assert set(results.keys()) == {1, 2, 5, 10}
