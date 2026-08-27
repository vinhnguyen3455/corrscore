"""Circular block-bootstrap significance testing on paired per-origin
score differentials, via `arch.bootstrap.CircularBlockBootstrap` -- kept
as a real dependency rather than vendored, since `arch` clears this
package's "widely used, actively maintained, field-standard" bar for
reuse, and block-bootstrap correctness (edge handling, unbiased block
placement) carries real reimplementation risk for a "just wrap it
correctly" component.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from arch.bootstrap import CircularBlockBootstrap

__all__ = ["BootstrapResult", "circular_block_bootstrap"]


class BootstrapResult(NamedTuple):
    """Result of `circular_block_bootstrap` for one block length.

    Attributes
    ----------
    obs : float
        The observed mean of `scores_a - scores_b`.
    ci_lo, ci_hi : float
        95% percentile confidence interval from the bootstrap
        distribution of that mean.
    p_value : float
        Two-sided bootstrap p-value for H0: true mean difference = 0.
    significant : bool
        `ci_lo > 0 or ci_hi < 0` -- whether zero falls outside the 95%
        interval.
    block_len : int
    n_boot : int
    """

    obs: float
    ci_lo: float
    ci_hi: float
    p_value: float
    significant: bool
    block_len: int
    n_boot: int


def circular_block_bootstrap(
    scores_a: npt.ArrayLike,
    scores_b: npt.ArrayLike,
    block_lengths: list[int],
    n_boot: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> dict[int, BootstrapResult]:
    """Percentile circular block bootstrap on the paired differential
    `d_i = scores_a[i] - scores_b[i]`, swept across `block_lengths` as a
    sensitivity check rather than relying on one automatically "optimal"
    block length.

    Returns a dict keyed by each entry of `block_lengths`. The most
    conservative (largest-p) entry is the one worth reporting as the
    headline result.
    """
    diffs = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    results: dict[int, BootstrapResult] = {}
    for block_len in block_lengths:
        bs = CircularBlockBootstrap(block_len, diffs, seed=seed)
        boot_means = bs.apply(lambda z: np.mean(z), n_boot).ravel()
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        tail_frac = min(float(np.mean(boot_means <= 0)), float(np.mean(boot_means >= 0)))
        p_value = min(1.0, 2.0 * tail_frac)
        results[block_len] = BootstrapResult(
            obs=float(diffs.mean()),
            ci_lo=float(ci_lo),
            ci_hi=float(ci_hi),
            p_value=p_value,
            significant=bool(ci_lo > 0 or ci_hi < 0),
            block_len=block_len,
            n_boot=n_boot,
        )
    return results
