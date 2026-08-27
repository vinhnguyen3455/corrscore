"""The Model Confidence Set (Hansen, Lunde & Nason, 2011), vendored
rather than depending on `model-confidence-set` (JLDC's GitHub-only,
no-PyPI-release Python port): reproduced directly from the paper, with
R's actively-maintained CRAN `MCS` package (Catania & Bernardi) and the
JLDC port used only as development-time correctness references, never
imported at runtime.

Implements the range statistic (T_R) elimination algorithm: repeatedly
test whether the current candidate set is statistically distinguishable
from its own best member; if so, drop the single worst-performing model
and repeat, until the surviving set cannot be rejected. The null
distribution and each round's studentizing standard errors both come
from the SAME joint circular block bootstrap of the loss matrix
(reusing `arch.bootstrap.CircularBlockBootstrap`, matching this
package's dependency policy).

Honest scope note: this is this package's own reasonable
operationalization of Hansen et al.'s range statistic and elimination
rule, not a literal line-by-line transcription of any specific existing
implementation's internal choices (e.g. R's `MCS` package's automatic
block-length selection is not reproduced -- `block_len` is a required,
caller-chosen parameter here, swept manually if robustness to it
matters, the same discipline `circular_block_bootstrap` already uses).
The test suite cross-checks this implementation's *verdict* (which
models survive) against R's `MCS::MCSprocedure` on fixed synthetic
data where the correct verdict is unambiguous by construction, not
byte-exact statistic/p-value agreement -- see
`tests/test_mcs.py` for why that is the honest bar for this
specific dependency.
"""
from __future__ import annotations

from typing import Mapping, NamedTuple

import numpy as np
import numpy.typing as npt
from arch.bootstrap import CircularBlockBootstrap

__all__ = ["MCSResult", "model_confidence_set"]


class MCSResult(NamedTuple):
    """Result of `model_confidence_set`.

    Attributes
    ----------
    survivors : list of str
        Models statistically indistinguishable from the best one at
        `alpha`, in no particular order.
    alpha : float
    eliminated : list of (str, float)
        Models removed, in elimination order, paired with the round's
        p-value at the moment of that model's removal.
    final_p_value : float
        The surviving set's own p-value (>= alpha; or 1.0 if only one
        model ever remained, a case in which rejection is trivially
        impossible).
    """

    survivors: list[str]
    alpha: float
    eliminated: list[tuple[str, float]]
    final_p_value: float


def _round_statistics(
    sub_loss: npt.NDArray[np.float64], block_len: int, n_boot: int, seed: int | np.random.Generator | None
) -> tuple[npt.NDArray[np.float64], float]:
    """One elimination round: pairwise studentized statistics for the
    current model set, and the bootstrap p-value for the joint range
    null. Returns `(t_obs, p_value)`, `t_obs` shape (m, m)."""
    m = sub_loss.shape[1]
    dbar = np.array([[np.mean(sub_loss[:, i] - sub_loss[:, j]) for j in range(m)] for i in range(m)])

    bs = CircularBlockBootstrap(block_len, sub_loss, seed=seed)
    boot_dbar = np.empty((n_boot, m, m))
    for rep, (pos, _kw) in enumerate(bs.bootstrap(n_boot)):
        resampled = pos[0]
        boot_dbar[rep] = np.array(
            [[np.mean(resampled[:, i] - resampled[:, j]) for j in range(m)] for i in range(m)]
        )

    se = boot_dbar.std(axis=0, ddof=1)
    se[se == 0] = np.inf  # identical-loss pairs (including the diagonal) -> t_ij = 0, never inf/nan
    t_obs = dbar / se
    t_range_obs = float(np.abs(t_obs).max())

    t_boot = (boot_dbar - dbar[None, :, :]) / se[None, :, :]
    t_range_boot = np.abs(t_boot).max(axis=(1, 2))
    p_value = float(np.mean(t_range_boot >= t_range_obs))
    return t_obs, p_value


def model_confidence_set(
    scores: Mapping[str, npt.ArrayLike],
    alpha: float = 0.10,
    block_len: int = 5,
    n_boot: int = 1000,
    seed: int | np.random.Generator | None = None,
) -> MCSResult:
    """The Model Confidence Set via the range statistic.

    Parameters
    ----------
    scores : dict of str -> array-like
        Per-model, per-origin losses, all the same length (e.g.
        `BacktestResult.scores`). At least two models required.
    alpha : float, default=0.10
    block_len : int, default=5
        Circular block-bootstrap block length. Required and
        caller-chosen (see module docstring) -- rerun with different
        values to check robustness, matching `circular_block_bootstrap`.
    n_boot : int, default=1000
    seed : int, np.random.Generator, or None

    Returns
    -------
    MCSResult
    """
    names = list(scores.keys())
    if len(names) < 2:
        raise ValueError("model_confidence_set needs at least two models")
    loss = np.column_stack([np.asarray(scores[name], dtype=float) for name in names])

    current = list(range(len(names)))
    eliminated: list[tuple[str, float]] = []
    p_value = 1.0

    while True:
        if len(current) == 1:
            p_value = 1.0
            break
        sub_names = [names[i] for i in current]
        t_obs, p_value = _round_statistics(loss[:, current], block_len, n_boot, seed)
        if p_value >= alpha:
            break
        avg_t = t_obs.mean(axis=1)
        worst_local = int(np.argmax(avg_t))
        eliminated.append((sub_names[worst_local], p_value))
        current.pop(worst_local)

    survivors = [names[i] for i in current]
    return MCSResult(survivors=survivors, alpha=alpha, eliminated=eliminated, final_p_value=p_value)
