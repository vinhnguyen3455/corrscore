"""Matrix-aware proper scoring rules for correlation/covariance-matrix
forecasts: the energy score (Gneiting & Raftery, 2007) and the variogram
score (Scheuerer & Hamill, 2015), both adapted to score a full K x K
matrix rather than the vector-valued observation either was originally
published for.

Forecast representation (see `matrix_energy_score` and
`matrix_variogram_score` docstrings for the exact shape of each): a
plain dict tagged by `"kind"`, dispatching across the closed-form
tractability spectrum documented in `corrscore-package-design.html`
Sec. 2.4 --

  - "point":                     a single deterministic matrix (M=1).
  - "mixture":                   any number of discrete atoms (Tier 1,
                                  Eq. 4) -- exact, O(K_atoms^2) cost, no
                                  simulation, for any number of atoms
                                  (not hard-coded to two regimes).
  - "isotropic_gaussian_mixture": discrete atoms each with an isotropic
                                  Gaussian scatter (Tier 2, Eq. 5) --
                                  exact for the ENERGY score via the
                                  confluent hypergeometric mean-norm
                                  formula; only ever reachable through
                                  this explicit kind, never inferred
                                  from a plain ensemble automatically,
                                  because the isotropy assumption is a
                                  real one (see the module's own honest
                                  caveat in the design doc).
  - "ensemble":                   a general Monte Carlo draw set (Tier
                                  3) -- the only kind with no closed
                                  form; used for anything else,
                                  including RM-DCC's own simulated
                                  multivariate-t paths.

`matrix_variogram_score` has a narrower closed form than the energy
score: exact for "point" and "mixture" (both are fully determined by a
finite set of deterministic atoms), Monte-Carlo-estimated for "ensemble"
and "isotropic_gaussian_mixture" -- no closed form for a fractional
absolute moment of a Gaussian difference has been derived for this
package; sampling is the honest v1 answer for those two kinds.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from scipy.special import gammaln, hyp1f1

Matrix = npt.NDArray[np.float64]
Forecast = Mapping[str, Any]

__all__ = ["matrix_energy_score", "matrix_variogram_score"]


def _frobenius(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Full K x K Frobenius distance -- matches the convention already
    used throughout this project's real-data scripts
    (`scripts/regime-detection-pfa-filter.py`'s `energy_score_mix`/
    `energy_score_point`), not the upper-triangle-only convention
    `matrix_variogram_score` uses (see that function's own docstring for
    why the two conventions deliberately differ)."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), ord="fro"))


def _upper(m: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The K(K-1)/2 free upper-triangle entries (diagonal, always 1 for
    a correlation matrix, and the mirrored lower triangle both excluded
    -- see corrscore-package-design.html Sec. 2.5's design decision)."""
    arr = np.asarray(m, dtype=float)
    k = arr.shape[0]
    return arr[np.triu_indices(k, k=1)]


def _expected_norm_isotropic_gaussian(norm_mu_sq: float, sigma: float, n: int) -> float:
    """E[||Z||] for Z ~ N(mu, sigma^2 I_n), via the noncentral-chi mean
    formula (Johnson, Kotz & Balakrishnan, 1994; corrscore-package-
    design.html Eq. 5), evaluated through the confluent hypergeometric
    function `scipy.special.hyp1f1` rather than simulated.

    `norm_mu_sq` is ||mu||^2 -- the formula only depends on mu through
    this scalar, by spherical symmetry. `n == 0` (a degenerate,
    parameter-free space, e.g. K=1) returns 0 directly.
    """
    if n <= 0:
        return 0.0
    if sigma <= 0.0:
        return float(np.sqrt(norm_mu_sq))
    log_ratio = gammaln((n + 1) / 2.0) - gammaln(n / 2.0)
    coef = sigma * np.sqrt(2.0) * np.exp(log_ratio)
    return float(coef * hyp1f1(-0.5, n / 2.0, -norm_mu_sq / (2.0 * sigma**2)))


def _expected_matrix_norm_isotropic(diff_free_entries: npt.NDArray[np.float64], sigma: float) -> float:
    """E[||A||_F] where A is the (implicit) symmetric, zero-diagonal K x
    K matrix built by placing an isotropic Gaussian free-entry vector u
    ~ N(diff_free_entries, sigma^2 I_n) (n = K(K-1)/2) into both the
    upper and lower triangle. Each free entry appears twice in A (once
    per triangle), so ||A||_F = sqrt(2) * ||u||_2 -- this is Eq. 5 with
    that bookkeeping factor folded in, not a separate derivation."""
    n = diff_free_entries.size
    norm_mu_sq = float(diff_free_entries @ diff_free_entries)
    return np.sqrt(2.0) * _expected_norm_isotropic_gaussian(norm_mu_sq, sigma, n)


def _energy_score_point(q: npt.ArrayLike, y: npt.ArrayLike) -> float:
    return _frobenius(q, y)


def _energy_score_mixture(components: Sequence[tuple[float, npt.ArrayLike]], y: npt.ArrayLike) -> float:
    weights = np.array([float(p) for p, _ in components])
    mats = [np.asarray(q, dtype=float) for _, q in components]
    k = len(mats)
    term1 = sum(weights[i] * _frobenius(mats[i], y) for i in range(k))
    term2 = 0.0
    for i in range(k):
        for j in range(k):
            term2 += weights[i] * weights[j] * _frobenius(mats[i], mats[j])
    return float(term1 - 0.5 * term2)


def _energy_score_isotropic_gaussian_mixture(
    components: Sequence[tuple[float, npt.ArrayLike, float]], y: npt.ArrayLike
) -> float:
    weights = np.array([float(p) for p, _, _ in components])
    mats_u = [_upper(q) for _, q, _ in components]
    sigmas = [float(s) for _, _, s in components]
    y_u = _upper(y)
    k = len(components)

    term1 = 0.0
    for i in range(k):
        term1 += weights[i] * _expected_matrix_norm_isotropic(mats_u[i] - y_u, sigmas[i])

    term2 = 0.0
    for i in range(k):
        for j in range(k):
            combined_sigma = float(np.sqrt(sigmas[i] ** 2 + sigmas[j] ** 2))
            term2 += weights[i] * weights[j] * _expected_matrix_norm_isotropic(mats_u[i] - mats_u[j], combined_sigma)

    return float(term1 - 0.5 * term2)


def _energy_score_ensemble(draws: Sequence[npt.ArrayLike], y: npt.ArrayLike) -> float:
    flat = np.stack([np.asarray(d, dtype=float).ravel() for d in draws])
    y_flat = np.asarray(y, dtype=float).ravel()
    m = flat.shape[0]
    term1 = float(np.mean(np.linalg.norm(flat - y_flat, axis=1)))
    if m < 2:
        return term1
    pairwise = pdist(flat, metric="euclidean")
    term2 = float(pairwise.sum() / (m**2))
    return term1 - term2


def matrix_energy_score(forecast: Forecast, y: npt.ArrayLike) -> float:
    """The energy score (Eq. 1), dispatched across the closed-form
    spectrum (module docstring) by `forecast["kind"]`:

    - {"kind": "point", "Q": Q} -> Eq. 1's M=1 special case, plain
      Frobenius distance.
    - {"kind": "mixture", "components": [(p_1, Q_1), ..., (p_K, Q_K)]}
      -> Tier 1, Eq. 4. Any number of atoms, not just two.
    - {"kind": "isotropic_gaussian_mixture",
       "components": [(p_1, Q_1, sigma_1), ...]} -> Tier 2, Eq. 5.
      `sigma_k` is the per-component scatter in the K(K-1)/2-dimensional
      free-entry space (corrscore-package-design.html Sec. 5.1), not in
      the full K x K ambient space.
    - {"kind": "ensemble", "draws": [Q_1, ..., Q_M]} -> Tier 3, the
      general Monte Carlo form (Eq. 1), O(M^2) pairwise distances.

    `y` is the realized K x K correlation (or covariance) matrix.
    """
    kind = forecast["kind"]
    if kind == "point":
        return _energy_score_point(forecast["Q"], y)
    if kind == "mixture":
        return _energy_score_mixture(forecast["components"], y)
    if kind == "isotropic_gaussian_mixture":
        return _energy_score_isotropic_gaussian_mixture(forecast["components"], y)
    if kind == "ensemble":
        return _energy_score_ensemble(forecast["draws"], y)
    raise ValueError(f"Unknown forecast kind: {kind!r}")


def _variogram_diff_p_point(q: npt.ArrayLike, p: float) -> npt.NDArray[np.float64]:
    q_u = _upper(q)
    return np.abs(q_u[:, None] - q_u[None, :]) ** p


def _variogram_diff_p_mixture(components: Sequence[tuple[float, npt.ArrayLike]], p: float) -> npt.NDArray[np.float64]:
    """Exact: a discrete mixture's entries are fully determined once the
    atom is known, so E|X_i - X_j|^p is just the atom-weighted average
    of the deterministic per-atom values -- no sampling needed."""
    weights = [float(w) for w, _ in components]
    total = None
    for w, q in zip(weights, (c[1] for c in components)):
        term = w * _variogram_diff_p_point(q, p)
        total = term if total is None else total + term
    assert total is not None
    return total


def _sample_upper_triangle(forecast: Forecast, n_samples: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """Draws used only by `matrix_variogram_score`'s Monte Carlo path
    (see module docstring for why "ensemble" and
    "isotropic_gaussian_mixture" have no closed form here). "ensemble"
    reuses its existing draws directly (deterministic, no extra
    sampling); "isotropic_gaussian_mixture" is genuinely sampled."""
    kind = forecast["kind"]
    if kind == "ensemble":
        return np.stack([_upper(d) for d in forecast["draws"]])
    if kind == "isotropic_gaussian_mixture":
        components = forecast["components"]
        weights = np.array([float(w) for w, _, _ in components])
        weights = weights / weights.sum()
        atom_idx = rng.choice(len(components), size=n_samples, p=weights)
        n_free = _upper(components[0][1]).size
        out = np.empty((n_samples, n_free))
        for k, (_, q, sigma) in enumerate(components):
            mask = atom_idx == k
            count = int(mask.sum())
            if count == 0:
                continue
            noise = rng.normal(0.0, float(sigma), size=(count, n_free)) if sigma > 0 else 0.0
            out[mask] = _upper(q) + noise
        return out
    raise ValueError(f"Forecast kind {kind!r} has no Monte Carlo sampling path")


def matrix_variogram_score(
    forecast: Forecast,
    y: npt.ArrayLike,
    p: float = 0.5,
    weights: npt.ArrayLike | None = None,
    n_samples: int = 500,
    random_state: int | np.random.Generator | None = None,
) -> float:
    """The variogram score (Eq. 6), adapted to index i, j over the
    K(K-1)/2 free upper-triangle entries of the correlation matrix
    (corrscore-package-design.html Sec. 2.5's design decision) rather
    than the K original series the published formula was defined for.

    Exact (no sampling) for "point" and "mixture" forecasts. For
    "ensemble" and "isotropic_gaussian_mixture", `E|X_i - X_j|^p` is
    estimated by Monte Carlo (`n_samples` draws) -- no closed form for
    this fractional absolute moment has been derived for this package
    (unlike `matrix_energy_score`'s Tier 2 path); see the module
    docstring. Cost and memory scale as O(n_samples * n_entries^2),
    where n_entries = K(K-1)/2 -- the default `n_samples=500` keeps this
    modest through K=16 (n_entries=120); pass a smaller value for larger
    K if memory becomes a concern.

    `weights` (if given) must be an (n_entries, n_entries) array;
    defaults to uniform (all-ones), matching Scheuerer & Hamill's own
    default and the published formula literally.
    """
    y_u = _upper(y)
    n = y_u.size
    y_diff_p = np.abs(y_u[:, None] - y_u[None, :]) ** p
    w = np.ones((n, n)) if weights is None else np.asarray(weights, dtype=float)

    kind = forecast["kind"]
    if kind == "point":
        exp_diff_p = _variogram_diff_p_point(forecast["Q"], p)
    elif kind == "mixture":
        exp_diff_p = _variogram_diff_p_mixture(forecast["components"], p)
    elif kind in ("ensemble", "isotropic_gaussian_mixture"):
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        samples = _sample_upper_triangle(forecast, n_samples, rng)
        exp_diff_p = (np.abs(samples[:, :, None] - samples[:, None, :]) ** p).mean(axis=0)
    else:
        raise ValueError(f"Unknown forecast kind: {kind!r}")

    return float(np.sum(w * (y_diff_p - exp_diff_p) ** 2))
