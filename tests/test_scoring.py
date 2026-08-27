"""Tests for matrix_energy_score and matrix_variogram_score: property
tests for the closed-form tiers (Sec. 2.4 of the design doc) plus
Monte Carlo cross-checks confirming each closed form actually matches
brute-force simulation of the object it claims to score exactly --
these formulas have no external package to check against (Sec. 5.2's
survey found none), so internal consistency plus simulation is the
actual evidence here, the same standard this project's own manuscripts
hold themselves to for a derived closed form.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from corrscore import matrix_energy_score, matrix_geodesic_variogram_score, matrix_variogram_score
from corrscore.scoring import _phi


def _rand_corr(k, rho, rng):
    off = rng.uniform(-0.05, 0.05, size=(k, k))
    off = 0.5 * (off + off.T)
    g = np.full((k, k), rho) + off
    np.fill_diagonal(g, 1.0)
    return g


def _equicorr(k, rho):
    g = np.full((k, k), rho)
    np.fill_diagonal(g, 1.0)
    return g


# ---------------------------------------------------------------------
# matrix_energy_score: point / mixture (Tier 1)
# ---------------------------------------------------------------------


def test_point_is_plain_frobenius_distance():
    q = _equicorr(4, 0.3)
    y = _equicorr(4, 0.5)
    expected = np.linalg.norm(q - y, ord="fro")
    assert matrix_energy_score({"kind": "point", "Q": q}, y) == pytest.approx(expected)


@given(k=st.integers(2, 6), rho1=st.floats(-0.4, 0.4), rho2=st.floats(-0.4, 0.4), p_c=st.floats(0.01, 0.99))
@settings(max_examples=50, deadline=None)
def test_two_point_mixture_matches_hand_derived_closed_form(k, rho1, rho2, p_c):
    """Eq. 2/3, transcribed independently of scoring.py's own
    implementation -- the actual cross-check, not a self-comparison."""
    rng = np.random.default_rng(0)
    qc, qs, y = _equicorr(k, rho1), _equicorr(k, rho2), _rand_corr(k, 0.5 * (rho1 + rho2), rng)
    d = lambda a, b: np.linalg.norm(a - b, ord="fro")
    p_s = 1 - p_c
    expected = p_c * d(qc, y) + p_s * d(qs, y) - p_c * p_s * d(qc, qs)
    got = matrix_energy_score({"kind": "mixture", "components": [(p_c, qc), (p_s, qs)]}, y)
    assert got == pytest.approx(expected, abs=1e-9)


def test_mixture_generalizes_to_more_than_two_atoms():
    """Tier 1's whole point: nothing in the derivation is specific to
    K=2 atoms. Cross-check a 4-atom mixture against a hand-written
    double sum over Eq. 4 directly."""
    rng = np.random.default_rng(1)
    k = 5
    atoms = [_rand_corr(k, rho, rng) for rho in (0.1, 0.3, 0.5, 0.7)]
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    y = _rand_corr(k, 0.4, rng)
    d = lambda a, b: np.linalg.norm(a - b, ord="fro")
    term1 = sum(w * d(a, y) for w, a in zip(weights, atoms))
    term2 = sum(weights[i] * weights[j] * d(atoms[i], atoms[j]) for i in range(4) for j in range(4))
    expected = term1 - 0.5 * term2
    got = matrix_energy_score({"kind": "mixture", "components": list(zip(weights, atoms))}, y)
    assert got == pytest.approx(expected, abs=1e-9)


@given(k=st.integers(2, 6), seed=st.integers(0, 10_000))
@settings(max_examples=30, deadline=None)
def test_mixture_is_zero_when_all_atoms_equal_y(k, seed):
    rng = np.random.default_rng(seed)
    y = _rand_corr(k, 0.3, rng)
    got = matrix_energy_score({"kind": "mixture", "components": [(0.5, y), (0.5, y)]}, y)
    assert got == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------
# matrix_energy_score: isotropic_gaussian_mixture (Tier 2)
# ---------------------------------------------------------------------


@given(k=st.integers(2, 6), sigma=st.floats(0.001, 0.3), seed=st.integers(0, 10_000))
@settings(max_examples=40, deadline=None)
def test_isotropic_gaussian_mixture_reduces_to_discrete_mixture_at_sigma_zero(k, sigma, seed):
    """The degenerate sigma=0 case must reproduce Tier 1 exactly -- a
    direct, cheap consistency check between the two closed forms,
    independent of any Monte Carlo noise."""
    rng = np.random.default_rng(seed)
    qc, qs, y = _rand_corr(k, 0.2, rng), _rand_corr(k, 0.6, rng), _rand_corr(k, 0.4, rng)
    mixture = matrix_energy_score({"kind": "mixture", "components": [(0.4, qc), (0.6, qs)]}, y)
    isotropic = matrix_energy_score(
        {"kind": "isotropic_gaussian_mixture", "components": [(0.4, qc, 0.0), (0.6, qs, 0.0)]}, y
    )
    assert isotropic == pytest.approx(mixture, abs=1e-8)


def test_isotropic_gaussian_mixture_matches_brute_force_monte_carlo():
    """The one property sigma=0 can't check: does Eq. 5's closed form
    actually match the *simulated* isotropic-Gaussian-scatter object it
    claims to describe, at a real (nonzero) sigma? Draws directly from
    the mixture (independent of scoring.py's own sampling helper) and
    scores the sample via the Tier 3 "ensemble" path, so this exercises
    two independently-motivated code paths against each other."""
    rng = np.random.default_rng(42)
    k = 5
    qc, qs = _equicorr(k, 0.2), _equicorr(k, 0.6)
    y = _equicorr(k, 0.35)
    sigma = 0.05
    iu = np.triu_indices(k, k=1)

    closed = matrix_energy_score(
        {"kind": "isotropic_gaussian_mixture", "components": [(0.4, qc, sigma), (0.6, qs, sigma)]}, y
    )

    n_draws, n_seeds = 4000, 30
    estimates = []
    for seed in range(n_seeds):
        r = np.random.default_rng(1000 + seed)
        idx = r.choice(2, size=n_draws, p=[0.4, 0.6])
        draws = []
        for i in idx:
            base = qc if i == 0 else qs
            u = base[iu] + r.normal(0, sigma, size=len(iu[0]))
            m = np.eye(k)
            m[iu] = u
            m.T[iu] = u
            draws.append(m)
        estimates.append(matrix_energy_score({"kind": "ensemble", "draws": draws}, y))
    mc_mean = np.mean(estimates)
    mc_se = np.std(estimates, ddof=1) / np.sqrt(n_seeds)
    assert abs(mc_mean - closed) < 4 * mc_se, (mc_mean, closed, mc_se)


def test_isotropic_gaussian_mixture_handles_k_equals_one_degenerate_case():
    """K=1: zero free entries (n=0). Must not raise or return NaN."""
    q = np.array([[1.0]])
    y = np.array([[1.0]])
    got = matrix_energy_score({"kind": "isotropic_gaussian_mixture", "components": [(1.0, q, 0.1)]}, y)
    assert got == pytest.approx(0.0)


# ---------------------------------------------------------------------
# matrix_energy_score: ensemble (Tier 3)
# ---------------------------------------------------------------------


def test_ensemble_matches_hand_written_double_loop():
    rng = np.random.default_rng(2)
    draws = [rng.standard_normal((3, 3)) for _ in range(12)]
    y = rng.standard_normal((3, 3))
    d = lambda a, b: np.linalg.norm(a - b, ord="fro")
    m = len(draws)
    term1 = np.mean([d(x, y) for x in draws])
    term2 = sum(d(draws[i], draws[j]) for i in range(m) for j in range(m)) / (2 * m**2)
    expected = term1 - term2
    got = matrix_energy_score({"kind": "ensemble", "draws": draws}, y)
    assert got == pytest.approx(expected, abs=1e-9)


def test_ensemble_of_one_draw_equals_point_score():
    rng = np.random.default_rng(3)
    q = rng.standard_normal((4, 4))
    y = rng.standard_normal((4, 4))
    ensemble = matrix_energy_score({"kind": "ensemble", "draws": [q]}, y)
    point = matrix_energy_score({"kind": "point", "Q": q}, y)
    assert ensemble == pytest.approx(point)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        matrix_energy_score({"kind": "nonsense"}, np.eye(3))


# ---------------------------------------------------------------------
# matrix_variogram_score
# ---------------------------------------------------------------------


def test_variogram_score_is_zero_for_perfect_point_forecast():
    y = _equicorr(5, 0.4)
    assert matrix_variogram_score({"kind": "point", "Q": y}, y) == pytest.approx(0.0, abs=1e-10)


def test_variogram_score_mixture_matches_hand_derived_weighted_average():
    """The mixture case is exact (module docstring): E|X_i-X_j|^p is
    just the atom-weighted average of the deterministic per-atom
    values. Cross-check against that formula written out by hand."""
    rng = np.random.default_rng(4)
    k = 4
    qc, qs = _rand_corr(k, 0.1, rng), _rand_corr(k, 0.7, rng)
    y = _rand_corr(k, 0.4, rng)
    p = 0.5
    iu = np.triu_indices(k, k=1)
    y_u = y[iu]
    qc_u, qs_u = qc[iu], qs[iu]
    y_diff = np.abs(y_u[:, None] - y_u[None, :]) ** p
    exp_diff = 0.4 * np.abs(qc_u[:, None] - qc_u[None, :]) ** p + 0.6 * np.abs(qs_u[:, None] - qs_u[None, :]) ** p
    expected = float(np.sum((y_diff - exp_diff) ** 2))
    got = matrix_variogram_score({"kind": "mixture", "components": [(0.4, qc), (0.6, qs)]}, y, p=p)
    assert got == pytest.approx(expected, abs=1e-8)


def test_variogram_score_ensemble_uses_draws_directly_not_extra_sampling():
    """For "ensemble", the Monte Carlo estimate should use the supplied
    draws deterministically (no extra randomness), so two calls with
    the same draws must be identical."""
    rng = np.random.default_rng(5)
    k = 4
    draws = [_rand_corr(k, rho, rng) for rho in np.linspace(0.1, 0.6, 20)]
    y = _rand_corr(k, 0.4, rng)
    a = matrix_variogram_score({"kind": "ensemble", "draws": draws}, y)
    b = matrix_variogram_score({"kind": "ensemble", "draws": draws}, y)
    assert a == pytest.approx(b)


@given(k=st.integers(3, 6), seed=st.integers(0, 10_000))
@settings(max_examples=30, deadline=None)
def test_variogram_score_is_nonnegative(k, seed):
    rng = np.random.default_rng(seed)
    q, y = _rand_corr(k, 0.3, rng), _rand_corr(k, 0.5, rng)
    assert matrix_variogram_score({"kind": "point", "Q": q}, y) >= 0.0


# ---------------------------------------------------------------------
# matrix_geodesic_variogram_score
# ---------------------------------------------------------------------


@given(rho=st.floats(-0.999, 0.999))
@settings(max_examples=200, deadline=None)
def test_phi_is_finite_and_odd(rho):
    """phi(-rho) == -phi(rho): the construction (geometric-correlation-
    scoring-explainer.html Eq. 8) is explicitly a SIGNED distance, and
    everything downstream (strict monotonicity, the Fisher z-transform
    analogy) depends on this holding exactly, not approximately."""
    assert np.isfinite(_phi(rho))
    assert _phi(-rho) == pytest.approx(-_phi(rho))


def test_phi_is_strictly_increasing():
    """The mathematical property the whole construction rests on
    (geodesic-scoring-rules-theoretical-foundations.html Sec. 5's
    "warped ruler" argument only makes sense for a monotonic warp) --
    checked directly, not assumed, matching this project's own
    "verify, don't assume" discipline for exactly this claim."""
    rhos = np.linspace(-0.999, 0.999, 4001)
    assert np.all(np.diff(_phi(rhos)) > 0)


def test_phi_matches_leading_order_taylor_expansion_near_zero():
    """phi(rho) = rho + O(rho^3) near rho=0 (log(1+-rho) ~ +-rho -
    rho^2/2 to leading order, so the two log-squared terms combine to
    give phi(rho) ~ rho): the geodesic score should reduce to the flat
    score for near-independent correlations, not just "look similar"."""
    for rho in (1e-4, 1e-3, 1e-2):
        assert _phi(rho) == pytest.approx(rho, rel=1e-3)


def test_geodesic_variogram_score_matches_flat_score_via_explicit_phi_transform():
    """Independent cross-check (not a self-comparison of the
    implementation against itself): apply the closed-form phi to Q and y
    by hand, then call the *flat* matrix_variogram_score directly on the
    transformed matrices; this must equal matrix_geodesic_variogram_score
    applied to the untransformed inputs."""
    rng = np.random.default_rng(6)
    k = 5
    q, y = _rand_corr(k, 0.4, rng), _rand_corr(k, 0.6, rng)

    def phi_matrix(m):
        m = m.copy()
        np.fill_diagonal(m, 0.0)
        return np.sign(m) * np.sqrt(0.5 * (np.log1p(np.abs(m)) ** 2 + np.log1p(-np.abs(m)) ** 2))

    expected = matrix_variogram_score({"kind": "point", "Q": phi_matrix(q)}, phi_matrix(y))
    got = matrix_geodesic_variogram_score({"kind": "point", "Q": q}, y)
    assert got == pytest.approx(expected, abs=1e-9)


def test_geodesic_variogram_score_is_zero_for_perfect_point_forecast():
    y = _equicorr(5, 0.4)
    assert matrix_geodesic_variogram_score({"kind": "point", "Q": y}, y) == pytest.approx(0.0, abs=1e-10)


def test_geodesic_variogram_score_mixture_matches_hand_derived_weighted_average():
    """Same structure as the flat mixture cross-check, but with phi
    applied to every entry first -- an independent transcription, not a
    call into the package's own _phi helper."""
    rng = np.random.default_rng(7)
    k = 4
    qc, qs = _rand_corr(k, 0.1, rng), _rand_corr(k, 0.7, rng)
    y = _rand_corr(k, 0.4, rng)
    p = 0.5

    def phi(r):
        a = np.abs(r)
        return np.sign(r) * np.sqrt(0.5 * (np.log1p(a) ** 2 + np.log1p(-a) ** 2))

    iu = np.triu_indices(k, k=1)
    y_u, qc_u, qs_u = phi(y[iu]), phi(qc[iu]), phi(qs[iu])
    y_diff = np.abs(y_u[:, None] - y_u[None, :]) ** p
    exp_diff = 0.4 * np.abs(qc_u[:, None] - qc_u[None, :]) ** p + 0.6 * np.abs(qs_u[:, None] - qs_u[None, :]) ** p
    expected = float(np.sum((y_diff - exp_diff) ** 2))
    got = matrix_geodesic_variogram_score({"kind": "mixture", "components": [(0.4, qc), (0.6, qs)]}, y, p=p)
    assert got == pytest.approx(expected, abs=1e-8)


def test_geodesic_variogram_score_ensemble_uses_draws_directly_not_extra_sampling():
    rng = np.random.default_rng(8)
    k = 4
    draws = [_rand_corr(k, rho, rng) for rho in np.linspace(0.1, 0.6, 20)]
    y = _rand_corr(k, 0.4, rng)
    a = matrix_geodesic_variogram_score({"kind": "ensemble", "draws": draws}, y)
    b = matrix_geodesic_variogram_score({"kind": "ensemble", "draws": draws}, y)
    assert a == pytest.approx(b)


@given(k=st.integers(3, 6), seed=st.integers(0, 10_000))
@settings(max_examples=30, deadline=None)
def test_geodesic_variogram_score_is_nonnegative(k, seed):
    rng = np.random.default_rng(seed)
    q, y = _rand_corr(k, 0.3, rng), _rand_corr(k, 0.5, rng)
    assert matrix_geodesic_variogram_score({"kind": "point", "Q": q}, y) >= 0.0


@given(k=st.integers(3, 6), seed=st.integers(0, 10_000))
@settings(max_examples=30, deadline=None)
def test_geodesic_variogram_score_approx_equals_flat_score_for_small_correlations(k, seed):
    """phi(rho) ~ rho near 0 (verified directly above), so for inputs
    confined to a small neighborhood of independence the geodesic and
    flat variogram scores should agree closely -- the construction's
    own consistency check, not just an isolated property of phi."""
    rng = np.random.default_rng(seed)
    q = _rand_corr(k, 0.0, rng) * 0.05
    y = _rand_corr(k, 0.0, rng) * 0.05
    np.fill_diagonal(q, 1.0)
    np.fill_diagonal(y, 1.0)
    flat = matrix_variogram_score({"kind": "point", "Q": q}, y)
    geo = matrix_geodesic_variogram_score({"kind": "point", "Q": q}, y)
    assert geo == pytest.approx(flat, rel=0.05, abs=1e-6)


def test_geodesic_variogram_score_unknown_kind_raises():
    with pytest.raises(ValueError):
        matrix_geodesic_variogram_score({"kind": "nonsense"}, np.eye(3))
