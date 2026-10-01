"""Closed-form scores versus the Monte Carlo ensemble fallback.

Scores the same forecast twice: once through its exact kind ("mixture" or
"isotropic_gaussian_mixture") and once as an "ensemble" of draws from it.
Run with:  python examples/closed_form_vs_mc.py
"""
import numpy as np
from corrscore import matrix_energy_score, matrix_variogram_score

K, M, SEEDS, SIGMA, P_VS = 4, 2000, 50, 0.05, 0.5
IU = np.triu_indices(K, k=1)

def corr(entries):
    m = np.eye(K); m[IU] = entries; m.T[IU] = entries
    return m

q_calm = corr([0.10, 0.25, 0.15, 0.30, 0.05, 0.20])
q_stress = corr([0.55, 0.65, 0.50, 0.70, 0.60, 0.45])
y = corr([0.20, 0.35, 0.25, 0.40, 0.15, 0.30])
weights = np.array([0.7, 0.3])
atoms = [q_calm, q_stress]

mixture = {"kind": "mixture", "components": [(0.7, q_calm), (0.3, q_stress)]}
iso = {"kind": "isotropic_gaussian_mixture",
       "components": [(0.7, q_calm, SIGMA), (0.3, q_stress, SIGMA)]}

def draw_ensemble(rng, sigma):
    """M draws: pick an atom, add symmetric free-entry Gaussian scatter."""
    out = []
    for k in rng.choice(2, size=M, p=weights):
        u = atoms[k][IU] + (rng.normal(0.0, sigma, IU[0].size) if sigma > 0 else 0.0)
        out.append(corr(u))
    return {"kind": "ensemble", "draws": out}

def row(label, exact, estimates):
    est = np.asarray(estimates)
    se = est.std(ddof=1) / np.sqrt(est.size)
    print(f"{label:<30s} exact={exact:.5f}  ens={est.mean():.5f}+-{se:.5f}  "
          f"({(est.mean() - exact) / se:+.1f} se)")

rngs = [np.random.default_rng(100 + s) for s in range(SEEDS)]
print(f"K={K}, ensemble size M={M}, mean over {SEEDS} seeds")

row("energy, mixture", matrix_energy_score(mixture, y),
    [matrix_energy_score(draw_ensemble(r, 0.0), y) for r in rngs])
rngs = [np.random.default_rng(200 + s) for s in range(SEEDS)]
row("energy, isotropic Gauss. mix.", matrix_energy_score(iso, y),
    [matrix_energy_score(draw_ensemble(r, SIGMA), y) for r in rngs])
rngs = [np.random.default_rng(300 + s) for s in range(SEEDS)]
row("variogram, mixture", matrix_variogram_score(mixture, y, p=P_VS),
    [matrix_variogram_score(draw_ensemble(r, 0.0), y, p=P_VS) for r in rngs])

# No closed form exists for the isotropic Gaussian mixture's variogram score:
# both sides are Monte Carlo. Compare the kind's own sampler to an ensemble.
rngs = [np.random.default_rng(400 + s) for s in range(SEEDS)]
own = [matrix_variogram_score(iso, y, p=P_VS, n_samples=M, random_state=500 + s) for s in range(SEEDS)]
ens = [matrix_variogram_score(draw_ensemble(r, SIGMA), y, p=P_VS) for r in rngs]
se_own, se_ens = np.std(own, ddof=1) / np.sqrt(SEEDS), np.std(ens, ddof=1) / np.sqrt(SEEDS)
z = (np.mean(own) - np.mean(ens)) / np.hypot(se_own, se_ens)
print(f"{'variogram, isotropic (both MC)':<30s} sampler={np.mean(own):.5f}+-{se_own:.5f}  "
      f"ens={np.mean(ens):.5f}+-{se_ens:.5f}  ({z:+.1f} se)")
