"""Section 5 worked example from the corrscore paper, verbatim.

Run with:  python examples/section5_example.py
Every random draw is seeded (data: default_rng(0), bootstrap: seed=1,
MCS: seed=2) and every forecast is a "point" forecast, so the output is
deterministic for a fixed set of library versions (see
requirements-paper.txt).
"""
import numpy as np
from corrscore import (backtest_zero_overlap, circular_block_bootstrap,
                        diebold_mariano, matrix_geodesic_variogram_score,
                        matrix_variogram_score, model_confidence_set)

rng = np.random.default_rng(0)
K, n_days = 4, 900
true_rho = 0.15 + 0.35 * (0.5 + 0.5 * np.sin(np.arange(n_days) / 60.0))
returns = np.empty((n_days, K))
for t in range(n_days):
    corr = np.full((K, K), true_rho[t]); np.fill_diagonal(corr, 1.0)
    returns[t] = rng.multivariate_normal(np.zeros(K), corr)

def sample_corr(start, end):
    return np.corrcoef(returns[start:end], rowvar=False)

def persistence_forecast(origin, window=60):
    return {"kind": "point", "Q": sample_corr(origin - window + 1, origin + 1)}

def shrinkage_forecast(origin, window=60, alpha=0.3):
    q = sample_corr(origin - window + 1, origin + 1)
    return {"kind": "point", "Q": (1 - alpha) * q + alpha * np.eye(K)}

origins, horizon = list(range(120, n_days - 10, 10)), 10
result = backtest_zero_overlap(
    forecast_fns={"persistence": persistence_forecast, "shrinkage": shrinkage_forecast},
    ground_truth_fn=sample_corr, origins=origins, horizon=horizon,
    score_fn=matrix_variogram_score)

print("mean VS, persistence:", result.scores["persistence"].mean())
print("mean VS, shrinkage:  ", result.scores["shrinkage"].mean())

boot = circular_block_bootstrap(result.scores["shrinkage"], result.scores["persistence"],
                                 block_lengths=[5, 10], seed=1)
def fmt_boot_p(r):
    # A percentile bootstrap with n_boot resamples cannot resolve a p-value
    # below 2/n_boot (two tails of one resample each); report an upper bound.
    floor = 2.0 / r.n_boot
    return f"p<{floor:g}" if r.p_value < floor else f"p={r.p_value:.4f}"

for block_len, r in boot.items():
    print(f"bootstrap block_len={block_len}: mean diff={r.obs:.4f}, "
          f"95% CI=({r.ci_lo:.4f}, {r.ci_hi:.4f}), {fmt_boot_p(r)}")

dm = diebold_mariano(result.scores["shrinkage"], result.scores["persistence"], h=horizon)
print(f"Diebold-Mariano: stat={dm.statistic:.3f}, p={dm.p_value:.1e}")

mcs = model_confidence_set(result.scores, alpha=0.10, block_len=10, seed=2)
print("MCS survivors:", mcs.survivors, " eliminated:", mcs.eliminated)
