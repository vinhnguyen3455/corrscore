"""A near-tie among six candidate models: where the significance tools earn their place.

Same simulated K=4 panel as section5_example.py. The six candidates are
persistence forecasts that differ only in their trailing-window length
(20, 25, 30, 35, 40, 60 days), so their per-origin score differences change
sign and the verdict is not decided in advance.
Run with:  python examples/near_tie_example.py
"""
import numpy as np
from corrscore import (backtest_zero_overlap, circular_block_bootstrap,
                        diebold_mariano, matrix_variogram_score,
                        model_confidence_set)

rng = np.random.default_rng(0)
K, n_days = 4, 900
true_rho = 0.15 + 0.35 * (0.5 + 0.5 * np.sin(np.arange(n_days) / 60.0))
returns = np.empty((n_days, K))
for t in range(n_days):
    corr = np.full((K, K), true_rho[t]); np.fill_diagonal(corr, 1.0)
    returns[t] = rng.multivariate_normal(np.zeros(K), corr)

def sample_corr(start, end):
    return np.corrcoef(returns[start:end], rowvar=False)

def persistence(window):
    def forecast(origin):
        return {"kind": "point", "Q": sample_corr(origin - window + 1, origin + 1)}
    return forecast

windows = [20, 25, 30, 35, 40, 60]
origins, horizon = list(range(120, n_days - 10, 10)), 10
result = backtest_zero_overlap(
    forecast_fns={f"win{w}": persistence(w) for w in windows},
    ground_truth_fn=sample_corr, origins=origins, horizon=horizon,
    score_fn=matrix_variogram_score)
scores = result.scores
print(f"{len(origins)} origins, horizon {horizon}; mean variogram score:")
print("  " + "  ".join(f"{m}={v.mean():.3f}" for m, v in scores.items()))

BLOCKS, N_BOOT = [2, 5, 10], 2000
def fmt_p(p, prefix=False):
    # Resampling cannot resolve below ~1/n_boot, so report small values as a bound.
    body = "<0.001" if p < 0.001 else f"{p:.3f}"
    return ("p" + body if p < 0.001 else "p=" + body) if prefix else body

print("\nEach model against win30 (mean diff > 0 means worse than win30):")
print("  model   diff    DM p   bootstrap p at block 2 / 5 / 10")
for m in scores:
    if m == "win30":
        continue
    dm = diebold_mariano(scores[m], scores["win30"], h=1)
    boot = circular_block_bootstrap(scores[m], scores["win30"], block_lengths=BLOCKS,
                                    n_boot=N_BOOT, seed=1)
    ps = " / ".join(fmt_p(boot[b].p_value) for b in BLOCKS)
    print(f"  {m:<6s} {dm.mean_diff:+.3f}  {fmt_p(dm.p_value):>6s}   {ps}")

print("\nModel Confidence Set, alpha=0.10 (elimination path, with p at each step):")
for b in BLOCKS:
    mcs = model_confidence_set(scores, alpha=0.10, block_len=b, seed=2)
    path = " -> ".join(f"{m} ({fmt_p(p, prefix=True)})" for m, p in mcs.eliminated)
    print(f"  block {b:>2d}: survivors {mcs.survivors}")
    print(f"            eliminated: {path}")
