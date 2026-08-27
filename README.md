# corrscore

Matrix-aware proper-scoring-rule backtesting for correlation and covariance forecasts, in Python.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why this exists

Forecasting a correlation or covariance matrix is common in risk management and portfolio
construction — but evaluating that forecast correctly is not routine. Two mistakes are easy to
make and hard to notice:

1. **Naive matrix-comparison metrics aren't proper scoring rules.** A metric that isn't a proper
   scoring rule can reward a forecaster for hedging toward a "safe" answer instead of reporting
   their honest best guess — the evaluation itself creates bad incentives.
2. **Walk-forward evaluation windows are easy to overlap with the estimation window**, silently
   leaking future information into a backtest and inflating apparent skill.

`corrscore` evaluates a forecast; it never produces one. It doesn't fit a model, doesn't implement
any particular correlation-dynamics model, and doesn't fetch or clean data — it's a focused
evaluation layer you drop on top of whatever you're already forecasting with.

## What it offers

- **Matrix-aware energy and variogram scores.** The two standard proper scoring rules from the
  forecast-verification literature, generalized from their usual vector-valued form to score a
  full `K x K` correlation/covariance matrix directly.
- **A geometric variant of the variogram score**, aware of the fact that correlation matrices live
  on a curved space rather than flat Euclidean space — sharper at detecting forecast danger as a
  matrix approaches the boundary of validity (near-singular, highly correlated regimes).
- **Four forecast representations**, not just point forecasts: a single deterministic matrix, a
  discrete mixture of any number of atoms, an isotropic-Gaussian mixture, or a general Monte Carlo
  ensemble — with closed-form scoring wherever one exists, Monte Carlo estimation only where it
  doesn't.
- **A backtesting harness (`backtest_zero_overlap`)** that makes the specific, easy-to-make
  lookahead bug — ground truth computed from a window that overlaps the forecast origin —
  structurally impossible to reproduce, rather than something you have to remember to get right.
- **Significance testing**, not just point comparisons: a circular block bootstrap for
  serially-dependent score differentials, the Diebold-Mariano test, and the Model Confidence Set
  — so "is model A really better than model B" has an actual answer.

```python
from corrscore import matrix_energy_score, backtest_zero_overlap
from corrscore import circular_block_bootstrap, diebold_mariano, model_confidence_set

result = backtest_zero_overlap(
    forecast_fns={"naive": naive_forecast, "filter": my_model.forecast},
    ground_truth_fn=realized_correlation,   # (start, end) -> K x K matrix
    origins=origins,
    horizon=10,
)
sig = circular_block_bootstrap(result.scores["filter"], result.scores["naive"], block_lengths=[1, 3, 6, 10])
mcs = model_confidence_set(result.scores, alpha=0.10)
```

## Install

```bash
pip install corrscore
```

Requires Python 3.10-3.12. Runtime dependencies are `numpy`, `scipy`, and `arch` (for the circular
block bootstrap) — nothing else.

## Validation

Every scoring rule and test in this package is checked against an independent source of truth, not
just its own self-consistency: `diebold_mariano` and `model_confidence_set` are cross-checked
against live-generated R oracles (`forecast::dm.test` byte-exact, `MCS::MCSprocedure`
verdict-matched), and every closed-form scoring formula is checked against brute-force Monte Carlo
simulation of the object it claims to score. Fully type-hinted and `mypy`-clean. See `tests/` for
the full suite.

## Related work

No existing package (Python or R) treats a correlation/covariance matrix as the forecast object
with purge-aware walk-forward splitting and a proper scoring rule built in — see
[`docs/survey/`](docs/survey/) for the full landscape survey and the specific reuse-vs.-vendor
decision behind each dependency.

## Citation

A companion paper describing the package's design and methodology is available as a working
paper on SSRN: [Nguyen (2026), "corrscore: Matrix-Aware Proper Scoring Rules and Significance
Testing for Correlation and Covariance Forecasts in
Python"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7358478). See
[`CITATION.cff`](CITATION.cff) for a machine-readable citation.

## Development

```bash
git clone https://github.com/vinhnguyen3455/corrscore
cd corrscore
pip install -e ".[dev]"
pytest -q
mypy src/corrscore
```

Issues and pull requests welcome.

## License

MIT — see [LICENSE](LICENSE).
