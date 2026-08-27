# corrscore

Matrix-aware proper-scoring-rule backtesting for correlation and covariance forecasts.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this is

Forecasting a correlation or covariance matrix is common in risk management and portfolio
construction — but evaluating that forecast correctly is not routine. Two mistakes are easy to
make and hard to notice:

1. **Naive matrix-comparison metrics aren't proper scoring rules.** A metric that isn't a proper
   scoring rule can reward a forecaster for hedging toward a "safe" answer instead of reporting
   their honest best guess — the evaluation itself creates bad incentives.
2. **Walk-forward evaluation windows are easy to overlap with the estimation window**, silently
   leaking future information into a backtest and inflating apparent skill.

`corrscore` provides matrix-aware scoring rules (the energy score and the variogram score, both
adapted from their usual vector-valued form to score a full `K x K` matrix), a backtesting harness
that enforces a strict no-overlap timing discipline by construction, and the significance-testing
machinery (circular block bootstrap, Diebold-Mariano test, Model Confidence Set) needed to say
whether one forecast actually beats another, not just looks better on one sample.

Deliberately narrow in scope: this package does not fit a forecasting model, does not implement
any particular correlation-dynamics model, and does not fetch or clean data. It evaluates a
forecast; it never produces one.

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
pip install -e ".[dev]"
```

Requires Python 3.10-3.12. Runtime dependencies are `numpy`, `scipy`, and `arch` (for the
circular block bootstrap); `pytest`/`hypothesis`/`mypy` are dev-only.

## Status

v1: `matrix_energy_score`, `matrix_variogram_score`, `matrix_geodesic_variogram_score`,
`backtest_zero_overlap`, `circular_block_bootstrap`, `diebold_mariano`, `model_confidence_set`,
and `asymmetric_weighted_mean` all live in `src/corrscore/`, fully type-hinted (`py.typed` marker
included, `mypy src/corrscore` clean). Property tests throughout; `diebold_mariano` and
`model_confidence_set` are each cross-checked against a live-generated R oracle
(`forecast::dm.test` byte-exact, `MCS::MCSprocedure` verdict-matched — see `tests/_reference/`).

## Why this scope

`backtest_zero_overlap()` cannot and does not police how much history a caller's `forecast_fn`
consults internally (a full-history discounted filter and a short trailing window are both opaque
to it — the same responsibility boundary scikit-learn's `TimeSeriesSplit` leaves to its caller).
What it genuinely can and does enforce unconditionally is that `ground_truth_fn` is only ever
called with a start point strictly after the origin (`origin + 1 + purge_gap`, `purge_gap=0` by
default). See `src/corrscore/backtest.py`'s module docstring for the full reasoning.

## Dependency policy

Reuse only what's genuinely industry-standard; reproduce everything else in-house with
attribution and, where one exists, an oracle cross-check against a reference implementation.

| Package | Role | Decision |
|---|---|---|
| `numpy`, `scipy` | array math, `hyp1f1`/`gammaln` for a closed-form scoring-rule case, `pdist` | dependency |
| `arch` (Sheppard) | `CircularBlockBootstrap`, reused directly in `bootstrap.py` and (for its joint/multivariate resampling) `mcs.py` | dependency |
| — | `matrix_energy_score`/`matrix_variogram_score` | vendored (`scoring.py`) — no existing package treats a correlation matrix as the forecast object (see `docs/survey/`) |
| — | `diebold_mariano` | vendored (`diebold_mariano.py`), formula-matched against R's `forecast::dm.test` as a development-time oracle (see `tests/_reference/`) |
| — | `model_confidence_set` | vendored (`mcs.py`), verdict-checked (not byte-exact) against R's `MCS::MCSprocedure`, reproduced from Hansen, Lunde & Nason (2011) directly |

Full rationale and the live software-landscape survey behind these decisions: `docs/survey/`.

## Layout

```
corrscore/
  README.md
  LICENSE
  pyproject.toml            hatchling backend, mypy config; pip install -e ".[dev]" works
  .github/workflows/
    test.yml                 pytest + mypy matrix (Python 3.10-3.12)
  src/
    corrscore/
      __init__.py
      py.typed                PEP 561 marker
      scoring.py              matrix_energy_score, matrix_variogram_score, matrix_geodesic_variogram_score
      backtest.py             backtest_zero_overlap, BacktestResult
      bootstrap.py            circular_block_bootstrap, BootstrapResult (via arch)
      diebold_mariano.py      diebold_mariano, DieboldMarianoResult
      mcs.py                  model_confidence_set, MCSResult
      utils.py                asymmetric_weighted_mean
  tests/
    test_scoring.py           property tests + closed-form-vs-Monte-Carlo cross-checks
    test_backtest.py          zero-overlap-by-construction checks
    test_bootstrap.py
    test_diebold_mariano.py   property tests + R forecast::dm.test oracle cross-check
    test_mcs.py                property tests + R MCS::MCSprocedure verdict cross-check
    test_utils.py
    _reference/
      dm_test_oracle_values.py   fixed (data, R-computed statistic/p-value) tuples, generated once via Rscript
      mcs_oracle_check.py         helper that shells out to Rscript + MCS::MCSprocedure for the verdict cross-check
  docs/
    survey/                   the related-software landscape survey behind this package's scope
```

## Contributing

Issues and pull requests welcome. Run the test suite with `pytest -q` and type-check with
`mypy src/corrscore` before opening a PR.

## License

MIT — see [LICENSE](LICENSE).
