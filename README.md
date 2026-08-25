# corrscore

**Design doc**: [`../corrscore-package-design.html`](../corrscore-package-design.html) — the
pedagogical companion and package-design document this package implements: proper scoring
rules from scratch (the energy score, the closed-form spectrum across point/discrete-mixture/
isotropic-Gaussian-mixture/general-ensemble forecasts, the variogram score), the zero-overlap
walk-forward discipline (purged/embargoed cross-validation, unfolded), significance testing
(circular block bootstrap, Diebold-Mariano, Model Confidence Set), the live related-software
survey behind this package's scope, and the dependency policy (Sec. 6.4) this README's
"Dependency policy" section below implements concretely.

**Status**: 2026-08-25. v1 implemented: `matrix_energy_score`, `matrix_variogram_score`,
`backtest_zero_overlap`, `circular_block_bootstrap`, `diebold_mariano`,
`model_confidence_set`, and `asymmetric_weighted_mean` all live in `src/corrscore/`, fully
type-hinted (`py.typed` marker included, `mypy src/corrscore` clean), 52 passing tests
(property tests throughout; `diebold_mariano` and `model_confidence_set` each cross-checked
against a live-generated R oracle — `forecast::dm.test` byte-exact after catching a real
n-vs-(n-lag) autocovariance-normalization bug during development, `MCS::MCSprocedure` verdict-
matched — see `tests/_reference/`), installable via `conda run -n dev pip install -e ".[dev]"`
(this repository's established `dev` conda environment — see the parent repo's
`factor-shrinkage/README.md` for why that environment, not a fresh one, is the right target:
it already carries the matching numpy/scipy/scikit-learn toolchain `arch` needed as this
package's one real dependency). Dogfooded against real data immediately: see "Next steps"
item 1 below.

## What this is

A matrix-aware proper-scoring-rule backtesting harness for correlation and covariance-matrix
forecasts. The forecast object is always a `K x K` matrix — a point estimate, a discrete
mixture (any number of atoms), an isotropic-Gaussian mixture, or a general Monte Carlo
ensemble — scored against a realized, strictly-future ground-truth matrix, with the timing
discipline that prevents the specific bug this project made once already (design doc Sec. 3.3):
a `backtest_zero_overlap()` ground-truth call whose start point a caller can never move earlier
than `origin + 1`.

Deliberately narrow, matching `factor-shrinkage`'s own "interoperate, don't compete" scope:
this package does not fit a regime-switching model, does not implement DCC/RSDC/RM-DCC, and
does not fetch or clean data. It evaluates a forecast; it never produces one.

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

## Why this scope, and why the design doc's own API sketch changed slightly

Working through the exact mechanics of "zero overlap" during implementation surfaced a
cleaner, more honestly-scoped contract than the design doc's original `window` parameter
sketch (Sec. 6.2): `backtest_zero_overlap()` cannot and does not police how much history a
caller's `forecast_fn` consults internally (a full-history discounted filter and a short
trailing window are both opaque to it, same responsibility boundary scikit-learn's
`TimeSeriesSplit` leaves to its caller) — what it genuinely can and does enforce
unconditionally is that `ground_truth_fn` is only ever called with a start point strictly
after the origin (`origin + 1 + purge_gap`, `purge_gap=0` by default). See
`src/corrscore/backtest.py`'s module docstring for the full reasoning; this is a refinement
made once the mechanics were worked out precisely, not a scope change from the design doc's
intent.

## Dependency policy

Per the design doc's Sec. 6.4 rule (reuse only what's genuinely industry-standard; reproduce
everything else in-house with attribution and, where one exists, an oracle cross-check):

| Package | Role | Decision |
|---|---|---|
| `numpy`, `scipy` | array math, `hyp1f1`/`gammaln` for the Tier-2 closed form, `pdist` for Tier-3 | dependency |
| `arch` (Sheppard) | `CircularBlockBootstrap`, reused directly in `bootstrap.py` and (for its joint/multivariate resampling) `mcs.py` | dependency |
| — | `matrix_energy_score`/`matrix_variogram_score` | vendored (`scoring.py`) — not built on `scoringrules`, per the design doc's flag on that package's release-metadata inconsistency |
| — | `diebold_mariano` | vendored (`diebold_mariano.py`), formula-matched against R's `forecast::dm.test` as a development-time oracle (see `tests/_reference/`) |
| — | `model_confidence_set` | vendored (`mcs.py`), verdict-checked (not byte-exact) against R's `MCS::MCSprocedure` — the design doc's own flagged "flimsiest dependency in the survey," reproduced from Hansen, Lunde & Nason (2011) directly |

## Layout

```
corrscore/
  README.md                this file
  pyproject.toml            hatchling backend, mypy config; pip install -e ".[dev]" works
  .github/workflows/
    test.yml                  pytest + mypy matrix (scaffolded, not yet live — no git remote configured)
  src/
    corrscore/
      __init__.py
      py.typed                 PEP 561 marker
      scoring.py              matrix_energy_score, matrix_variogram_score (Tiers 1-3, design doc Sec. 2.4)
      backtest.py              backtest_zero_overlap, BacktestResult
      bootstrap.py             circular_block_bootstrap, BootstrapResult (via arch)
      diebold_mariano.py       diebold_mariano, DieboldMarianoResult
      mcs.py                    model_confidence_set, MCSResult
      utils.py                 asymmetric_weighted_mean (generalized from regime-detection-pfa-filter.py)
  tests/
    test_scoring.py            property tests + Tier1/Tier2/Tier3 cross-checks (incl. Monte Carlo validation of Eq. 5)
    test_backtest.py           zero-overlap-by-construction checks
    test_bootstrap.py
    test_diebold_mariano.py    property tests + R forecast::dm.test oracle cross-check
    test_mcs.py                 property tests + R MCS::MCSprocedure verdict cross-check
    test_utils.py
    _reference/
      dm_test_oracle_values.py   fixed (data, R-computed statistic/p-value) tuples, generated once via Rscript
      mcs_oracle_check.py         helper that shells out to Rscript + MCS::MCSprocedure for the verdict cross-check
  docs/
    survey/                   the design doc's Sec. 5.2 live software survey, as its own record
```

## Next steps (in order)

1. ~~**Refactor `scripts/regime-detection-pfa-filter.py`** against this package.~~ **Done**,
   2026-08-25 — as a separate file, `scripts/regime-detection-pfa-filter-corrscore.py`, per
   direct instruction (the original stays the source of record backing the manuscript's own
   cited numbers). Validated by direct diff against a captured baseline run of the original:
   Stages 3 and 4 (deterministic point-estimate scoring) are **bit-identical**; Stage 6's
   numbers (-17.6%/-19.5% at K=5, -15.4%/-18.0% at K=16) match the published manuscript
   exactly; Stage 5 (bootstrap significance) matches qualitatively as expected — same
   conclusion (p≈0.0000, significant at every block length) but not bit-identical digits,
   since `circular_block_bootstrap` resamples via `arch`'s own RNG/algorithm rather than the
   original script's hand-rolled one (the intended effect of the dependency-policy swap, not
   a discrepancy).
2. **Rerun the two comparisons flagged in the design doc's Sec. 2.5 box** under
   `matrix_variogram_score` — RM-DCC's mixture-vs-ensemble margin and the K=16 purity-gap
   chase — now that real data is wired through `regime-detection-pfa-filter-corrscore.py`.
3. Decide the final public surface (plain functions, as shipped, vs. an `sklearn`-style
   scorer-object convention) now that the refactor's friction points are known: the shared-
   ground-truth-across-models design (`backtest_zero_overlap(forecast_fns={...}, ...)`) held
   up well and removed a real duplication (Stage 4's original two independently-constructed
   origin lists collapsed into one).
