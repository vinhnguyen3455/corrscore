# Related-software survey

Live-verified 2026-08-25 against PyPI/CRAN package pages, GitHub license/activity
checks, and source inspection where relevant.

| Package | Ecosystem | License | Last verified activity | Relevant to this harness |
|---|---|---|---|---|
| `scoringrules` | Python/PyPI | Apache-2.0 | v0.11.0; GitHub and PyPI release-date metadata disagree — unresolved | Energy + variogram score, vector-valued only, no matrix awareness |
| `properscoring` | Python/PyPI | Apache-2.0 | v0.1, 2015 — dead | Superseded by `scoringrules` |
| `arch` (Sheppard) | Python/PyPI | NCSA | v8.0.0, actively maintained | `arch.bootstrap.CircularBlockBootstrap` — the canonical Python implementation this package reuses directly |
| `tscv` / `timeseriescv` | Python/PyPI | BSD-3 / MIT | 2023 / 2018, stale | Gap-based CV splitters, not matrix-aware; `backtest_zero_overlap`'s own mechanics are thin enough not to depend on either |
| `mlfinlab` (Hudson & Thames) | Python/PyPI | closed-source, paid | confirmed live | López de Prado's own purged/embargoed K-fold code — no longer usable as a free dependency |
| `dieboldmariano` | Python/PyPI | MIT | v1.1.0, Jan 2025 | Standalone DM test, scalar loss series — vendored instead (see below) |
| `model-confidence-set` (JLDC) | Python/GitHub only | MIT | 21★, no PyPI release | Only maintained Python MCS port found — also the flimsiest dependency in the whole survey; vendored instead |
| `scoringRules` | R/CRAN | GPL-2/3 | v1.1.3, 2024-09-18 | Vector-valued energy score only, same matrix gap as the Python side |
| `MCS` (Catania) | R/CRAN | GPL-2 | v0.2.0, **2026-03-19** | Canonical Hansen et al. (2011) MCS — the best-maintained MCS implementation in either language; used as a development-time oracle |
| `multDM` | R/CRAN | GPL-3 | v1.1.5, 2025-03-08 | Multivariate DM test |
| `forecast::dm.test` | R (base ecosystem) | — | long-established | Canonical scalar DM test in R; used as this package's `diebold_mariano` development-time oracle |

**The gap check, adversarially searched, not falsified.** Targeted searches
("correlation matrix forecast evaluation package," "covariance forecast
backtesting python," "regime-switching correlation model validation
software," GitHub topic search on `covariance-matrix`) turned up no
package, in either language, that treats a correlation or covariance
matrix as the forecast object with purge-aware walk-forward splitting and
a proper scoring rule built in.

## Dependency decisions (mirrors `README.md`'s own table)

| Package | Role in `corrscore` | Decision |
|---|---|---|
| `numpy`, `scipy` | array math, `hyp1f1`/`gammaln`, `pdist` | dependency |
| `arch` | `CircularBlockBootstrap` | dependency |
| `scoringrules` | energy/variogram score math | vendored (`scoring.py`) |
| `dieboldmariano` | DM test | vendored (`diebold_mariano.py`), oracle-checked against `forecast::dm.test` |
| `model-confidence-set` | Model Confidence Set | vendored (`mcs.py`), verdict-checked against `MCS::MCSprocedure` |
