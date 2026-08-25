# Related-software survey — findings (complete 2026-08-25)

This package's scope rests on one survey, run live against PyPI/CRAN/GitHub
(license, last-activity, and — the actual point of the exercise — an
adversarial search trying to falsify the claimed gap) rather than assumed
from general familiarity, matching the standard `factor-shrinkage`'s own
survey used. Full writeup, with worked derivations for every scoring rule
and significance test this package implements: `../../../corrscore-package-
design.html` Sec. 5.2 and Table 3. This directory's own
[`related-software-survey.md`](related-software-survey.md) is a
code-repo-facing transcription of that table, kept in sync with it, for a
reader who lands in `docs/survey/` directly rather than the design doc.

## What this settles, concretely

1. **The gap is real and survives an adversarial search.** No package, in
   either Python or R, treats a correlation/covariance matrix as the
   forecast object with purge-aware walk-forward splitting and a proper
   scoring rule built in — every existing tool operates one layer down (a
   scalar loss, a vector observation, or a generic index-based CV
   splitter). Confirmed by targeted searches for the specific claim
   ("correlation matrix forecast evaluation package," GitHub topic search
   on `covariance-matrix`), not just an absence of a name that happened to
   come to mind.

2. **`arch` (Sheppard) is the one dependency that clears this package's own
   bar** (widely used, actively maintained, field-standard) — kept
   directly for `CircularBlockBootstrap` rather than reimplemented.
   Everything else found in the survey either isn't clearly established
   enough (`scoringrules`: 98★, and its own PyPI/GitHub release metadata
   disagreed on dates during this check — an unresolved integrity flag)
   or isn't installable at all as a real dependency (`model-confidence-set`:
   GitHub-only, no PyPI release, 21★; `mlfinlab`: gone commercial, same
   pattern already seen once with RSOME in `factor-shrinkage`'s own
   survey).

3. **R is comparatively well-served; Python is the real gap.** R's
   `scoringRules` (CRAN, 2024-09-18), `MCS` (Catania, CRAN, **2026-03-19** —
   genuinely current), `multDM`, and base `forecast::dm.test` together
   cover almost everything this package implements — just not assembled
   into one matrix-aware, zero-overlap pipeline. That absence of assembly,
   not absence of building blocks, is why an R companion package remains a
   plausible but non-urgent future extension (design doc Sec. 6.4), not a
   v1 requirement.

4. **Dependency policy, applied concretely per package**: `README.md`'s
   own "Dependency policy" table and `src/corrscore/`'s module docstrings
   (`scoring.py`, `diebold_mariano.py`, `mcs.py`) each state the specific
   reuse-vs.-vendor decision and why, rather than leaving it implicit.
   `model_confidence_set` in particular is checked against R's
   `MCS::MCSprocedure` verdict on a fixed synthetic case
   (`tests/_reference/mcs_oracle_case.py`) rather than claimed correct by
   construction — the design doc itself flagged this as "the clearest case
   in the survey" for vendoring, and it's also the implementation with the
   most room for a subtle bug, so the highest-value place to have an
   external check.

## What this survey did not cover

Whether `matrix_variogram_score`'s specific entry-indexing convention
(Sec. 2.5/2.4 of the design doc) matches any convention a reader coming
from the meteorology/forecast-verification literature would expect by
default — there is no existing matrix-valued precedent to check against,
which is the whole reason a convention had to be chosen rather than
adopted.
