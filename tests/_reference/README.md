# R oracle fixtures

`corrscore`'s `diebold_mariano` and `model_confidence_set` are checked against R. The R side is run
**once**, and its inputs and outputs are stored here, so the checks run from Python alone: no R,
`forecast` or `MCS` installation is needed to run `pytest`.

| File | What it holds | Used by |
|---|---|---|
| `dm_test_oracle_values.py` | Six cases (h = 1, 5, 10; `acf` and `bartlett`). R's own `rnorm()` draws as `loss_a` / `loss_b`, with the `statistic` and `p_value` that `forecast::dm.test` returned. | `tests/test_diebold_mariano.py`, asserted to 1e-6 |
| `mcs_oracle_case.py` | A fixed 150 x 4 loss matrix drawn in R (`set.seed(7)`) and the verdict of `MCS::MCSprocedure` (survivors `best`, `close2nd`). | `tests/test_mcs.py`, verdict only |
| `mcs_oracle_r_output.txt` | The full R output for that matrix: average losses and R's MCS p-values (`close2nd` 0.9002, `best` 1.0). Not asserted; for comparison. | reference |
| `gen_dm_oracle.R`, `gen_mcs_case.R` | The generating scripts. | `make check-oracles` |

The MCS check compares the verdict (which models survive), not p-values or elimination order. Both
differ from R by design (see the `src/corrscore/mcs.py` module docstring). On the fixed matrix,
`model_confidence_set` gives a final p-value of about 0.89 against R's 0.90.

## Regenerating

With R and the packages installed, `make check-oracles` regenerates both fixtures and fails if
either differs from what is stored. The stored files were produced and re-verified with
R 4.6.0 (2026-04-24), `forecast` 9.0.2 and `MCS` 0.2.0 on macOS arm64. R's random number
stream is stable across these versions for `set.seed` + `rnorm`, but if a newer R changes the
output, treat the stored files as authoritative and regenerate deliberately.
