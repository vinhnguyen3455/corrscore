suppressMessages(library(forecast))

cases <- list(
  list(seed=1, n=40, ma=1.2, mb=1.0, sa=0.5, sb=0.5, h=1, var="acf"),
  list(seed=2, n=60, ma=2.0, mb=1.8, sa=0.8, sb=0.6, h=1, var="acf"),
  list(seed=3, n=80, ma=1.5, mb=1.5, sa=0.4, sb=0.4, h=1, var="acf"),
  list(seed=4, n=100, ma=3.0, mb=2.5, sa=1.0, sb=1.0, h=5, var="acf"),
  list(seed=5, n=100, ma=3.0, mb=2.5, sa=1.0, sb=1.0, h=5, var="bartlett"),
  list(seed=6, n=50, ma=1.0, mb=1.3, sa=0.3, sb=0.3, h=10, var="acf")
)

pyarr <- function(x) paste0("[", paste(sprintf("%.12f", x), collapse=", "), "]")

cat("# Auto-generated ONCE via Rscript against forecast::dm.test -- see\n")
cat("# gen_dm_oracle2.R (kept alongside this file) for the exact generating\n")
cat("# script. This package does not depend on R or rpy2 at runtime or test\n")
cat("# time -- `loss_a`/`loss_b` below are R's own actual rnorm() draws,\n")
cat("# baked in as fixed data, not regenerated from a shared seed (R's and\n")
cat("# numpy's RNG streams are not interchangeable).\n")
cat("DM_ORACLE_CASES = [\n")
for (c in cases) {
  set.seed(c$seed)
  a <- rnorm(c$n, mean=c$ma, sd=c$sa) + 10
  b <- rnorm(c$n, mean=c$mb, sd=c$sb) + 10
  r <- dm.test(a, b, h=c$h, power=1, varestimator=c$var)
  cat(sprintf(
    "    dict(\n        loss_a=%s,\n        loss_b=%s,\n        h=%d, varestimator=%s,\n        statistic=%.10f, p_value=%.10f,\n    ),\n",
    pyarr(a), pyarr(b), c$h, paste0("'", c$var, "'"), unname(r$statistic), unname(r$p.value)
  ))
}
cat("]\n")
