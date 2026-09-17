suppressMessages(library(MCS))
set.seed(7)
n <- 150
# 4 models: "best" clearly lowest loss, "close2nd" close to best,
# "mediocre" clearly worse, "bad" clearly worst.
best <- 1.0 + rnorm(n, 0, 0.3)
close2nd <- 1.05 + rnorm(n, 0, 0.3)
mediocre <- 1.6 + rnorm(n, 0, 0.3)
bad <- 2.5 + rnorm(n, 0, 0.3)
Loss <- cbind(best=best, close2nd=close2nd, mediocre=mediocre, bad=bad)

pyarr <- function(x) paste0("[", paste(sprintf("%.10f", x), collapse=", "), "]")
cat("# fixed synthetic loss matrix, R-generated (set.seed(7)) -- see gen_mcs_case.R\n")
cat("MCS_LOSS_MATRIX = dict(\n")
for (nm in colnames(Loss)) {
  cat(sprintf("    %s=%s,\n", nm, pyarr(Loss[, nm])))
}
cat(")\n\n")

alpha <- 0.10
res <- MCSprocedure(Loss, alpha=alpha, B=5000, statistic="Tmax", seed=7, verbose=FALSE)
survivors <- rownames(res@show)[res@show[, "MCS p-Value"] >= alpha]
cat("# R MCSprocedure(alpha=0.10, statistic='Tmax') survivors:\n")
print(survivors)
show(res)
