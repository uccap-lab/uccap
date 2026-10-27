# Independent reference values computed with R (MASS, fitdistrplus, nortest).
# Usage: Rscript validation/reference.R  ->  validation/reference.json
suppressMessages({library(fitdistrplus); library(nortest)})
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
q <- c(0.00135, 0.5, 0.99865)
dgumbelmax <- function(x, a, b) exp(-(x - a) / b - exp(-(x - a) / b)) / b
pgumbelmax <- function(q, a, b) exp(-exp(-(q - a) / b))
qgumbelmax <- function(p, a, b) a - b * log(-log(p))
dgumbelmin <- function(x, a, b) dgumbelmax(-x, -a, b)
pgumbelmin <- function(q, a, b) 1 - pgumbelmax(-q, -a, b)
qgumbelmin <- function(p, a, b) -qgumbelmax(1 - p, -a, b)
fit_one <- function(x, family) {
  f <- switch(family,
    normal = fitdist(x, "norm", method = "mle"),
    lognormal = fitdist(x, "lnorm"),
    weibull = fitdist(x, "weibull"),
    gamma = fitdist(x, "gamma", lower = c(1e-8, 1e-8)),
    exponential = fitdist(x, "exp"),
    logistic = fitdist(x, "logis"),
    lev = fitdist(x, "gumbelmax", start = list(a = mean(x) - 0.45 * sd(x), b = 0.78 * sd(x))),
    sev = fitdist(x, "gumbelmin", start = list(a = mean(x) + 0.45 * sd(x), b = 0.78 * sd(x))))
  qf <- switch(family, normal = "qnorm", lognormal = "qlnorm", weibull = "qweibull", gamma = "qgamma",
               exponential = "qexp", logistic = "qlogis", lev = "qgumbelmax", sev = "qgumbelmin")
  list(loglik = f$loglik, estimate = as.list(f$estimate),
       quantiles = do.call(qf, c(list(q), as.list(f$estimate))))
}
d2 <- function(n) integrate(function(x) 1 - (1 - pnorm(x))^n - pnorm(x)^n, -Inf, Inf, rel.tol = 1e-12)$value
c4 <- function(n) sqrt(2 / (n - 1)) * exp(lgamma(n / 2) - lgamma((n - 1) / 2))
out <- list(constants = list(n = 2:10, d2 = sapply(2:10, d2), c4 = sapply(2:25, c4), c4_n = 2:25), sets = list())
for (fam in c("normal", "lognormal", "weibull", "gamma", "exponential", "lev", "sev", "logistic")) {
  x <- scan(file.path(here, "data", paste0(fam, ".csv")), quiet = TRUE)
  fits <- list(); fits[[fam]] <- fit_one(x, fam)
  if (fam != "normal") fits[["normal"]] <- fit_one(x, "normal")
  out$sets[[fam]] <- list(
    n = length(x), mean = mean(x), sd = sd(x),
    sigma_amr = mean(abs(diff(x))) / d2(2),
    empirical_quantiles = unname(quantile(x, q, type = 7)),
    anderson_p = ad.test(x)$p.value, anderson_stat = unname(ad.test(x)$statistic),
    shapiro_p = shapiro.test(x)$p.value, lilliefors_p = lillie.test(x)$p.value,
    fits = fits)
}
jsonlite::write_json(out, file.path(here, "reference.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("validation/reference.json written\n")
