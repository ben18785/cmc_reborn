# NLME software baseline: the transfection ODE model (our `transfection`; the 'simple Froehlich' model of
# Arruda et al. 2024) fitted with nlmixr2 by SAEM or FOCEI, as a practitioner would.
# Log-normal population on (delta, gamma, km0, t0, offset) with diagonal covariance; additive error on the
# log scale with a shared sd (as in Arruda et al., the error sd does not vary across cells).
# The closed-form solution is written in the stable symmetric form used by cmc (lo = min(delta, gamma)).
# Initial values: prior means of the other methods, with delta and gamma moved apart (-2, -4) because the
# model is symmetric in them. Writes a JSON with the population mean and covariance of the 6 log
# parameters (log sigma has zero variance) and the sampling covariance of the means.
#   Rscript fit_nlmixr2.R data.csv saem|focei seed out.json
suppressPackageStartupMessages({library(nlmixr2); library(jsonlite)})
args <- commandArgs(TRUE)
d <- read.csv(args[1]); est <- args[2]; seed <- as.integer(args[3]); out <- args[4]
set.seed(seed)

mod <- function() {
  ini({
    ldelta <- -2; lgamma <- -4; lkm <- 5; lt0 <- 0; loff <- 0
    eta.delta ~ 1; eta.gamma ~ 1; eta.km ~ 1; eta.t0 ~ 1; eta.off ~ 1
    sig <- 0.37
  })
  model({
    dl <- exp(ldelta + eta.delta)
    gm <- exp(lgamma + eta.gamma)
    km <- exp(lkm + eta.km)
    t0 <- exp(lt0 + eta.t0)
    off <- exp(loff + eta.off)
    tau <- time - t0
    if (tau < 0) tau <- 0
    lo <- dl
    hi <- gm
    if (gm < dl) {
      lo <- gm
      hi <- dl
    }
    dd <- hi - lo
    rel <- tau
    if (dd * tau > 1e-8) rel <- (1 - exp(-dd * tau)) / dd
    y <- log(km * exp(-lo * tau) * rel + off)
    y ~ add(sig)
  })
}

ctl <- if (est == "saem") saemControl(seed = seed, print = 0) else foceiControl(print = 0)
t0 <- proc.time()[["elapsed"]]
fit <- nlmixr2(mod, d, est = est, control = ctl)
secs <- proc.time()[["elapsed"]] - t0
print(fit)

th <- fixef(fit)
nm <- c("ldelta", "lgamma", "lkm", "lt0", "loff")
sig <- unname(th[["sig"]])
mean <- c(unname(th[nm]), log(sig))
om <- fit$omega[c("eta.delta", "eta.gamma", "eta.km", "eta.t0", "eta.off"),
                c("eta.delta", "eta.gamma", "eta.km", "eta.t0", "eta.off")]
pop_cov <- matrix(0, 6, 6); pop_cov[1:5, 1:5] <- om
mean_cov <- matrix(0, 6, 6); cov_ok <- FALSE
if (!is.null(fit$cov)) {
  C <- fit$cov
  keep <- intersect(c(nm, "sig"), rownames(C))
  if (length(keep) == 6) {
    C <- C[c(nm, "sig"), c(nm, "sig")]
    J <- diag(c(rep(1, 5), 1 / sig))  # delta method for log sigma
    mean_cov <- J %*% C %*% J; cov_ok <- TRUE
  }
}
write_json(list(mean = mean, pop_cov = pop_cov, mean_cov = mean_cov, cov_ok = cov_ok, seconds = secs,
                objf = fit$objf, n_cells = length(unique(d$ID)), n_times = 30, est = est,
                version = as.character(packageVersion("nlmixr2"))),
           out, digits = NA, auto_unbox = TRUE, matrix = "rowmajor")
