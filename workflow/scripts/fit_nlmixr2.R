# NLME software baseline: the transfection ODE model (our `transfection`; the 'simple Froehlich' model of
# Arruda et al. 2024) fitted with nlmixr2 by SAEM or FOCEI, as a practitioner would.
# Log-normal population on (delta, gamma, km0, t0, offset) with diagonal covariance; additive error on the
# log scale with a shared sd (as in Arruda et al., the error sd does not vary across cells).
# The closed-form solution is written in the stable symmetric form used by cmc (lo = min(delta, gamma)).
# Initial values: prior means of the other methods, with delta and gamma moved apart (-2, -4) because the
# model is symmetric in them, except the offset, which a practitioner reads off the data: before expression
# starts y = log(offset) + noise, so loff starts at the median of y at the first time point (from 0, FOCEI
# stopped at its starting values with 'false convergence'). If FOCEI still reports false convergence it is
# refitted from its estimates with the derivative-free outer optimiser nlmixr2 suggests (bobyqa). Writes a JSON with the population mean and covariance of the 6 log
# parameters (log sigma has zero variance) and the sampling covariance of the means.
#   Rscript fit_nlmixr2.R data.csv saem|focei seed out.json
suppressPackageStartupMessages({library(nlmixr2); library(jsonlite)})
args <- commandArgs(TRUE)
d <- read.csv(args[1]); est <- args[2]; seed <- as.integer(args[3]); out <- args[4]
set.seed(seed)
# nlmixr2est 5.0.2 (inner.cpp, foceiSetupEta_) sizes a work buffer with a term n_records^2 computed in 32-bit
# integers, which overflows above 46340 observation records (d2eGFP, 2923 cells x 30 times, then asks for
# 31 GB; at 2565 cells the size wraps to a smaller, wrong buffer). Every nlmixr2 fit therefore uses a random
# subset of cells with at most 46000 records (1533 cells at 30 times); the number used is recorded.
n_all_cells <- length(unique(d$ID))
per_cell <- max(table(d$ID[d$EVID == 0]))
max_cells <- floor(46000 / per_cell)
if (n_all_cells > max_cells) {
  keep <- sort(sample(unique(d$ID), max_cells))
  d <- d[d$ID %in% keep, ]
}
cat("fitting", length(unique(d$ID)), "of", n_all_cells, "cells\n")
loff0 <- median(d$DV[d$EVID == 0 & d$TIME == min(d$TIME[d$EVID == 0])])
cat("initial loff", loff0, "\n")

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

# handleUninformativeEtas = FALSE: nlmixr2est 5.0.2's check for uninformative etas solves the model on
# (cells x 3 x n_eta) eta settings against the whole data set, so its memory grows with the square of the
# number of cells (about 150 GB for 2565 cells). All five etas here are identified; the check only resets
# the starting etas of cells whose data do not inform them.
ctl <- if (est == "saem") saemControl(seed = seed, print = 0, handleUninformativeEtas = FALSE) else foceiControl(print = 0)
t0 <- proc.time()[["elapsed"]]
ui0 <- nlmixr2(mod)
ui <- eval(bquote(ini(ui0, loff = .(loff0))))  # the value inlined: ini() evaluates its arguments lazily
fit <- nlmixr2(ui, d, est = est, control = ctl)
refit <- FALSE
if (est == "focei" && grepl("false convergence", paste(fit$message, collapse = " "))) {
  print(fit)
  cat("FOCEI false convergence: refitting with outerOpt = 'bobyqa' from these estimates\n")
  fit <- nlmixr2(fit, d, est = "focei", control = foceiControl(print = 0, outerOpt = "bobyqa"))
  refit <- TRUE
}
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
write_json(list(mean = mean, pop_cov = pop_cov, mean_cov = mean_cov, cov_ok = cov_ok, seconds = secs, refit_bobyqa = refit, loff_init = loff0,
                objf = fit$objf, n_cells = length(unique(d$ID)), n_cells_all = n_all_cells, n_times = 30, est = est,
                version = as.character(packageVersion("nlmixr2"))),
           out, digits = NA, auto_unbox = TRUE, matrix = "rowmajor")
