"""Nonparametric population PK baseline: NPAG (Neely et al. 2012; Yamada et al. 2021), or NPOD, as
implemented in PMcore, the Rust engine of Pmetrics (LAPKB, github.com/LAPKB/PMcore, GPL-3.0). NPAG is a
likelihood-based nonparametric maximum-likelihood (NPMLE) estimate of the population (mixing)
distribution: a discrete distribution on support points, found by alternating an interior-point
maximisation of the weights (Burke's IPM) with adaptive-grid expansion of the support and a step on the
error-model factor. Applied to the transfection ODE examples (tf_sim, tf_egfp, tf_d2egfp: one tracked
block, each cell observed hourly at 1..30 h).

The model is the closed form of cmc.models.Transfection (and of fit_nlmixr2.R), written in the Rust
binary workflow/scripts/pmcore_tf (src/main.rs): parameters log(delta, gamma, km0, t0, offset) vary across
cells; the observation is y = log(p + offset) + N(0, sigma^2) with one sigma shared by all cells.
This script writes the data in Pmetrics format, runs the binary, and turns PMcore's support points and
probabilities into population draws.

Route: PMcore called directly from Rust rather than through the Pmetrics R package. Pmetrics 3.x is an R
front end that writes the model as Rust source and compiles it against PMcore at run time, so it needs
the same Rust toolchain plus R, extendr and Pmetrics (not on CRAN or conda-forge); calling the engine
directly runs the same algorithm with one pinned toolchain (conda-forge rust) and exactly pinned crates.
PMcore 0.29.0 is used with pharmsol 0.29.4; the latest Pmetrics release (3.2.6) bundles PMcore 0.28.0,
whose NPAG, NPOD, IPM, adaptive grid, QR condensation and error-model step are the same source files
(0.28.0 -> 0.29.0 changed only prediction output and the pharmsol version). PMcore 0.29.0 needs the
pharmsol 0.29 model macros, which the bundled 0.28.0 predates.

Settings and their sources (deviations marked):
  * Parameters on the log scale (the model exponentiates), as for every other method; ranges are the
    prior boxes used by the other baselines, prior mean +- 4 prior sd (Transfection.prior_mean/prior_sd).
    NPAG needs finite ranges; it is otherwise prior-free.
  * Error model: Pmetrics' additive model, sd = sqrt(poly(obs)^2 + lambda^2), with the assay polynomial
    set to zero, so sd = lambda: additive Gaussian error on the log scale with one sd for all cells, the
    model of the other methods. lambda is estimated (PMcore's multiplicative up/down search after each
    cycle), starting from exp(prior mean of log sigma) = 0.37 (also nlmixr2's initial value).
  * Initial grid: a Sobol sequence of `points` points in the box, PMcore's default 2028 and default seed
    22 (the Pmetrics R front end defaults to 100 points; 2028 covers the wide 5-d box better).
  * Cycles: PMcore's default limit (1000 for NPAG, 100 for NPOD) and its convergence rule, plus
    (deviation) a wall-clock budget `minutes` (default 240) so that the job fits the 350-min CI limit:
    the fit stops after the last cycle that fits in the budget, and `converged`/`stopped_for_time`
    record which happened. All other NPAG settings are PMcore's defaults.
  * Cells: all cells by default; `max_cells` (deviation, off by default) fits a random subset if the
    cost per cycle turns out too high. `n_cells_used` records the number fitted.
  * Output: S = 1 (a point estimate). pop_draws are n_pop draws from the discrete NPAG distribution
    (support points with PMcore's probabilities) with log sigma appended as a constant column.
Writes an npz that fit.py (method kind `external`) scores like every other method; it also holds the
support points and weights, -2LL, cycles and timings."""
import argparse
import json
import os
import subprocess
import tempfile
import time

import numpy as np
import yaml

from cmc.models import make_model

ap = argparse.ArgumentParser()
ap.add_argument("--example", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--binary", required=True, help="the pmcore_tf executable")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
ex = cfg["examples"][a.example]
meth = cfg["methods"][a.method]
n_pop = int(cfg["settings"].get("n_pop", 1000))
rng = np.random.default_rng(1000 * a.seed + 11)
model = make_model(ex.get("kind", a.example), **ex["model"])
assert model.name == "transfection", "the PMcore model is written for the transfection ODE"
fit_blocks = [j for j in range(len(model.blocks)) if j not in ex["holdout"]]
assert fit_blocks == [0], "tracked transfection data have one block"
Y = np.load(a.data)["block0"].astype(float)
t = np.asarray(model.meta["times"], float)
n_cells = len(Y)

max_cells = meth.get("max_cells")
keep = np.arange(n_cells)
if max_cells is not None and n_cells > int(max_cells):
    keep = np.sort(rng.choice(n_cells, int(max_cells), replace=False))
print(f"fitting {len(keep)} of {n_cells} cells")

lo = model.prior_mean[:5] - 4 * model.prior_sd[:5]
hi = model.prior_mean[:5] + 4 * model.prior_sd[:5]
settings = dict(algorithm=meth.get("algorithm", "npag"), names=["ldelta", "lgamma", "lkm", "lt0", "loff"],
                lower=lo.tolist(), upper=hi.tolist(), points=int(meth.get("points", 2028)),
                sobol_seed=int(meth.get("sobol_seed", 22)),
                max_cycles=int(meth.get("max_cycles", 1000 if meth.get("algorithm", "npag") == "npag" else 100)),
                minutes=float(meth.get("minutes", 240)), sigma0=float(np.exp(model.prior_mean[5])))

with tempfile.TemporaryDirectory() as tmp:
    csv = os.path.join(tmp, "data.csv")
    with open(csv, "w") as f:  # Pmetrics format: observation rows only (no doses); OUTEQ 1 = outeq_1
        f.write("ID,EVID,TIME,DUR,DOSE,ADDL,II,INPUT,OUT,OUTEQ\n")
        for i in keep:
            for tt, y in zip(t, Y[i]):
                if np.isfinite(y):
                    f.write(f"{i + 1},0,{tt:g},.,.,.,.,.,{y:.6f},1\n")
    sfile = os.path.join(tmp, "settings.json")
    json.dump(settings, open(sfile, "w"))
    outdir = os.path.join(tmp, "out")
    env = dict(os.environ, RAYON_NUM_THREADS=str(a.threads))
    t0 = time.time()
    subprocess.run([os.path.abspath(a.binary), csv, sfile, outdir], check=True, env=env, cwd=tmp)
    seconds = time.time() - t0
    summ = json.load(open(os.path.join(outdir, "summary.json")))
    th = np.genfromtxt(os.path.join(outdir, "theta.csv"), delimiter=",", names=True)
    cycles_csv = open(os.path.join(outdir, "cycles.csv")).read()

th = np.atleast_1d(th)
support = np.column_stack([th[n] for n in settings["names"]]).astype(float)
weights = np.asarray(th["prob"], float)
weights = weights / weights.sum()
sigma = float(summ["sigma"])
idx = rng.choice(len(support), size=n_pop, p=weights)
pop = np.column_stack([support[idx], np.full(n_pop, np.log(sigma))])
print(f"{summ['algorithm']}: {summ['cycles']} cycles, converged {summ['converged']}, "
      f"stopped for time {summ['stopped_for_time']}, -2LL {summ['neg2ll']:.2f}, "
      f"{len(support)} support points, sigma {sigma:.4f}, {seconds / 60:.1f} min")
np.savez_compressed(a.out, pop_draws=pop[None].astype(np.float32), seconds=seconds, n_train_sims=-1,
                    n_cells=n_cells, n_times=len(t), n_cells_used=len(keep), cells_used=keep,
                    support=support, weights=weights, sigma=sigma, neg2ll=float(summ["neg2ll"]),
                    steps=int(summ["cycles"]), cycles=int(summ["cycles"]), converged=bool(summ["converged"]),
                    stopped_for_time=bool(summ["stopped_for_time"]), algorithm=summ["algorithm"],
                    points=settings["points"], max_cycles=settings["max_cycles"], minutes=settings["minutes"],
                    lower=lo, upper=hi, cycle_log=cycles_csv, pmcore_version=summ["pmcore_version"],
                    pharmsol_version=summ["pharmsol_version"])
print("done", round(seconds / 60, 1), "min")
