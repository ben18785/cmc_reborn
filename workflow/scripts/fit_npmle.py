"""Exact-likelihood nonparametric maximum likelihood (NPMLE) oracle for the population.

The fixed point that iterated CMC targets is the NPMLE of the mixing distribution: the population P
that maximises sum_i log int K(y_i | theta) dP(theta) (Kiefer & Wolfowitz 1956; Laird 1978; Lindsay
1983). When the per-cell likelihood K is available in closed form the NPMLE can be computed directly,
with no simulation. This script does so, as a reference for what CMC should reach on the examples with
an exact likelihood (the same `ExactLik` that the exact Bayesian sampler uses).

Method (cmc/npmle.py; the adaptive-grid scheme of nonparametric population PK, cf. NPAG, Leary et
al. 2001):
  1. candidates: `n_grid` draws from the prior (box or Gaussian);
  2. for a fixed candidate set the problem is convex in the weights; it is solved by EM (Laird 1978;
     Vardi et al. 1985) with SQUAREM acceleration, to a relative change in log-likelihood below `tol`;
  3. refinement: keep the support (weights above `keep` x max), add `n_new` candidates jittered around
     it (Gaussian, sd = h x prior scale, h halving each round), re-solve; `n_rounds` rounds.
Identical observations are merged (count data repeat), with multiplicities.

Uncertainty (`n_boot` > 0): a Bayesian bootstrap (Rubin 1981) on the final grid, re-solving the weights
with Dirichlet(1, ..., 1) weights on the cells. The grid is not re-adapted per replicate, so replicates
can only reweight the final candidates (an approximation, stated).

Writes an npz that fit.py (method kind `external`) scores like every other method."""
import argparse
import time

import numpy as np
import yaml

from cmc.models import make_model
from cmc.npmle import Observations, npmle, solve_weights

ap = argparse.ArgumentParser()
ap.add_argument("--example", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
ex, meth = cfg["examples"][a.example], cfg["methods"][a.method]
grid = {k: meth[k] for k in ("n_grid", "n_new", "n_rounds", "keep", "h0", "minutes") if k in meth}
solver = {k: meth[k] for k in ("gtol", "max_iter") if k in meth}
n_boot, n_pop = int(meth.get("n_boot", 0)), int(cfg["settings"].get("n_pop", 200))
rng = np.random.default_rng(1000 * a.seed + 61)

model = make_model(ex.get("kind", a.example), **ex["model"])
z = np.load(a.data)
fit_blocks = [j for j in range(len(model.blocks)) if j not in ex["holdout"]]
obs = Observations(model, {j: z[f"block{j}"] for j in fit_blocks}, fit_blocks)
n_cells = int(obs.c.sum())
print(f"{a.example}: {n_cells} cells, {len(obs.c)} distinct observations, {len(fit_blocks)} blocks, "
      f"p = {model.p}", flush=True)

t0 = time.time()
theta, w, logL, hist, n_lik = npmle(obs, rng, **grid, **solver, log=lambda s: print(s, flush=True))
t_fit = time.time() - t0
pops = [theta[rng.choice(len(theta), size=n_pop, p=w)]]
if n_boot:
    sup = w > 1e-6 * w.max()
    th_b, lL_b = theta[sup], logL[:, sup]
    row = np.repeat(np.arange(len(obs.c)), obs.c.astype(int))  # cell -> distinct observation
    pops, t_b = [], time.time()
    boot_end = t_b + 60 * float(meth.get("boot_minutes", 60))
    for b in range(n_boot):  # equal share of the remaining bootstrap budget per replicate
        dl = time.time() + (boot_end - time.time()) / (n_boot - b)
        cb = np.bincount(row, weights=rng.exponential(size=n_cells), minlength=len(obs.c))
        wb, _, _, _ = solve_weights(lL_b, cb, w[sup], deadline=dl, **solver)
        pops.append(th_b[rng.choice(len(th_b), size=n_pop, p=wb)])
seconds = time.time() - t0
np.savez_compressed(a.out, pop_draws=np.array(pops, np.float32), support=theta[w > 0], weights=w[w > 0],
                    loglik=hist[-1]["loglik"], seconds=seconds, t_fit=t_fit, n_train_sims=-1,
                    n_lik_evals=n_lik, n_cells=n_cells,
                    n_times=int(sum(len(model.blocks[j]) for j in fit_blocks)),
                    steps=np.array([x["loglik"] for x in hist]),
                    stopped_early=bool(hist[-1].get("stopped_early", False)),
                    max_D=np.array([x["max_D"] for x in hist]))
print("done", round(seconds / 60, 1), "min; final log-lik", round(hist[-1]["loglik"], 2))
