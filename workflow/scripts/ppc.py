"""Posterior predictive check: simulate each method's fitted population and store the predicted
outputs next to the observed ones, column by column (one column = one output at one time or
condition). Plotting is separate (plot_ppc.py), so this is the only step that simulates.

For every method with a fit of (example, seed): n cells are drawn from the fit's population draws
(`draws`, pooled over posterior populations where a method has several, so the prediction includes
that method's population uncertainty), simulated with noise from the true model, and summarised by
quantiles per column. Counts are stored on the log1p scale, as in the energy distances. A random
subset of the simulated and observed cells is also kept (`sim_<method>`, `obs_block<j>`) for
trajectory and histogram plots."""
import argparse
import os

import numpy as np
import yaml

from cmc.models import make_model

QS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

ap = argparse.ArgumentParser()
ap.add_argument("--example", required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--config", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--fits", required=True, help="directory with <method>_seed<seed>.npz fits of this example")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=2000, help="simulated cells per method")
ap.add_argument("--n-keep", type=int, default=300, help="simulated / observed cells kept raw")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
ex = cfg["examples"][a.example]
model = make_model(ex.get("kind", a.example), **ex["model"])
z = np.load(a.data)
blocks = [np.asarray(b) for b in model.blocks]
D = max(int(b.max()) for b in blocks) + 1
hold = set(ex.get("holdout", []))
tr = (lambda x: np.log1p(np.maximum(x, 0))) if model.discrete else (lambda x: x)

# outputs per time / condition: columns are laid out condition-major (column = d * k + s)
name = type(model).__name__
d = getattr(model, "n_obs", None) or {"GrowthFactor": 2, "GeneExpression": 3}.get(name) or \
    (len(blocks[0]) if name == "LinearGaussian" else 1)
col_x, col_s = np.arange(D) // d, np.arange(D) % d
times = model.meta.get("times") if isinstance(getattr(model, "meta", None), dict) else None
if times is None and hasattr(model, "times"):
    times = model.times
if name == "GrowthFactor":
    xlab = np.array([f"{L:g},{t:g}" for L, t in model.conds])
elif times is not None and len(np.atleast_1d(times)) == D // d:
    xlab = np.array([f"{t:g}" for t in np.atleast_1d(times)])
elif len(blocks) == D // d:
    xlab = np.array(model.block_labels)
else:
    xlab = np.array([str(k) for k in range(D // d)])

rng = np.random.default_rng(10_000 * a.seed + 913)
obs_cols = [[] for _ in range(D)]
col_hold = np.zeros(D, bool)
out = {}
for j, b in enumerate(blocks):
    y = tr(np.asarray(z[f"block{j}"], float))
    for i, c in enumerate(b):
        obs_cols[c].append(y[:, i])
        col_hold[c] |= j in hold
    out[f"obs_block{j}"] = y[rng.permutation(len(y))[:a.n_keep]].astype(np.float32)
q_obs = np.array([np.quantile(np.concatenate(v), QS) if v else np.full(len(QS), np.nan) for v in obs_cols])

methods = [m for m in ex.get("methods", cfg["methods"]) if "smoke" not in m
           and os.path.exists(os.path.join(a.fits, f"{m}_seed{a.seed}.npz"))]
for m in methods:
    f = np.load(os.path.join(a.fits, f"{m}_seed{a.seed}.npz"))
    th = np.asarray(f["draws"], float)
    th = th[np.all(np.isfinite(th), axis=1)]
    th = th[rng.integers(len(th), size=a.n)]
    if getattr(model, "lower", None) is not None:
        th = np.clip(th, model.lower, model.upper)
    sim = tr(np.asarray(model.simulate(th, rng, noise=True), float))
    out[f"q_{m}"] = np.nanquantile(sim, QS, axis=0).T.astype(np.float32)  # D x len(QS)
    out[f"sim_{m}"] = sim[:a.n_keep].astype(np.float32)
    print(f"{m}: {len(th)} cells simulated", flush=True)

np.savez_compressed(a.out, qs=QS, q_obs=q_obs, methods=np.array(methods), col_x=col_x, col_s=col_s,
                    col_hold=col_hold, xlab=xlab, n_out=d, discrete=bool(model.discrete),
                    blocks=np.array([",".join(map(str, b)) for b in blocks]), **out)
print("wrote", a.out, "with", len(methods), "methods")
