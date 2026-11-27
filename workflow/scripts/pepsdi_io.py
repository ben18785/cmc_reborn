"""Input and output of the PEPSDI baseline (workflow/scripts/fit_pepsdi.jl); see that script for the method.

    prep     pick the cells PEPSDI fits (a seeded random subset of the example's cells with all 30
             time points finite), write them in PEPSDI's tidy format (time, obs, obs_id, id), and
             write the run settings for the Julia script;
    collect  turn PEPSDI's posterior draws of the population parameters into the npz that fit.py
             (method kind `external`) reads: pop_draws (S x n_pop x 8, log parameters in the order
             of TransfectionSSA.param_names), seconds, n_train_sims (-1), n_cells, n_times, and the
             run diagnostics.

PEPSDI's population (option 2): log(delta, gamma, k, m0, t0) ~ N(mu, D(tau) Omega D(tau)) across
cells; log(scale, offset) are cell constants kappa and log(sigma) the measurement-noise parameter
xi. One population per kept posterior draw: n_pop cells from the Gaussian, the constants at that
draw's (kappa_pop, xi_pop)."""
import argparse
import os
import shutil

import numpy as np
import yaml

from cmc.models import make_model

RANDOM = [0, 1, 2, 3, 5]  # delta, gamma, k, m0, t0: PEPSDI's individual parameters c_i (in this order)
CONST = [4, 6, 7]  # scale, offset (kappa), sigma (xi): PEPSDI's kappa_sigma (in this order)
SETTINGS = dict(n_cells=100, n_particles_pilot=200, n_pilot_min=300, n_pilot_max=5000, n_main_min=1000,
                n_main_max=50000, minutes=210, pilot_frac=0.25, burn_frac=0.2, n_keep_rows=2000,
                n_times_run_filter=50, init_log_sigma=-0.5, n_keep=50)


def prep(a, cfg, ex, meth, model):
    st = {k: meth.get(k, v) for k, v in SETTINGS.items()}
    Y = np.load(a.data)["block0"]
    t = np.asarray(model.meta["times"], float)
    ok = np.where(np.all(np.isfinite(Y), axis=1))[0]
    rng = np.random.default_rng(1000 * a.seed + 41)
    n = min(int(st["n_cells"]), len(ok))
    cells = np.sort(rng.choice(ok, n, replace=False))
    if os.path.isdir(a.work):  # PEPSDI reuses pilot runs it finds on disk: always start clean
        shutil.rmtree(a.work)
    os.makedirs(a.work)
    with open(os.path.join(a.work, "data.csv"), "w") as f:
        f.write("time,obs,obs_id,id\n")
        for i, c in enumerate(cells):
            for tt, y in zip(t, Y[c]):
                f.write(f"{tt:g},{y:.8g},1,{i + 1}\n")
    np.savetxt(os.path.join(a.work, "cells.txt"), cells, fmt="%d")
    with open(os.path.join(a.work, "settings.csv"), "w") as f:
        f.write("key,value\n")
        for k, v in {**st, "seed": a.seed, "n_cells": n, "n_cells_available": len(Y)}.items():
            f.write(f"{k},{v}\n")
    print(f"PEPSDI input: {n} of {len(Y)} cells ({len(Y) - len(ok)} with non-finite values excluded), "
          f"{len(t)} time points")


def ess(x):
    """Effective sample size of a chain (Geyer's initial positive sequence)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 4 or np.var(x) == 0:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (np.var(x) * n)
    s, k = 0.0, 1
    while k + 1 < n:
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        s += pair
        k += 2
    return float(n / (1 + 2 * s))


def collect(a, cfg, ex, meth, model):
    n_pop = int(cfg["settings"].get("n_pop", 1000))
    n_keep = int(meth.get("n_keep", SETTINGS["n_keep"]))
    rng = np.random.default_rng(1000 * a.seed + 43)
    head = open(os.path.join(a.work, "draws.csv")).readline().strip().split(",")
    D = np.loadtxt(os.path.join(a.work, "draws.csv"), delimiter=",", skiprows=1, ndmin=2)
    col = {h: j for j, h in enumerate(head)}
    info = {}
    for line in open(os.path.join(a.work, "info.csv")).read().splitlines()[1:]:
        k, v = line.split(",", 1)
        try:
            info[k] = float(v)
        except ValueError:
            info[k] = v
    d = len(RANDOM)
    mu = D[:, [col[f"mu{i + 1}"] for i in range(d)]]
    tau = D[:, [col[f"tau{i + 1}"] for i in range(d)]]
    corr = D[:, [col[f"corr{i + 1}_{j + 1}"] for j in range(d) for i in range(d)]].reshape(-1, d, d, order="F")
    ks = D[:, [col[f"ks{i + 1}"] for i in range(len(CONST))]]
    idx = np.linspace(0, len(D) - 1, min(n_keep, len(D))).astype(int)
    pops = []
    for r in idx:
        C = np.diag(tau[r]) @ corr[r] @ np.diag(tau[r])
        C = (C + C.T) / 2
        P = np.empty((n_pop, model.p))
        P[:, RANDOM] = rng.multivariate_normal(mu[r], C, size=n_pop, method="eigh")
        P[:, CONST] = ks[r]
        pops.append(P)
    names = [f"mu{i + 1}" for i in range(d)] + [f"tau{i + 1}" for i in range(d)] + [f"ks{i + 1}" for i in range(3)]
    ess_rows = np.array([ess(D[:, col[h]]) for h in names])
    cells = np.loadtxt(os.path.join(a.work, "cells.txt"), dtype=int, ndmin=1)
    extra = {k: v for k, v in info.items() if k not in ("seconds", "n_cells", "n_times")}
    np.savez_compressed(
        a.out, pop_draws=np.array(pops, np.float32), seconds=float(info["seconds"]), n_train_sims=-1,
        n_cells=int(info["n_cells"]), n_times=int(info["n_times"]), n_cells_used=int(info["n_cells"]),
        cells=cells, mu_mean=mu.mean(0), tau_mean=tau.mean(0), kappa_sigma_mean=ks.mean(0),
        ess_names=np.array(names), ess_kept_rows=ess_rows, n_rows_kept=len(D),
        **{k: np.array(v) for k, v in extra.items()})
    print(f"PEPSDI: {len(idx)} populations of {n_pop} cells; ESS over {len(D)} kept rows: "
          f"min {ess_rows.min():.0f} ({names[int(np.argmin(ess_rows))]})")


ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["prep", "collect"])
ap.add_argument("--example", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--data", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
ex = cfg["examples"][a.example]
meth = cfg["methods"][a.method]
model = make_model(ex.get("kind", a.example), **ex["model"])
assert model.name == "transfection_ssa", "the PEPSDI baseline is set up for the stochastic transfection model"
(prep if a.step == "prep" else collect)(a, cfg, ex, meth, model)
