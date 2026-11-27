"""Fit one (example, method, seed) and score it.

Outputs a JSON (settings, per-sweep history, final metrics) and an npz with
`draws` (1000 draws of the fitted population; for the Bayesian methods, pooled
over the posterior) and `pop_draws` (S x n x p: one population per posterior
draw or bootstrap replicate; S = 1 for the point-estimate methods)."""
import argparse
import json

import yaml

import numpy as np

from cmc.bayes import BayesSettings, fit_hierarchical
from cmc.infer import Settings, ess, iterate_blocks, one_step_joint, weighted_moments
from cmc.metrics import block_discrepancies, energy_distance, resample
from cmc.models import make_model

ap = argparse.ArgumentParser()
ap.add_argument("--example", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out-json", required=True)
ap.add_argument("--out-npz", required=True)
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
ex = cfg["examples"][a.example]
meth = cfg["methods"][a.method]
# an example may override settings, and a method may override both (e.g. a tweak of the sampler)
st = {**cfg["settings"], **ex.get("settings", {}), **meth.get("settings", {})}
model = make_model(ex.get("kind", a.example), **ex["model"])
z = np.load(a.data)
data = [z[f"block{j}"] for j in range(len(model.blocks))]
truth, floor = z["truth"], z["floor"]
fit_blocks = [j for j in range(len(model.blocks)) if j not in ex["holdout"]]
hold_blocks = list(ex["holdout"])
rng = np.random.default_rng(10_000 * a.seed + 7)

if model.lower is not None:
    theta_scale = (model.upper - model.lower) / np.sqrt(12)
else:
    theta_scale = model.prior_sd


def score(theta, logw, r):
    out = {}
    out["ed_fit"] = block_discrepancies(model, theta, logw, data, fit_blocks, r)
    if hold_blocks:
        out["ed_holdout"] = block_discrepancies(model, theta, logw, data, hold_blocks, r)
    if len(truth):  # real data have no known population
        draws = resample(theta, logw, 2000, r)
        out["ed_theta"] = energy_distance(draws, truth, scale=theta_scale, rng=r)
    return out


def callback(theta, logw, r):
    s = score(theta, logw, r)
    return {"ed_fit_mean": float(np.mean(s["ed_fit"])),
            "ed_holdout_mean": float(np.mean(s["ed_holdout"])) if hold_blocks else None,
            "ed_theta": s.get("ed_theta")}


def _boot_one(args):
    """one Bayesian-bootstrap replicate (module level so worker processes can run it)"""
    ex_, s_, w_ = args
    from threadpoolctl import threadpool_limits
    m_ = make_model(ex_.get("kind", a.example), **ex_["model"])
    with threadpool_limits(1):
        return iterate_blocks(m_, data, fit_blocks, s_, data_weights=w_)


common = dict(n_reps=st["n_reps"], n_fit=st["n_fit"], ess_frac=st["ess_frac"], shrink=st["shrink"],
              clip=st["clip"], step=st["step"], rejuv=st.get("rejuv", "ess"), step_frac=st.get("step_frac", 1.0), threads=st["threads"], seed=a.seed)
if meth["kind"] == "one_step":
    s = Settings(ratio="vine", n_particles=st["one_step_particles"], n_sweeps=1,
                 noise=meth["noise"], **common)
    res = one_step_joint(model, data, fit_blocks, s)
    res["history"][0].update(callback(res["theta"], res["logw"], rng))
elif meth["kind"] == "iterate":
    s = Settings(ratio=meth["ratio"], n_particles=st["n_particles"], n_sweeps=ex["n_sweeps"],
                 noise=True, **common)
    res = iterate_blocks(model, data, fit_blocks, s, callback=callback,
                         init_theta=truth if meth.get("init") == "truth" else None)
elif meth["kind"] == "bootstrap":
    # Bayesian bootstrap (Rubin 1981) of the iteration: Dirichlet(1,...,1) weights on the
    # observed cells of every block, one full iterated fit per replicate
    # replicates are independent: run them in parallel worker processes (one BLAS thread each)
    common_b = {k: v for k, v in common.items() if k != "seed"}
    jobs = []
    for b in range(meth["n_boot"]):
        wts = {j: rng.exponential(size=len(data[j])) for j in fit_blocks}
        wts = {j: v / v.mean() for j, v in wts.items()}
        s = Settings(ratio=meth["ratio"], n_particles=st["n_particles"], n_sweeps=ex["n_sweeps"],
                     noise=True, seed=1000 * a.seed + b, **common_b)
        jobs.append((s, wts))
    workers = int(meth.get("workers", st.get("boot_workers", 1)))
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        # fork: workers inherit the data and model from this script (no re-import)
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
            reps = list(pool.map(_boot_one, [(ex, s, w) for s, w in jobs]))
    else:
        reps = [_boot_one((ex, s, w)) for s, w in jobs]
    pops, hist, t_tot = [], [], 0.0
    for b, r_b in enumerate(reps):
        pops.append(resample(r_b["theta"], r_b["logw"], st["n_pop"], rng).astype(np.float32))
        t_tot += r_b["history"][-1]["seconds"]  # summed CPU-side time over replicates
        n_sol = (hist[-1]["n_solves"] if hist else 0) + r_b["history"][-1]["n_solves"]
        hist.append(dict(replicate=b, mean=r_b["history"][-1]["mean"], sd=r_b["history"][-1]["sd"],
                         seconds=t_tot, n_solves=n_sol))
    pop_draws = np.array(pops)
    pooled = pop_draws.reshape(-1, model.p).astype(float)
    res = dict(theta=pooled, logw=np.full(len(pooled), -np.log(len(pooled))), history=hist,
               pop_draws=pop_draws)
elif meth["kind"] == "bayes":
    bs = BayesSettings(lik=meth["lik"], n_comp=meth["n_comp"], alpha=meth.get("alpha", 1.0),
                       seed=a.seed, **{**cfg["bayes"], **meth.get("bayes", {})})
    s = bs
    r_b = fit_hierarchical(model, data, fit_blocks, bs, callback=callback, n_checks=meth.get("n_checks", 10),
                           init_pop=truth if meth.get("init") == "truth" else None)
    pooled = r_b["pop_draws"].reshape(-1, model.p).astype(float)
    res = dict(theta=pooled, logw=np.full(len(pooled), -np.log(len(pooled))), pop_draws=r_b["pop_draws"],
               history=r_b["history"] + [dict(seconds=r_b["seconds"], n_solves=r_b["n_solves"],
                                              n_solves_train=r_b["n_solves_train"])])
else:
    raise ValueError(meth["kind"])

theta, logw = model.canonical(res["theta"]), res["logw"]
if "pop_draws" in res:
    res["pop_draws"] = np.array([model.canonical(P.astype(float)) for P in res["pop_draws"]], np.float32)
m, C = weighted_moments(theta, logw)
final = score(theta, logw, rng)
final.update(
    mean=m.tolist(),
    sd=np.sqrt(np.diag(C)).tolist(),
    cov=C.tolist(),
    truth_mean=truth.mean(0).tolist() if len(truth) else None,
    truth_sd=truth.std(0).tolist() if len(truth) else None,
    truth_cov=np.cov(truth.T).tolist() if len(truth) else None,
    ess=float(ess(logw)),
    floor_fit=float(np.mean(floor[fit_blocks])),
    floor_holdout=float(np.mean(floor[hold_blocks])) if hold_blocks else None,
    seconds=res["history"][-1]["seconds"],
    n_solves=res["history"][-1].get("n_solves"),
    n_solves_train=res["history"][-1].get("n_solves_train", 0),
)
out = dict(example=a.example, method=a.method, seed=a.seed, settings=s.to_dict(),
           param_names=model.param_names, n_out_fit=int(sum(len(model.blocks[j]) for j in fit_blocks)),
           n_blocks_fit=len(fit_blocks), history=res["history"], final=final)
with open(a.out_json, "w") as f:
    json.dump(out, f, indent=1)
pop_draws = res.get("pop_draws")
if pop_draws is None:
    pop_draws = resample(theta, logw, st["n_pop"], rng)[None].astype(np.float32)
np.savez_compressed(a.out_npz, draws=resample(theta, logw, 1000, rng), pop_draws=pop_draws)
