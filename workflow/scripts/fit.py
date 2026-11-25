"""Fit one (example, method, seed) and score it.

Outputs a JSON (settings, per-sweep history, final metrics) and an npz of
1000 posterior draws (resampled by weight)."""
import argparse
import json

import yaml

import numpy as np

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
st = cfg["settings"]
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
    draws = resample(theta, logw, 2000, r)
    out["ed_theta"] = energy_distance(draws, truth, scale=theta_scale, rng=r)
    return out


def callback(theta, logw, r):
    s = score(theta, logw, r)
    return {"ed_fit_mean": float(np.mean(s["ed_fit"])),
            "ed_holdout_mean": float(np.mean(s["ed_holdout"])) if hold_blocks else None,
            "ed_theta": s["ed_theta"]}


common = dict(n_reps=st["n_reps"], n_fit=st["n_fit"], ess_frac=st["ess_frac"], shrink=st["shrink"],
              clip=st["clip"], step=st["step"], threads=st["threads"], seed=a.seed)
if meth["kind"] == "one_step":
    s = Settings(ratio="vine", n_particles=st["one_step_particles"], n_sweeps=1,
                 noise=meth["noise"], **common)
    res = one_step_joint(model, data, fit_blocks, s)
    res["history"][0].update(callback(res["theta"], res["logw"], rng))
else:
    s = Settings(ratio=meth["ratio"], n_particles=st["n_particles"], n_sweeps=ex["n_sweeps"],
                 noise=True, **common)
    res = iterate_blocks(model, data, fit_blocks, s, callback=callback)

theta, logw = res["theta"], res["logw"]
m, C = weighted_moments(theta, logw)
final = score(theta, logw, rng)
final.update(
    mean=m.tolist(),
    sd=np.sqrt(np.diag(C)).tolist(),
    cov=C.tolist(),
    truth_mean=truth.mean(0).tolist(),
    truth_sd=truth.std(0).tolist(),
    truth_cov=np.cov(truth.T).tolist(),
    ess=float(ess(logw)),
    floor_fit=float(np.mean(floor[fit_blocks])),
    floor_holdout=float(np.mean(floor[hold_blocks])) if hold_blocks else None,
    seconds=res["history"][-1]["seconds"],
)
out = dict(example=a.example, method=a.method, seed=a.seed, settings=s.to_dict(),
           param_names=model.param_names, n_out_fit=int(sum(len(model.blocks[j]) for j in fit_blocks)),
           n_blocks_fit=len(fit_blocks), history=res["history"], final=final)
with open(a.out_json, "w") as f:
    json.dump(out, f, indent=1)
np.savez_compressed(a.out_npz, draws=resample(theta, logw, 1000, rng))
