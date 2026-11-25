"""Simulate asynchronous snapshot data for one example and seed.

Writes the observed blocks, an independent replicate data set (to measure the
sampling floor of the discrepancy) and draws from the true population."""
import argparse
import json

import yaml

import numpy as np

from cmc.metrics import energy_distance
from cmc.models import make_model, make_snapshot_data

ap = argparse.ArgumentParser()
ap.add_argument("--example", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
ex = cfg["examples"][a.example]
model = make_model(ex.get("kind", a.example), **ex["model"])
rng = np.random.default_rng(1000 * a.seed + 17)
if "data_tracks" in ex:  # real data (transfection): log fluorescence of each cell at the model's times
    rows = [r for r in open(ex["data_tracks"]).read().splitlines() if not r.startswith("#")]
    t = np.array([float(r.split(",")[0]) for r in rows[2:]])
    Y = np.array([[float(x) for x in r.split(",")[1:]] for r in rows[2:]])
    idx = [int(np.argmin(np.abs(t - tt))) for tt in model.meta["times"]]
    assert np.allclose(t[idx], model.meta["times"], atol=0.01)
    data = [np.log(Y[idx].T)]
    np.savez_compressed(a.out, truth=np.zeros((0, model.p)), floor=np.full(len(data), np.nan), block0=data[0])
    raise SystemExit
if "data_csv" in ex:  # real data (fish functional response): one block per initial prey number
    import csv

    rows = [r for r in csv.DictReader(open(ex["data_csv"])) if r["exclude"] == "No"]
    data = [np.array([[float(r["n_prey_eaten"])] for r in rows if int(r["n_prey_initial"]) == n0])
            for n0 in model.n0s]
    np.savez_compressed(a.out, truth=np.zeros((0, model.p)), floor=np.full(len(data), np.nan),
                        **{f"block{j}": d for j, d in enumerate(data)})
    raise SystemExit
data = make_snapshot_data(model, ex["n_cells"], rng)
replicate = make_snapshot_data(model, ex["n_cells"], rng)
truth = model.canonical(model.sample_population({**cfg["settings"], **ex.get("settings", {})}["n_truth"], rng))
floor = []
for d, r in zip(data, replicate):
    if model.discrete:
        d, r = np.log1p(d), np.log1p(r)
    floor.append(energy_distance(d, r, rng=rng))
np.savez_compressed(
    a.out,
    truth=truth,
    floor=np.array(floor),
    **{f"block{j}": d for j, d in enumerate(data)},
)
