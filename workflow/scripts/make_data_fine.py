"""Synthetic twin on the 10-min grid of Arruda et al. (2024) for their amortised baseline, whose
network reads 180 measurements per cell (1/6 h to 30 h). Same population and model as the twin the
other methods fit (hourly points), fresh cells, our simulator (e.g. exact Gillespie for
transfection_ssa). Writes obs (cells x 180, log fluorescence) and the grid."""
import argparse

import numpy as np
import yaml

from cmc.models import make_model

ap = argparse.ArgumentParser()
ap.add_argument("--example", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
ex = yaml.safe_load(open(a.config))["examples"][a.example]
grid = np.linspace(1 / 6, 30, 180)
model = make_model(ex.get("kind", a.example), **{**ex["model"], "times": tuple(grid)})
rng = np.random.default_rng(1000 * a.seed + 41)
theta = model.sample_population(ex["n_cells"], rng)
obs = model.simulate_cols(theta, np.arange(len(grid)), rng)
np.savez_compressed(a.out, obs=obs.astype(np.float32), grid=grid)
print(obs.shape, np.isfinite(obs).mean())
