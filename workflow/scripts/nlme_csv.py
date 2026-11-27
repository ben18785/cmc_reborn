"""Long-format data (ID, TIME, DV, EVID) for the NLME software baseline (nlmixr2): exactly the data the
other methods fit (hourly log fluorescence of every cell, from the example's data file)."""
import argparse

import numpy as np
import yaml

from cmc.models import make_model

ap = argparse.ArgumentParser()
ap.add_argument("--example", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
ex = yaml.safe_load(open(a.config))["examples"][a.example]
model = make_model(ex.get("kind", a.example), **ex["model"])
Y = np.load(a.data)["block0"]
t = np.asarray(model.meta["times"], float)
with open(a.out, "w") as f:
    f.write("ID,TIME,DV,EVID\n")
    for i, row in enumerate(Y):
        for tt, y in zip(t, row):
            if np.isfinite(y):
                f.write(f"{i + 1},{tt:g},{y:.6f},0\n")
