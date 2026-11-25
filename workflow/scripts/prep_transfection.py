"""Single-cell mRNA transfection data (Fröhlich et al. 2018, npj Syst. Biol. Appl.) from the
paper's Zenodo archive (10.5281/zenodo.1228899, code.zip, CC BY-SA 4.0).

Reads code/project/data/{date}_mean_{eGFP,d2eGFP}.xlsx (first column: time in seconds; one
column per cell: GFP fluorescence) with the standard library (an .xlsx file is zipped XML) and
writes one CSV per construct: time in hours (up to 30 h, as in Arruda et al. 2024), experiment
date, then one column per cell.

    python workflow/scripts/prep_transfection.py --zip code.zip --out data/transfection
"""
import argparse
import hashlib
import io
import os
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

MD5 = "e720c42fa6c7358418eab63477460698"
DATES = ["20160427", "20160513", "20160603"]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def col_index(ref):
    letters = re.match(r"[A-Z]+", ref).group(0)
    k = 0
    for ch in letters:
        k = 26 * k + ord(ch) - 64
    return k - 1, int(ref[len(letters):]) - 1


def read_xlsx(raw):
    z = zipfile.ZipFile(io.BytesIO(raw))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")) for si in root.findall("m:si", NS)]
    root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    cells = {}
    for c in root.iter(f"{{{NS['m']}}}c"):
        v = c.find("m:v", NS)
        if v is None:
            continue
        j, i = col_index(c.get("r"))
        val = shared[int(v.text)] if c.get("t") == "s" else v.text
        try:
            cells[(i, j)] = float(val)
        except ValueError:
            cells[(i, j)] = np.nan
    n_i = max(i for i, _ in cells) + 1
    n_j = max(j for _, j in cells) + 1
    out = np.full((n_i, n_j), np.nan)
    for (i, j), v in cells.items():
        out[i, j] = v
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
raw = open(a.zip, "rb").read()
assert hashlib.md5(raw).hexdigest() == MD5, "unexpected archive"
arc = zipfile.ZipFile(io.BytesIO(raw))
os.makedirs(a.out, exist_ok=True)
for construct in ["eGFP", "d2eGFP"]:
    blocks = []
    for d in DATES:
        X = read_xlsx(arc.read(f"code/project/data/{d}_mean_{construct}.xlsx"))
        X = X[~np.isnan(X[:, 0])]
        t = X[:, 0] / 3600.0
        keep = t <= 30
        blocks.append((d, t[keep], X[keep, 1:]))
    t0 = blocks[0][1]
    for d, t, _ in blocks:
        assert len(t) == len(t0) and np.allclose(t, t0), f"time grids differ in {d}"
    Y = np.hstack([b[2] for b in blocks])
    date = np.concatenate([[int(d)] * b[2].shape[1] for d, b in zip(DATES, blocks)])
    with open(os.path.join(a.out, f"{construct}.csv"), "w") as f:
        f.write("# GFP fluorescence (a.u.); rows: time (h); columns: cells. Second row: experiment date.\n")
        f.write("time," + ",".join(f"cell{k}" for k in range(Y.shape[1])) + "\n")
        f.write("date," + ",".join(str(x) for x in date) + "\n")
        for ti, row in zip(t0, Y):
            f.write(f"{ti:.6g}," + ",".join("" if np.isnan(v) else f"{v:.6g}" for v in row) + "\n")
    print(construct, "cells:", Y.shape[1], "times:", len(t0), "NaN fraction:", round(float(np.isnan(Y).mean()), 3),
          "min:", np.nanmin(Y))
