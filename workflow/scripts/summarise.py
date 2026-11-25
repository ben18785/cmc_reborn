"""Tables and figures from all fit JSONs."""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from cmc.gaussian import linear_rl_path, scalar_rl_path
from cmc.models import make_model

ap = argparse.ArgumentParser()
ap.add_argument("--fits", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
os.makedirs(a.out, exist_ok=True)

runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.fits, "*", "*.json")))]
by = defaultdict(list)
for r in runs:
    by[(r["example"], r["method"])].append(r)

EXAMPLES = [e for e in cfg["examples"] if any(k[0] == e for k in by)]
METHODS = [m for m in cfg["methods"] if any(k[1] == m for k in by)]
LABEL = {"cmc_paper": "CMC (noise ignored)", "cmc_augmented": "CMC (augmented)",
         "iter_kde": "Iterated, KDE", "iter_logistic": "Iterated, logistic", "iter_mlp": "Iterated, MLP"}
COLOR = {"cmc_paper": "#7f7f7f", "cmc_augmented": "#bcbd22", "iter_kde": "#1f77b4",
         "iter_logistic": "#2ca02c", "iter_mlp": "#d62728"}


def het_ratio(f):
    """Mean posterior sd / true sd over parameters that vary in the population."""
    sd, tsd = np.array(f["sd"]), np.array(f["truth_sd"])
    keep = tsd > 1e-9
    return float(np.mean(sd[keep] / tsd[keep]))


def cov_err(f):
    C, T = np.array(f["cov"]), np.array(f["truth_cov"])
    return float(np.linalg.norm(C - T) / np.linalg.norm(T))


def ms(xs):
    xs = np.asarray(xs, float)
    return f"{xs.mean():.3g} ± {xs.std():.2g}" if len(xs) > 1 else f"{xs.mean():.3g}"


rows = []
for e in EXAMPLES:
    for m in METHODS:
        rs = by.get((e, m), [])
        if not rs:
            continue
        F = [r["final"] for r in rs]
        rows.append(dict(
            example=e, method=m, n=len(rs), dims=rs[0]["n_out_fit"], blocks=rs[0]["n_blocks_fit"],
            ed_theta=[f["ed_theta"] for f in F],
            fit_rel=[np.mean(f["ed_fit"]) / f["floor_fit"] for f in F],
            hold_rel=[np.mean(f["ed_holdout"]) / f["floor_holdout"] for f in F] if F[0].get("floor_holdout") else None,
            het=[het_ratio(f) for f in F],
            cov=[cov_err(f) for f in F] if e in ("scalar", "linear") else None,
            ess=[f["ess"] for f in F],
            secs=[f["seconds"] for f in F],
        ))

hdr = "| example | fitted dims (blocks) | method | θ energy dist. | fit ED / floor | held-out ED / floor | sd ratio | cov. rel. error | minutes |\n|---|---|---|---|---|---|---|---|---|\n"
md = hdr
tex = []
for r in rows:
    hold = ms(r["hold_rel"]) if r["hold_rel"] else "–"
    md += (f"| {r['example']} | {r['dims']} ({r['blocks']}) | {LABEL[r['method']]} | {ms(r['ed_theta'])} | "
           f"{ms(r['fit_rel'])} | {hold} | {ms(r['het'])} | {ms(r['cov']) if r['cov'] else '–'} | {np.mean(r['secs'])/60:.1f} |\n")
    f2 = lambda xs: f"{np.mean(xs):.2f}"
    f3 = lambda xs: f"{np.mean(xs):.3f}"
    name = f"{r['example']} ({r['dims']})" if r["method"] == METHODS[0] else ""
    tex.append(f"{name} & {LABEL[r['method']]} & {f3(r['ed_theta'])} & {f2(r['fit_rel'])} & "
               f"{f2(r['hold_rel']) if r['hold_rel'] else '--'} & {f2(r['het'])} \\\\"
               + (" \\addlinespace" if r["method"] == METHODS[-1] and r is not rows[-1] else ""))
open(os.path.join(a.out, "table.md"), "w").write(
    md + "\nθ energy distance: posterior vs true population (parameters scaled by prior sd). "
    "ED / floor: mean energy distance between observed and predicted snapshots divided by the distance "
    "between two independent data sets (1 = as good as the data allow). sd ratio: posterior sd / true "
    "sd averaged over parameters that vary across cells. Mean ± sd over seeds.\n")
open(os.path.join(a.out, "table.tex"), "w").write(
    "\\begin{tabular}{llrrrr}\n\\toprule\n"
    "example (dims) & method & $\\theta$ ED & fit ED/floor & held-out ED/floor & sd ratio \\\\\n\\midrule\n"
    + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- figure 1: scalar toy, sd of the population over iterations
if ("scalar", "iter_kde") in by or ("scalar", "iter_logistic") in by:
    model = make_model("scalar", **cfg["examples"]["scalar"]["model"])
    s2, tm, tv = model.sigma**2, model.pop_mean, model.pop_sd**2 + model.sigma**2
    K = cfg["examples"]["scalar"]["n_sweeps"]
    path = scalar_rl_path(0.0, model.meta["prior_var"], s2, tm, tv, K)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(np.arange(1, K + 1), np.sqrt(path[:, 1]), "k-", lw=1.5, label="closed form")
    for m in ["iter_kde", "iter_logistic", "iter_mlp"]:
        for i, r in enumerate(by.get(("scalar", m), [])):
            sd = [h["sd"][0] for h in r["history"]]
            ax.plot(np.arange(1, len(sd) + 1), sd, "o", ms=2.5, color=COLOR[m], alpha=0.7,
                    label=LABEL[m] if i == 0 else None)
    ax.axhline(model.pop_sd, ls="--", color="k", lw=0.8)
    ax.axhline(np.sqrt(tv), ls=":", color="#7f7f7f", lw=1)
    ax.text(K, model.pop_sd, " truth", va="bottom", ha="right", fontsize=7)
    ax.text(K, np.sqrt(tv), " noise ignored", va="bottom", ha="right", fontsize=7, color="#7f7f7f")
    ax.set_xlabel("iteration")
    ax.set_ylabel("sd of fitted population")
    ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "scalar.pdf"))
    plt.close(fig)

# ---- figure 2: parameter recovery over sweeps for the multi-block examples
multi = [e for e in EXAMPLES if e != "scalar"]
if multi:
    fig, axes = plt.subplots(1, len(multi), figsize=(3.3 * len(multi), 2.8), squeeze=False)
    for ax, e in zip(axes[0], multi):
        for m in METHODS:
            for i, r in enumerate(by.get((e, m), [])):
                h = r["history"]
                if len(h) == 1:
                    ax.axhline(h[0]["ed_theta"], color=COLOR[m], ls="--", lw=1, label=LABEL[m] if i == 0 else None)
                else:
                    ax.plot([x["sweep"] for x in h], [x["ed_theta"] for x in h], color=COLOR[m], lw=1,
                            alpha=0.8, label=LABEL[m] if i == 0 else None)
        ax.set_yscale("log")
        ax.set_title(f"{e} ({by[(e, METHODS[0])][0]['n_out_fit']} output dims)", fontsize=9)
        ax.set_xlabel("sweep")
    axes[0][0].set_ylabel("energy distance to true population")
    axes[0][-1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "recovery.pdf"))
    plt.close(fig)

# ---- figure 3: marginal posteriors of heterogeneous parameters
for e, params in [("growth", [0, 1]), ("gene", [0, 3]), ("linear", [0, 1])]:
    if (e, "iter_mlp") not in by:
        continue
    z = np.load(sorted(glob.glob(os.path.join(a.data, f"{e}_seed*.npz")))[0])
    truth = z["truth"]
    names = by[(e, "iter_mlp")][0]["param_names"]
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 2.6))
    for ax, k in zip(axes, params):
        lo, hi = np.percentile(truth[:, k], [0.5, 99.5])
        w = hi - lo
        bins = np.linspace(lo - 1.5 * w, hi + 1.5 * w, 60)
        ax.hist(truth[:, k], bins=bins, density=True, color="k", alpha=0.25, label="true population")
        for m in METHODS:
            r = by.get((e, m), [])
            if not r:
                continue
            f = os.path.join(a.fits, e, f"{m}_seed{r[0]['seed']}.npz")
            d = np.load(f)["draws"][:, k]
            ax.hist(d, bins=bins, density=True, histtype="step", color=COLOR[m], lw=1.2, label=LABEL[m])
        ax.set_xlabel(names[k])
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"marginals_{e}.pdf"))
    plt.close(fig)
print(md)
