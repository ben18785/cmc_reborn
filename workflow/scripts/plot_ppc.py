"""Posterior predictive figures from ppc.py output: for each example, one row per method and one
column per measured output. In each panel the grey bands are the observed data (5-95% and 25-75%
across cells, black line the median) at each time or condition, and the coloured lines are the
same quantiles of cells simulated from the method's fitted population (solid median, thin 5% and
95%). Held-out snapshots are shaded. A tracked example also gets a figure of individual
trajectories, observed against simulated."""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from cmc.models import make_model
from method_labels import COLOR, LABEL, complete

ap = argparse.ArgumentParser()
ap.add_argument("--ppc", required=True, help="directory of <example>_seed<seed>.npz from ppc.py")
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
complete(cfg["methods"])
os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
GREY_OUT, GREY_IN = "#d9d9d9", "#a6a6a6"


def output_names(model, d):
    name = type(model).__name__
    if "observed" in (getattr(model, "meta", None) or {}):
        return list(model.meta["observed"])
    if name == "GrowthFactor":
        return ["log R", "log P"]
    if name == "GeneExpression":
        return [f"gene {s + 1}" for s in range(d)]
    if name.startswith("Transfection"):
        return ["log GFP"]
    if name == "FunctionalResponse":
        return ["prey eaten"]
    return [f"output {s + 1}" for s in range(d)]


def ticks(ax, xlab):
    k = np.arange(len(xlab))
    step = max(1, len(xlab) // 8)
    ax.set_xticks(k[::step])
    ax.set_xticklabels(xlab[::step], rotation=0 if max(len(s) for s in xlab) < 5 else 45,
                       ha="center" if max(len(s) for s in xlab) < 5 else "right", fontsize=7)


for f in sorted(glob.glob(os.path.join(a.ppc, "*_seed*.npz"))):
    stem = os.path.basename(f)[:-4]
    example = stem.rsplit("_seed", 1)[0]
    if example not in cfg["examples"]:
        continue
    z = np.load(f, allow_pickle=False)
    ex = cfg["examples"][example]
    model = make_model(ex.get("kind", example), **ex["model"])
    methods, d = list(z["methods"]), int(z["n_out"])
    col_x, col_s, hold, xlab = z["col_x"], z["col_s"], z["col_hold"], z["xlab"]
    names = output_names(model, d)
    ylab = "log(1 + count)" if bool(z["discrete"]) else "value"
    if not methods:
        continue
    # several outputs: one row (sub-figure) per method, one column per output; one output: methods
    # wrap, 4 a row
    n = len(methods)
    if d > 1:
        fig = plt.figure(figsize=(2.7 * d + 0.4, 1.75 * n + 0.9), layout="constrained")
        rows = fig.subfigures(n, 1)
        rows = [rows] if n == 1 else list(rows)
        ax_of = {}
        for r, m in enumerate(methods):
            rows[r].suptitle(LABEL[m], fontsize=7, x=0.02, ha="left")
            axs = rows[r].subplots(1, d, sharey=True)
            for s in range(d):
                ax_of[r, s] = axs[s]
    else:
        nr, nc = int(np.ceil(n / 4)), min(n, 4)
        fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc + 0.4, 1.6 * nr + 0.9), sharex=True, sharey=True,
                                 squeeze=False, layout="constrained")
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        ax_of = {(r, 0): axes.flat[r] for r in range(n)}
    for s in range(d):
        cs = np.where(col_s == s)[0]
        cs = cs[np.argsort(col_x[cs])]
        x, qo = col_x[cs], z["q_obs"][cs]
        for r, m in enumerate(methods):
            ax = ax_of[r, s]
            for k in x[hold[cs]]:
                ax.axvspan(k - 0.5, k + 0.5, color="#fbe3c8", lw=0, zorder=0)
            ax.fill_between(x, qo[:, 0], qo[:, 4], color=GREY_OUT, lw=0, zorder=1)
            ax.fill_between(x, qo[:, 1], qo[:, 3], color=GREY_IN, lw=0, zorder=1)
            ax.plot(x, qo[:, 2], color="black", lw=1.2, zorder=2)
            qm = z[f"q_{m}"][cs]
            c = COLOR[m] if COLOR[m] not in ("#000000", "#4d4d4d", "#666666") else "#2a78d6"
            ax.plot(x, qm[:, 2], color=c, lw=2, zorder=3)
            ax.plot(x, qm[:, 0], color=c, lw=0.9, zorder=3)
            ax.plot(x, qm[:, 4], color=c, lw=0.9, zorder=3)
            if d == 1:
                ax.set_title(LABEL[m], fontsize=7, loc="left")
            else:
                ax.set_title(names[s], fontsize=7)
            ticks(ax, xlab)
            if d > 1 and r < n - 1:
                ax.tick_params(labelbottom=False)
    fig.supylabel(ylab + ("" if d > 1 else f" ({names[0]})"), fontsize=8)
    fig.supxlabel("time" if type(model).__name__ not in ("GrowthFactor", "LinearGaussian", "FunctionalResponse")
                  else {"GrowthFactor": "dose, time", "LinearGaussian": "snapshot",
                        "FunctionalResponse": "initial prey"}[type(model).__name__], fontsize=8)
    fig.suptitle(f"{example}: observed cells in grey (5–95%, 25–75%, median in black)\n"
                 f"cells simulated from each fit in colour (median, 5%, 95%)"
                 + ("; held-out snapshots shaded" if hold.any() else ""), fontsize=8)
    fig.savefig(os.path.join(a.out, f"ppc_{stem}.pdf"))
    fig.savefig(os.path.join(a.out, f"ppc_{stem}.png"), dpi=150)
    plt.close(fig)

    # tracked: individual trajectories, a few observed and a few simulated per method, first output
    blocks = [np.array(b.split(","), int) for b in z["blocks"]]
    if len(blocks) == 1 and len(blocks[0]) > 2 * d:
        cols = blocks[0][col_s[blocks[0]] == 0]
        cols = cols[np.argsort(col_x[cols])]
        pos = {c: i for i, c in enumerate(blocks[0])}
        yo = z["obs_block0"][:, [pos[c] for c in cols]][:30]
        nc = min(len(methods), 4)
        nr = int(np.ceil(len(methods) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc + 0.4, 1.7 * nr + 0.7), sharex=True, sharey=True,
                                 squeeze=False, layout="constrained")
        for ax in axes.flat[len(methods):]:
            ax.set_visible(False)
        for r, m in enumerate(methods):
            ax = axes.flat[r]
            ax.plot(col_x[cols], yo.T, color=GREY_IN, lw=0.6)
            ax.plot(col_x[cols], z[f"sim_{m}"][:30][:, cols].T,
                    color=COLOR[m] if COLOR[m] not in ("#000000", "#4d4d4d") else "#2a78d6", lw=0.6, alpha=0.8)
            ax.set_title(LABEL[m], fontsize=7, loc="left")
            ticks(ax, xlab)
        fig.supylabel(f"{names[0]} ({ylab})", fontsize=8)
        fig.supxlabel("time", fontsize=8)
        fig.suptitle(f"{example}: 30 observed (grey) and 30 simulated (coloured) trajectories", fontsize=8)
        fig.savefig(os.path.join(a.out, f"ppc_traj_{stem}.pdf"))
        fig.savefig(os.path.join(a.out, f"ppc_traj_{stem}.png"), dpi=150)
        plt.close(fig)
    print("plotted", stem)
