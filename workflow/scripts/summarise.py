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

# examples with a `kind` belong to the tracking experiment (growth designs) or the Bayesian
# comparison; each has its own figure and table and stays out of the main table
TRACK = [e for e in cfg["examples"] if cfg["examples"][e].get("kind") == "growth"
         and "design" in cfg["examples"][e]["model"] and any(k[0] == e for k in by)]
EXAMPLES = [e for e in cfg["examples"] if "kind" not in cfg["examples"][e] and any(k[0] == e for k in by)]
METHODS = [m for m in cfg["methods"] if any(k[1] == m for k in by)]
LABEL = {"cmc_paper": "CMC (noise ignored)", "cmc_augmented": "CMC (augmented)",
         "iter_kde": "Iterated, KDE", "iter_logistic": "Iterated, logistic", "iter_mlp": "Iterated, MLP",
         "bb_logistic": "Iterated, logistic + Bayesian bootstrap", "bayes_gauss": "Bayes, Gaussian mixed effects",
         "bayes_dpm": "Bayes, DP mixture", "bayes_dpm_nre": "Bayes, DP mixture, likelihood-free"}
COLOR = {"cmc_paper": "#7f7f7f", "cmc_augmented": "#eda100", "iter_kde": "#1baf7a",
         "iter_logistic": "#2a78d6", "iter_mlp": "#eb6834",  # colour-blind-safe categorical order
         "bb_logistic": "#7fb3e8", "bayes_gauss": "#8e5bd3", "bayes_dpm": "#c2408f", "bayes_dpm_nre": "#4d4d4d",
         "bayes_gauss_nre": "#4d4d4d"}
LABEL.update({"bayes_gauss_nre": "Bayes, Gaussian mixed effects, likelihood-free",
              "iter_logistic_truth": "Iterated, logistic", "bayes_gauss_truth": "Bayes, exact likelihood",
              "bayes_gauss_nre_truth": "Bayes, likelihood-free (40k sims/block)",
              "bayes_gauss_nre_small_truth": "Bayes, likelihood-free (5k sims/block)",
              "bb_logistic_ssa": "Iterated, logistic + Bayesian bootstrap",
              "iter_bins": "Iterated, logistic on counts", "bb_bins": "Iterated, logistic on counts + Bayesian bootstrap"})
COLOR.update({"iter_bins": "#1baf7a", "bb_bins": "#8fd8b8", "bb_logistic_ssa": COLOR["bb_logistic"], "iter_logistic_truth": COLOR["iter_logistic"], "bayes_gauss_truth": COLOR["bayes_gauss"],
              "bayes_gauss_nre_truth": "#4d4d4d", "bayes_gauss_nre_small_truth": "#b0b0b0"})


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
    ex_rows = [x for x in rows if x["example"] == r["example"]]
    name = f"{r['example']} ({r['dims']})" if r is ex_rows[0] else ""
    tex.append(f"{name} & {LABEL[r['method']]} & {f3(r['ed_theta'])} & {f2(r['fit_rel'])} & "
               f"{f2(r['hold_rel']) if r['hold_rel'] else '--'} & {f2(r['het'])} \\\\"
               + (" \\addlinespace" if r is ex_rows[-1] and r is not rows[-1] else ""))
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

# ---- figure 5 + table: growth example, the two parameter combinations the snapshots identify
# names: log_RT, log_k1, log_km1, log_kdeg, log_kdegs
COMBOS = [("$\\log(k_1/k_{-1})$ (binding affinity)", lambda x: x[:, 1] - x[:, 2]),
          ("$\\log(R_T k_\\mathrm{deg})$ (receptor synthesis)", lambda x: x[:, 0] + x[:, 3])]
CM = [m for m in ["cmc_paper", "iter_logistic", "iter_mlp"] if ("growth", m) in by]
if CM:
    seeds = sorted(r["seed"] for r in by[("growth", CM[0])])
    truth = {sd: np.load(os.path.join(a.data, f"growth_seed{sd}.npz"))["truth"] for sd in seeds}
    draws = {(m, sd): np.load(os.path.join(a.fits, "growth", f"{m}_seed{sd}.npz"))["draws"]
             for m in CM for sd in seeds}
    rows_c = []
    for lab, f in COMBOS:
        t = [f(truth[sd]) for sd in seeds]
        cells = [f"{np.mean([x.mean() for x in t]):.2f} $\\pm$ {np.mean([x.std() for x in t]):.2f}"]
        for m in CM:
            v = [f(draws[(m, sd)]) for sd in seeds]
            cells.append(f"{np.mean([x.mean() for x in v]):.2f} $\\pm$ {np.mean([x.std() for x in v]):.2f}")
        rows_c.append(lab.split(" (")[0] + " & " + " & ".join(cells) + " \\\\")
    open(os.path.join(a.out, "combos_growth.tex"), "w").write(
        "\\begin{tabular}{l" + "c" * (len(CM) + 1) + "}\n\\toprule\n"
        "& true population & " + " & ".join(LABEL[m] for m in CM) + " \\\\\n\\midrule\n"
        + "\n".join(rows_c) + "\n\\bottomrule\n\\end{tabular}\n")
    sd0 = seeds[0]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, (lab, f) in zip(axes, COMBOS):
        t = f(truth[sd0])
        lo, hi = np.percentile(t, [0.5, 99.5])
        w = hi - lo
        bins = np.linspace(lo - 0.6 * w, hi + 0.6 * w, 50)
        ax.hist(t, bins=bins, density=True, color="k", alpha=0.25, label="true population")
        for m in CM:
            ax.hist(f(draws[(m, sd0)]), bins=bins, density=True, histtype="step", color=COLOR[m], lw=1.4,
                    label=LABEL[m])
        ax.set_xlabel(lab, fontsize=8)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "combos_growth.pdf"))
    fig.savefig(os.path.join(a.out, "combos_growth.png"), dpi=150)
    plt.close(fig)

# ---- figure 6: examples with an exact answer, analytic densities vs inferred populations (one data set)
from scipy.stats import norm
from cmc.gaussian import linear_rl_path as _lpath
if all(("scalar", m) in by for m in ["cmc_paper", "cmc_augmented", "iter_logistic", "iter_mlp"]) and \
        all(("linear", m) in by for m in ["iter_logistic", "iter_mlp"]):
    sm = make_model("scalar", **cfg["examples"]["scalar"]["model"])
    s2, mu, tv, pv = sm.sigma**2, sm.pop_mean, sm.pop_sd**2 + sm.sigma**2, sm.meta["prior_var"]
    Ks = cfg["examples"]["scalar"]["n_sweeps"]
    sp = scalar_rl_path(0.0, pv, s2, mu, tv, Ks)
    exact_s = {"cmc_paper": (mu, tv), "cmc_augmented": tuple(sp[0]),
               "iter_logistic": tuple(sp[-1]), "iter_mlp": tuple(sp[-1])}
    lex = cfg["examples"]["linear"]
    lm = make_model("linear", **lex["model"])
    fitb = [j for j in range(len(lm.blocks)) if j not in lex["holdout"]]
    rng_o = np.random.default_rng(0)
    lmean, lV = _lpath(lm, lambda _s: rng_o.permutation(fitb), lex["n_sweeps"])[-1]
    seed0 = sorted(r["seed"] for r in by[("scalar", "iter_logistic")])[0]
    dr = lambda e, m: np.load(os.path.join(a.fits, e, f"{m}_seed{seed0}.npz"))["draws"]
    p_lin = len(lmean)
    ncol = max(4, p_lin)
    fig, axes = plt.subplots(2, ncol, figsize=(1.6 * ncol, 4.0), squeeze=False)
    # row 1: scalar, one panel per method, each against its own exact answer
    for ax, m in zip(axes[0], ["cmc_paper", "cmc_augmented", "iter_logistic", "iter_mlp"]):
        em, ev = exact_s[m]
        x = np.linspace(mu - 5, mu + 5, 400)
        ax.hist(dr("scalar", m)[:, 0], bins=np.linspace(x[0], x[-1], 45), density=True,
                color=COLOR[m], alpha=0.55, label="inferred")
        ax.plot(x, norm.pdf(x, em, np.sqrt(ev)), "k-", lw=1.3, label="exact")
        ax.plot(x, norm.pdf(x, mu, sm.pop_sd), color="k", ls=":", lw=1, label="true population")
        ax.set_title(LABEL[m], fontsize=7, pad=2)
        ax.set_xlabel("θ", fontsize=7)
    for ax in axes[0][4:]:
        ax.axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    axes[0][4 if ncol > 4 else 3].legend(h, ["inferred (method colour)", "exact answer for that method",
                                             "true population"], fontsize=6, frameon=False, loc="center left")
    # row 2: linear, exact block-wise limit vs both classifier fits, per parameter
    for k in range(ncol):
        ax = axes[1][k]
        if k >= p_lin:
            ax.axis("off")
            continue
        sd_k = np.sqrt(lV[k, k])
        x = np.linspace(lmean[k] - 4 * sd_k, lmean[k] + 4 * sd_k, 400)
        for m in ["iter_logistic", "iter_mlp"]:
            ax.hist(dr("linear", m)[:, k], bins=np.linspace(x[0], x[-1], 35), density=True, histtype="step",
                    color=COLOR[m], lw=1.4, label=LABEL[m])
        ax.plot(x, norm.pdf(x, lmean[k], sd_k), "k-", lw=1.3, label="exact")
        ax.set_xlabel(f"θ{k + 1}", fontsize=7)
    axes[1][0].legend(fontsize=6, frameon=False, loc="upper left")
    for ax in axes.ravel():
        ax.set_yticks([])
        ax.tick_params(labelsize=6, length=2)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("scalar", fontsize=8, fontweight="bold")
    axes[1][0].set_ylabel("linear", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "exact.pdf"))
    fig.savefig(os.path.join(a.out, "exact.png"), dpi=150)
    plt.close(fig)

# ---- figure 7 + table: tracking experiment (growth model, snapshot vs pairs vs tracked cells)
if TRACK:
    TM = [m for m in ["iter_logistic", "iter_mlp"] if any((e, m) in by for e in TRACK)]
    DLAB = {"snapshot": "snapshots", "pairs": "pairs of times", "tracked": "tracked cells"}
    design = {e: cfg["examples"][e]["model"].get("design", "snapshot") for e in TRACK}
    names = by[(TRACK[0], TM[0])][0]["param_names"]

    def corr(f, i, j):
        C = np.array(f["cov"])
        return C[i, j] / np.sqrt(C[i, i] * C[j, j])

    lines = []
    for e in TRACK:
        for m in TM:
            F = [r["final"] for r in by.get((e, m), [])]
            if not F:
                continue
            r0 = by[(e, m)][0]
            sd = np.mean([f["sd"] for f in F], 0)
            tsd = np.array(F[0]["truth_sd"])
            cells = r0["n_blocks_fit"] * cfg["examples"][e]["n_cells"]
            lines.append(
                f"{DLAB[design[e]] if m == TM[0] else ''} & {r0['n_blocks_fit']} $\\times$ {r0['n_out_fit'] // r0['n_blocks_fit']} & "
                f"{cells} & {LABEL[m]} & {np.mean([f['ed_theta'] for f in F]):.3f} & "
                f"{sd[0] / tsd[0]:.2f} & {sd[1] / tsd[1]:.2f} & {np.mean(sd[2:4]):.2f} & "
                f"{np.mean([corr(f, 0, 3) for f in F]):.2f} & {np.mean([corr(f, 1, 2) for f in F]):.2f} \\\\")
    open(os.path.join(a.out, "tracking.tex"), "w").write(
        "\\begin{tabular}{lrrlrrrrrr}\n\\toprule\n"
        "design & blocks $\\times$ dims & cells & method & $\\theta$ ED & sd ratio $R_T$ & sd ratio $k_1$ & "
        "sd of fixed rates & corr$(R_T, k_\\mathrm{deg})$ & corr$(k_1, k_{-1})$ \\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    fig, axes = plt.subplots(1, len(names), figsize=(1.7 * len(names) + 0.4, 2.6), squeeze=False)
    MK = {"iter_logistic": "o", "iter_mlp": "D"}
    for k, ax in enumerate(axes[0]):
        F0 = by[(TRACK[0], TM[0])][0]["final"]
        tm, ts = F0["truth_mean"][k], F0["truth_sd"][k]
        ax.axhline(tm, color="k", lw=0.6, ls=":", zorder=0)
        ax.errorbar(0, tm, yerr=ts if ts > 1e-9 else None, fmt="_", ms=10, color="k", lw=1.6)
        for i, e in enumerate(TRACK, start=1):
            for j, m in enumerate(TM):
                F = [r["final"] for r in by.get((e, m), [])]
                if not F:
                    continue
                x = i + (j - (len(TM) - 1) / 2) * 0.25
                ax.errorbar(x, np.mean([f["mean"][k] for f in F]), yerr=np.mean([f["sd"][k] for f in F]),
                            fmt=MK[m], ms=4, color=COLOR[m], lw=1.4, mec="white", mew=0.5)
        ax.set_xticks(range(len(TRACK) + 1))
        ax.set_xticklabels(["truth"] + [DLAB[design[e]].replace(" ", "\n", 1) for e in TRACK], fontsize=6)
        ax.set_title(names[k].replace("log_", "log "), fontsize=7, pad=2)
        ax.tick_params(axis="y", labelsize=6, length=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color="k", marker="_", ms=10, lw=1.6, label="true population")] + [
        Line2D([], [], color=COLOR[m], marker=MK[m], ms=4, lw=1.4, label=LABEL[m]) for m in TM]
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(a.out, "tracking.pdf"))
    fig.savefig(os.path.join(a.out, "tracking.png"), dpi=150)
    plt.close(fig)

# ---- figure 8 + table: fully Bayesian hierarchical models vs iterated CMC
# Each method stores S populations (posterior draws / bootstrap replicates; S = 1 for CMC). For every
# quantity we take the population mean and the population sd (cell-to-cell spread) in each of them.
BM = [m for m in ["iter_logistic", "bb_logistic", "bayes_gauss", "bayes_dpm", "bayes_dpm_nre"] if m in cfg["methods"]]
BEX = [e for e in cfg["examples"] if cfg["examples"][e].get("experiment") not in ("robustness", "ssa", "fish")
       and any((e, m) in by for m in BM if cfg["methods"][m]["kind"] in ("bootstrap", "bayes"))]
SSA_M = [m for m in ["iter_logistic", "iter_mlp", "bb_logistic_ssa", "bayes_gauss", "bayes_gauss_nre"] if m in cfg["methods"]]
SSA_EX = [e for e in cfg["examples"] if cfg["examples"][e].get("experiment") == "ssa" and any(k[0] == e for k in by)]
PRETTY = {"log_RT": "log $R_T$", "log_k1": "log $k_1$", "log_km1": "log $k_{-1}$",
          "log_kdeg": "log $k_\\mathrm{deg}$", "log_kdegs": "log $k_\\mathrm{degs}$"}
def bayes_summary(BEX, BM, name):
    def quantities(e):
        names = next(by[k][0]["param_names"] for k in by if k[0] == e)
        q = [(PRETTY.get(n, n.replace("theta", "θ")), (lambda k: lambda x: x[..., k])(k)) for k, n in enumerate(names)]
        if cfg["examples"][e].get("kind") == "growth":
            q += [("log$(k_1/k_{-1})$", lambda x: x[..., 1] - x[..., 2]),
                  ("log$(R_T k_\\mathrm{deg})$", lambda x: x[..., 0] + x[..., 3])]
        return q

    def stats(e, m, sd_):
        P = np.load(os.path.join(a.fits, e, f"{m}_seed{sd_}.npz"))["pop_draws"].astype(float)
        return [(f(P).mean(1), f(P).std(1)) for _, f in quantities(e)]

    S = {}
    for e in BEX:
        for m in BM:
            for r in by.get((e, m), []):
                S[(e, m, r["seed"])] = stats(e, m, r["seed"])
    TR = {}
    for e in BEX:
        for f in glob.glob(os.path.join(a.data, f"{e}_seed*.npz")):
            sd_ = int(f.rsplit("seed", 1)[1].split(".")[0])
            t = np.load(f)["truth"]
            TR[(e, sd_)] = [(g(t).mean(), g(t).std()) for _, g in quantities(e)]

    def ci(x):
        return np.percentile(x, [2.5, 97.5]) if len(x) > 1 else np.array([x[0], x[0]])

    lines = []
    for e in BEX:
        ms_ = [m for m in BM if (e, m) in by]
        for m in ms_:
            rs = by[(e, m)]
            cov_m, cov_s, n_m, n_s, upper_fixed = 0, 0, 0, 0, []
            for r in rs:
                st_, tr = S[(e, m, r["seed"])], TR[(e, r["seed"])]
                for (mm, ss), (tm, ts) in zip(st_, tr):
                    lo, hi = ci(mm)
                    cov_m += lo <= tm <= hi
                    n_m += 1
                    if ts > 1e-9:
                        lo, hi = ci(ss)
                        cov_s += lo <= ts <= hi
                        n_s += 1
                    else:
                        upper_fixed.append(ci(ss)[1])
            point = len(S[(e, m, rs[0]["seed"])][0][0]) == 1
            dims = rs[0]["n_out_fit"] // rs[0]["n_blocks_fit"]
            lab = f"{e.replace('_', ' ')} ({rs[0]['n_blocks_fit']}$\\times${dims})" if m == ms_[0] else ""
            lines.append(
                f"{lab} & {LABEL[m]} & {np.mean([r['final']['ed_theta'] for r in rs]):.3f} & "
                + ("-- & -- & " if point else f"{cov_m}/{n_m} & {cov_s}/{n_s} & ")
                + (f"{np.mean(upper_fixed):.2f}" if upper_fixed else "--")
                + f" & {np.mean([(r['final'].get('n_solves') or 0) + (r['final'].get('n_solves_train') or 0) for r in rs]):.1e}"
                + f" & {np.mean([r['final']['seconds'] for r in rs]) / 60:.1f} \\\\"
                + (" \\addlinespace" if m == ms_[-1] and e != BEX[-1] else ""))
    open(os.path.join(a.out, f"{name}.tex"), "w").write(
        "\\begin{tabular}{llrrrrrr}\n\\toprule\n"
        "example (blocks$\\times$dims) & method & $\\theta$ ED & mean covered & sd covered & "
        "sd of fixed rates & model evaluations & minutes \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    ncol = max(len(quantities(e)) for e in BEX)
    MKB = {"iter_logistic": "o", "iter_mlp": "P", "bb_logistic": "s", "bb_logistic_ssa": "s", "bayes_gauss": "^",
           "bayes_dpm": "D", "bayes_dpm_nre": "v", "bayes_gauss_nre": "v"}
    fig, axes = plt.subplots(len(BEX), ncol, figsize=(1.45 * ncol + 0.5, 1.75 * len(BEX) + 0.6), squeeze=False)
    for i, e in enumerate(BEX):
        qs = quantities(e)
        ms_ = [m for m in BM if (e, m) in by]
        seeds = sorted(r["seed"] for r in by[(e, ms_[0])])
        for k in range(ncol):
            ax = axes[i][k]
            if k >= len(qs):
                ax.axis("off")
                continue
            for si, sd_ in enumerate(seeds):
                ts = TR[(e, sd_)][k][1]
                ax.plot([-0.5, len(ms_) - 0.5], [ts, ts], color="k", lw=0.6, ls=":", zorder=0)
                for j, m in enumerate(ms_):
                    if (e, m, sd_) not in S:
                        continue
                    ss = S[(e, m, sd_)][k][1]
                    x = j + (si - (len(seeds) - 1) / 2) * 0.22
                    lo, hi = ci(ss)
                    if len(ss) > 1:
                        ax.plot([x, x], [lo, hi], color=COLOR[m], lw=1.6, solid_capstyle="round")
                    ax.plot(x, np.median(ss), MKB[m], ms=3.5, color=COLOR[m], mec="white", mew=0.4)
            ax.set_xticks([])
            ax.set_ylim(bottom=0)
            ax.set_title(qs[k][0], fontsize=7, pad=2)
            ax.tick_params(labelsize=6, length=2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[i][0].set_ylabel(e.replace("_", " ") + "\npopulation sd", fontsize=7)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color="k", lw=0.6, ls=":", label="true population sd")] + [
        Line2D([], [], color=COLOR[m], marker=MKB[m], ms=4, lw=1.6, label=LABEL[m]) for m in BM
        if any((e, m) in by for e in BEX)]
    fig.legend(handles=hs, loc="upper center", ncol=3, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (1.75 * len(BEX) + 0.6)))
    fig.savefig(os.path.join(a.out, f"{name}.pdf"))
    fig.savefig(os.path.join(a.out, f"{name}.png"), dpi=150)
    plt.close(fig)


if BEX:
    bayes_summary(BEX, BM, "bayes")
if SSA_EX:
    bayes_summary(SSA_EX, SSA_M, "ssa")

# ---- figure 9 + table: robustness tests on the gene example
# test 1 (gene_truth): every method starts at the true population; distance to it over the run.
# test 2 (gene_n*): more cells per block at a fixed simulation budget; error, coverage, cost.
ROB = [e for e in cfg["examples"] if cfg["examples"][e].get("experiment") == "robustness" and any(k[0] == e for k in by)]
if ROB:
    def pop_cover(e, m, sd_):
        """fractions of parameters whose 95% interval covers the true population mean / sd"""
        P = np.load(os.path.join(a.fits, e, f"{m}_seed{sd_}.npz"))["pop_draws"].astype(float)
        if len(P) < 2:
            return None
        t = np.load(os.path.join(a.data, f"{e}_seed{sd_}.npz"))["truth"]
        lo_m, hi_m = np.percentile(P.mean(1), [2.5, 97.5], axis=0)
        lo_s, hi_s = np.percentile(P.std(1), [2.5, 97.5], axis=0)
        tm, ts = t.mean(0), t.std(0)
        return float(np.mean((lo_m <= tm) & (tm <= hi_m))), float(np.mean((lo_s <= ts) & (ts <= hi_s)))

    NS = sorted([(cfg["examples"][e]["n_cells"], e) for e in ROB if e.startswith("gene_n")])
    T1 = [m for m in ["iter_logistic_truth", "bayes_gauss_truth", "bayes_gauss_nre_truth",
                      "bayes_gauss_nre_small_truth"] if ("gene_truth", m) in by]
    T2 = [m for m in ["iter_logistic", "bb_logistic", "bayes_gauss", "bayes_gauss_nre"] if any((e, m) in by for _, e in NS)]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.9))
    ax = axes[0]
    for m in T1:
        for i, r in enumerate(by[("gene_truth", m)]):
            h = [x for x in r["history"] if "ed_theta" in x]
            n_tot = max(x["sweep"] for x in h)
            ax.plot([x["sweep"] / n_tot for x in h], [x["ed_theta"] for x in h], color=COLOR[m], lw=1.2,
                    ls="--" if "small" in m else "-", alpha=0.85, label=LABEL[m] if i == 0 else None)
    ax.set_xlabel("fraction of run", fontsize=8)
    ax.set_ylabel("distance to true population", fontsize=8)
    ax.set_title("Test 1: start at the truth", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    MK2 = {"iter_logistic": "o", "bb_logistic": "s", "bayes_gauss": "^", "bayes_gauss_nre": "v"}
    for m in T2:
        xs, ed, cm, cs, ev = [], [], [], [], []
        for n, e in NS:
            rs = by.get((e, m), [])
            if not rs:
                continue
            xs.append(n)
            ed.append(np.mean([r["final"]["ed_theta"] for r in rs]))
            ev.append(np.mean([(r["final"].get("n_solves") or 0) + (r["final"].get("n_solves_train") or 0) for r in rs]))
            c = [pop_cover(e, m, r["seed"]) for r in rs]
            c = [x for x in c if x is not None]
            cm.append(np.mean([x[0] for x in c]) if c else np.nan)
            cs.append(np.mean([x[1] for x in c]) if c else np.nan)
        kw = dict(color=COLOR[m], marker=MK2[m], ms=4, lw=1.3, label=LABEL[m])
        axes[1].plot(xs, ed, **kw)
        if not np.all(np.isnan(cs)):
            axes[2].plot(xs, cs, **kw)
            axes[2].plot(xs, cm, color=COLOR[m], marker=MK2[m], ms=3, lw=0.9, ls=":")
        axes[3].plot(xs, ev, **kw)
    axes[1].set_title("Test 2: error", fontsize=8)
    axes[1].set_ylabel("distance to true population", fontsize=8)
    axes[2].set_title("Test 2: 95% interval coverage\n(solid: population sd, dotted: mean)", fontsize=8)
    axes[2].axhline(0.95, color="k", lw=0.6, ls=":")
    axes[2].set_ylim(-0.03, 1.03)
    axes[3].set_title("Test 2: forward-model evaluations", fontsize=8)
    axes[3].set_yscale("log")
    axes[1].set_yscale("log")
    for ax in axes[1:]:
        ax.set_xscale("log")
        ax.set_xlabel("cells per block", fontsize=8)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=6.5, frameon=False, bbox_to_anchor=(0.62, 0))
    for ax in axes:
        ax.tick_params(labelsize=6.5, length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(a.out, "robustness.pdf"))
    fig.savefig(os.path.join(a.out, "robustness.png"), dpi=150)
    plt.close(fig)
    lines = []
    for m in T1:
        rs = by[("gene_truth", m)]
        h0 = [np.mean([[x for x in r["history"] if "ed_theta" in x][0]["ed_theta"] for r in rs])]
        lines.append(f"start at truth & {LABEL[m]} & -- & {h0[0]:.3f} & {np.mean([r['final']['ed_theta'] for r in rs]):.3f} & -- & -- \\\\")
    lines[-1] += " \\addlinespace" if lines and NS else ""
    for n, e in NS:
        for m in T2:
            rs = by.get((e, m), [])
            if not rs:
                continue
            c = [x for x in (pop_cover(e, m, r["seed"]) for r in rs) if x is not None]
            cov = (f"{np.mean([x[0] for x in c]):.2f} & {np.mean([x[1] for x in c]):.2f}") if c else "-- & --"
            lines.append(f"{n} cells/block & {LABEL[m]} & "
                         f"{np.mean([(r['final'].get('n_solves') or 0) + (r['final'].get('n_solves_train') or 0) for r in rs]):.1e} & -- & "
                         f"{np.mean([r['final']['ed_theta'] for r in rs]):.3f} & {cov} \\\\")
    open(os.path.join(a.out, "robustness.tex"), "w").write(
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "setting & method & evaluations & $\\theta$ ED at start & $\\theta$ ED at end & mean covered & sd covered \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- figure 10 + table: fish functional response (synthetic twins and real data)
FEX = [e for e in cfg["examples"] if cfg["examples"][e].get("experiment") == "fish" and any(k[0] == e for k in by)]
if FEX:
    VOR = np.log(5.0)  # log a above this: voracious (the two synthetic modes are at -2.2 and 5.7)
    FM = [m for m in ["iter_logistic", "iter_bins", "iter_mlp", "bb_bins", "bayes_gauss", "bayes_dpm"]
          if any((e, m) in by for e in FEX)]
    fmodel = make_model("fish")

    def fstats(P):  # P: (S, n, 2) populations -> per population: voracious fraction, mean and sd of log r
        return (P[:, :, 0] > VOR).mean(1), P[:, :, 1].mean(1), P[:, :, 1].std(1)

    def iv(x):
        x = np.asarray(x)
        if len(x) == 1:
            return f"{x[0]:.2f}"
        lo, md, hi = np.percentile(x, [2.5, 50, 97.5])
        return f"{md:.2f} [{lo:.2f}, {hi:.2f}]"

    lines = []
    for e in FEX:
        sd0 = 1
        t = np.load(os.path.join(a.data, f"{e}_seed{sd0}.npz"))["truth"]
        if len(t):
            tv, tm, ts = fstats(t[None])
            lines.append(f"{e.replace('_', ' ')} & true population & -- & {tv[0]:.2f} & {tm[0]:.2f} & {ts[0]:.2f} & -- \\\\")
        for m in FM:
            rs = by.get((e, m), [])
            if not rs:
                continue
            P = np.load(os.path.join(a.fits, e, f"{m}_seed{rs[0]['seed']}.npz"))["pop_draws"].astype(float)
            v, mr, sr = fstats(P)
            ed = [r["final"].get("ed_theta") for r in rs]
            ed = f"{np.mean(ed):.3f}" if ed[0] is not None else "--"
            lines.append(f"{'' if len(t) else e.replace('_', ' ') if m == FM[0] else ''} & {LABEL[m]} & {ed} & {iv(v)} & {iv(mr)} & {iv(sr)} & "
                         f"{np.mean([r['final']['seconds'] for r in rs]) / 60:.1f} \\\\")
        lines[-1] += " \\addlinespace" if e != FEX[-1] else ""
    open(os.path.join(a.out, "fish.tex"), "w").write(
        "\\begin{tabular}{llrrrrr}\n\\toprule\nexample & method & $\\theta$ ED & voracious fraction & "
        "mean $\\log r$ & sd $\\log r$ & minutes \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    SC = [m for m in ["iter_bins", "iter_mlp", "bayes_gauss", "bayes_dpm"] if m in FM]
    fig, axes = plt.subplots(len(FEX), len(SC) + 2, figsize=(2.3 * (len(SC) + 2), 2.3 * len(FEX)), squeeze=False)
    rng_f = np.random.default_rng(0)
    for i, e in enumerate(FEX):
        z = np.load(os.path.join(a.data, f"{e}_seed1.npz"))
        t = z["truth"]
        panels = [("true population", t if len(t) else None, "k")] + [
            (LABEL[m], np.load(os.path.join(a.fits, e, f"{m}_seed1.npz"))["draws"] if (e, m) in by else None, COLOR[m]) for m in SC]
        for k, (lab, d, col) in enumerate(panels):
            ax = axes[i][k]
            if d is None:
                ax.axis("off")
                continue
            d = d[rng_f.choice(len(d), min(len(d), 1500), replace=False)]
            ax.scatter(d[:, 0], d[:, 1], s=2, color=col, alpha=0.35, lw=0)
            ax.axvline(VOR, color="k", lw=0.5, ls=":")
            ax.set_xlim(fmodel.lower[0], fmodel.upper[0])
            ax.set_ylim(fmodel.lower[1], fmodel.upper[1])
            ax.set_title(lab, fontsize=6.5, pad=2)
            ax.set_xlabel("log a", fontsize=7)
            if k == 0 or (k == 1 and not len(t)):
                ax.set_ylabel(f"{e.replace('_', ' ')}\nlog r", fontsize=7)
        # posterior predictive: fraction of fish that eat every prey, by n0
        ax = axes[i][-1]
        n0s = np.array(fmodel.n0s)
        full = [np.mean(z[f"block{j}"][:, 0] == n0) for j, n0 in enumerate(n0s)]
        ax.plot(n0s, full, "ko", ms=4, label="data", zorder=5)
        for m in SC:
            if (e, m) not in by:
                continue
            d = np.load(os.path.join(a.fits, e, f"{m}_seed1.npz"))["draws"]
            q = fmodel.simulate(d, rng_f)
            ax.plot(n0s, (q == n0s[None, :]).mean(0), color=COLOR[m], lw=1.3, label=LABEL[m])
        ax.set_xlabel("initial prey", fontsize=7)
        ax.set_title("fraction eating all prey", fontsize=6.5, pad=2)
        if i == 0:
            ax.legend(fontsize=5.5, frameon=False)
    for ax in axes.ravel():
        ax.tick_params(labelsize=6, length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fish.pdf"))
    fig.savefig(os.path.join(a.out, "fish.png"), dpi=150)
    plt.close(fig)

# ---- figure 4: population mean +- sd across cells, truth vs methods, every parameter
PM = [m for m in ["cmc_paper", "iter_logistic", "iter_mlp"] if m in METHODS]
MARK = {"cmc_paper": "s", "iter_logistic": "o", "iter_mlp": "D"}
EX = [e for e in ["scalar", "linear", "growth", "gene"] if all((e, m) in by for m in PM)]
if EX and PM:
    ncol = max(len(by[(e, PM[0])][0]["param_names"]) for e in EX)
    fig, axes = plt.subplots(len(EX), ncol, figsize=(1.25 * ncol + 0.6, 1.55 * len(EX) + 0.5), squeeze=False)
    for i, e in enumerate(EX):
        names = by[(e, PM[0])][0]["param_names"]
        for k in range(ncol):
            ax = axes[i][k]
            if k >= len(names):
                ax.axis("off")
                continue
            F = by[(e, PM[0])][0]["final"]
            tm, ts = F["truth_mean"][k], F["truth_sd"][k]
            ax.axhline(tm, color="k", lw=0.6, ls=":", zorder=0)
            ax.errorbar(0, tm, yerr=ts if ts > 1e-9 else None, fmt="_", ms=10, color="k", lw=1.6, capsize=0)
            for j, m in enumerate(PM, start=1):
                Fs = [r["final"] for r in by[(e, m)]]
                mu = np.mean([f["mean"][k] for f in Fs])
                sd = np.mean([f["sd"][k] for f in Fs])
                ax.errorbar(j, mu, yerr=sd, fmt=MARK[m], ms=4, color=COLOR[m], lw=1.6, capsize=0,
                            mec="white", mew=0.5)
            ax.set_xlim(-0.6, len(PM) + 0.6)
            ax.set_xticks([])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=6, length=2)
            ax.set_title(names[k].replace("log_g", "log γ").replace("log_", "log ").replace("theta", "θ"), fontsize=7, pad=2)
        axes[i][0].set_ylabel(e, fontsize=8, fontweight="bold")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="k", marker="_", ms=10, lw=1.6, label="true population")] + [
        Line2D([], [], color=COLOR[m], marker=MARK[m], ms=4, lw=1.6, label=LABEL[m]) for m in PM]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(a.out, "params.pdf"))
    fig.savefig(os.path.join(a.out, "params.png"), dpi=150)
    plt.close(fig)
print(md)
