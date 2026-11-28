"""Display names and colours of the methods, shared by summarise.py and plot_ppc.py."""
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
LABEL.update({"iter_mlp_s15": "Iterated, MLP, step 1.5", "iter_mlp_s2": "Iterated, MLP, step 2",
              "iter_mlp_always": "Iterated, MLP, resample every update",
              "iter_mlp_always_s2": "Iterated, MLP, resample every update, step 2",
              "iter_mlp_truth": "Iterated, MLP"})
for k in ("iter_mlp_s15", "iter_mlp_s2", "iter_mlp_always", "iter_mlp_always_s2", "iter_mlp_truth"):
    COLOR.setdefault(k, COLOR["iter_mlp"])
LABEL.update({"bb_mlp": "Iterated, MLP + Bayesian bootstrap", "bb_mlp_ssa": "Iterated, MLP + Bayesian bootstrap",
              "bayes_gauss_nre_big": "Bayes, Gaussian mixed effects, likelihood-free",
              "bayes_gauss_nre_big1": "Bayes, Gaussian mixed effects, likelihood-free",
              "bayes_dpm_nre_big": "Bayes, DP mixture, likelihood-free",
              "bayes_dpm_nre_big1": "Bayes, DP mixture, likelihood-free"})
COLOR.update({"bb_mlp": "#f4a582", "bb_mlp_ssa": "#f4a582"})
LABEL.update({"arruda": "Amortised NLME (Arruda et al.)", "sourcerer": "Sourcerer (Vetter et al.)",
              "nlme_saem": "NLME software, SAEM (nlmixr2)", "nlme_focei": "NLME software, FOCEI (nlmixr2)"})
COLOR.update({"arruda": "#1b7837", "sourcerer": "#b35806", "nlme_saem": "#2166ac", "nlme_focei": "#92c5de"})
for k in ("bayes_gauss_nre_big", "bayes_gauss_nre_big1", "bayes_dpm_nre_big", "bayes_dpm_nre_big1"):
    COLOR[k] = "#4d4d4d"
LABEL.update({"neb": "Neural empirical Bayes (Vandegar et al.)", "npmle": "NPMLE, exact likelihood",
              "npmle_bb": "NPMLE, exact likelihood + Bayesian bootstrap", "pmetrics_npag": "NPAG (Pmetrics)",
              "mlnpe": "Amortised multilevel NPE (Habermann et al.)",
              "mlnpe_ssa": "Amortised multilevel NPE (Habermann et al.)", "pepsdi": "PEPSDI (Persson et al.)"})
COLOR.update({"neb": "#762a83", "npmle": "#000000", "npmle_bb": "#666666", "pmetrics_npag": "#01665e",
              "mlnpe": "#d6604d", "mlnpe_ssa": "#d6604d", "pepsdi": "#8c510a"})


def complete(methods):
    """labels and colours for every configured method (a missing label falls back to the key)"""
    for k in methods:
        LABEL.setdefault(k, k.replace("_", " "))
        COLOR.setdefault(k, "#999999")
