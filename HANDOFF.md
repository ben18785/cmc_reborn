# Handoff: state of the CMC-for-stochastic-models project

Last updated 2026-11-26. This is the working brief for whoever (human or agent) picks the
project up. It records what we are trying to show, how the code and pipeline work, what the
experiments found, which earlier statements were wrong, and what is left to do.

## 1. The project in one paragraph

We extend Contour Monte Carlo (CMC; Lambert et al. 2021) from deterministic models with one
output to stochastic forward maps (including Gillespie simulators) and many outputs, and write
a paper about it (candidate journal: *Bayesian Analysis*). The paper lives in a separate
Overleaf clone, `../cmc_stochastic_overleaf` (`main.tex`; pushing its `main` branch syncs to
Overleaf; there is no local LaTeX). This repo (`cmc_reborn`, public, branch `prototype`) holds
the package `cmc`, a Snakemake workflow and the results.

### Agreed framing (the user signed off on this)
- CMC is simulation-only and consistent by construction: it never approximates the *model*,
  only the *comparison* between data and simulations. The true population is a fixed point
  for any classifier (theorem in the paper appendix `app:robust`).
- No up-front training; changing the model, design or prior is a cheap rerun (amortised
  methods must retrain).
- Works for deterministic or stochastic models and for multi-species (multi-output) blocks.
- Uncertainty via the Bayesian bootstrap.
- Iterated block-wise CMC is EM / Richardson-Lucy for the nonparametric MLE (NPMLE) of the
  hierarchical model; the prior acts as a reference measure.
- **Claim that must be reworded** (see section 5): "cost independent of the number of cells".
  Accurate version: cost grows far more slowly than hierarchical MCMC (cells x sweeps x
  proposals), but simulations and sweeps must grow with the data.

## 2. Working conventions (from the user)
- Code changes on feature branches, never `main` (work is on `prototype`).
- All fits and simulation checks run on GitHub Actions, not on the user's Mac. Small unit
  tests and summarising locally are fine.
- Analyses go through the Snakemake workflow with pinned environments, not ad hoc scripts.
- In the paper always write `eq.~\eqref{...}`, never a bare `\eqref`.
- Never put the user's personal details (e.g. email) into requests to external services.

## 3. Code and pipeline

### Package `cmc/`
| file | contents |
|---|---|
| `infer.py` | `iterate_blocks` (iterated block-wise CMC: per update, fit a ratio estimator data/pushforward for one block, reweight particles by the averaged ratio over `n_reps` noisy outputs, Liu-West rejuvenation when ESS < `ess_frac`), `one_step_joint` (original CMC / Butler et al. augmented). `Settings` dataclass. `n_solves` counts forward-model runs. |
| `ratio.py` | ratio estimators: `kde`, `logistic2` (logistic on quadratic features), `mlp`, `logistic_bins` (one indicator per count; whole pmf of a 1-d count block). `fit(data, sim, weight=None)`; weights currently only for logistic kinds. |
| `bayes.py` | fully Bayesian hierarchical comparator: truncated stick-breaking DP mixture of Gaussians (`n_comp=1` is the Gaussian mixed-effects model); per-cell MH (independence + 2 adaptive RWs), group shift and linear-map moves, conjugate Normal/inverse-Wishart updates. `lik="exact"` or `lik="nre"` (neural ratio estimation per block, sequential retraining near current cells, trust region = range of training parameters). |
| `ssa.py` | vectorised exact Gillespie SSA, optional per-cell start time. |
| `models.py` | models: `scalar`, `linear`, `growth`, `gene`, `bd_ssa`, `feedback_ssa`, `fish`, `transfection`, `transfection_ssa` (registry `MODELS`). Model API: `latent`, `observe_cols`, `simulate_cols`, `latent_cols`, `log_lik`/`block_log_lik`, `canonical`, `stochastic_sim` flag. |

Tests: `tests/` (26 pass: `.venv/bin/python -m pytest -q tests`).

### Workflow
- Config: `config/config.yaml` (examples, methods, seeds, defaults); `config/smoke.yaml`
  mirrors it at tiny sizes for a quick end-to-end check.
- Examples carry `kind` (model), optional `experiment` (`robustness`, `ssa`, `fish`,
  `transfection`) which routes them to a summary section, optional per-example `settings`.
- Scripts: `workflow/scripts/make_data.py`, `fit.py`, `summarise.py`,
  `prep_transfection.py`.
- Real data: `data/fish/` (from the user's public functional-response repo) and
  `data/transfection/` (Arruda et al., Zenodo 10.5281/zenodo.1228899, CC BY-SA 4.0).

### Running
- Python: `.venv/bin/python` (plain `python` is not on PATH). For Snakemake:
  `PATH=$PWD/.venv/bin:$PATH snakemake ...`.
- Fits on GitHub: `gh workflow run experiments.yml --ref prototype -f only='<regex on example-method-seedN>'`.
  One job per (example, method, seed); ~20 run concurrently.
- Collecting results: `gh run download <id> -p 'fit-*' -D <scratch dir>`, then copy each
  artifact's `fits/` and `data/` into `results/` (`cp -R`), and commit with `git add -f`.
- Summaries: the Actions `summarise` job fails for partial runs (it needs every fit). Run it
  locally instead:
  `.venv/bin/python workflow/scripts/summarise.py --fits results/fits --data results/data --config config/config.yaml --out <dir>`.
  Tables report **model runs** (forward-model solves: simulations, or exact likelihood
  evaluations) and **surrogate evals** (NRE network passes) separately.

## 4. Results so far

Distance = energy distance between fitted and true population of per-cell parameters
(synthetic data only; ~0.01 is essentially exact). Coverage = parameters x seeds whose 95%
interval contains the true population mean / sd. 3 seeds throughout. Unless stated, CMC runs
used default settings (4k training sims, 20k particles, 20 sweeps), which section 4.3 shows
are too small for large data.

### 4.1 CMC vs full Bayes (examples `linear_bayes`, `growth_snapshot`, `growth_tracked`)
| example | method | distance | means cov. | sds cov. | model runs | surrogate evals |
|---|---|---|---|---|---|---|
| linear | CMC | 0.001 | - | - | 1.2e5 | - |
| | CMC + bootstrap | 0.001 | 15/15 | 13/15 | 2.6e6 | - |
| | exact Bayes Gaussian | 0.001 | 15/15 | 15/15 | 1.2e9 | - |
| | exact Bayes DP mixture | 0.001 | 15/15 | 15/15 | 1.2e9 | - |
| | likelihood-free Bayes | 0.002 | 13/15 | 12/15 | 2.9e6 | 1.2e9 |
| growth snapshot | CMC | 0.64 | - | - | not recorded | - |
| | CMC + bootstrap | 0.66 | 9/21 | 3/12 | 2.3e6 | - |
| | exact Bayes Gaussian | 0.12 | 18/21 | 5/12 | 6.0e8 | - |
| | exact Bayes DP mixture | 0.37 | 9/21 | 4/12 | 6.1e8 | - |
| | likelihood-free Bayes | 1.55 | 5/21 | 0/12 | 1.4e6 | 5.5e8 |
| growth tracked | CMC | 0.30 | - | - | not recorded | - |
| | CMC + bootstrap | 0.31 | 11/21 | 6/12 | 6.0e6 | - |
| | exact Bayes Gaussian | 0.091 | 21/21 | 6/12 | 1.0e8 | - |
| | exact Bayes DP mixture | 0.094 | 21/21 | 6/12 | 1.0e8 | - |
| | likelihood-free Bayes | 1.64 | 5/21 | 4/12 | 2.4e5 | 9.1e7 |

Exact Bayes wins when the likelihood is cheap and correctly specified. The bootstrap is
calibrated where CMC is accurate (linear) and under-covers where CMC is biased (growth; 20
sweeps is too few). The growth CMC point fits predate the run counter.

### 4.2 Robustness test 1: start at the truth (`gene_truth`)
CMC 0.001 -> 0.001; exact Bayes 0.001 -> 0.002; likelihood-free Bayes 0.001 -> 2.27 (40k
sims/block) and 0.007 -> 1.97 (5k). Confirms the fixed-point theorem; a learned likelihood
pulls the sampler off the truth regardless of training budget.

### 4.3 Robustness test 2: growing the data (`gene_n*`, `gene_big*`, `gene_huge32000`)
Gene model: induced transcription of 3 genes, snapshots at 8 times, 6 log-rate parameters per
cell, correlated Gaussian population, exact Poisson counts with 30% capture. Exact Bayes has the
correct population model (best case for Bayes).

Default settings:
| cells/block | 500 | 2,000 | 8,000 | 32,000 |
|---|---|---|---|---|
| CMC logistic distance | 0.12 | 0.09 | 0.10 | 0.11 |
| CMC NN distance | 0.049 | 0.087 | 0.10 | 0.13 |
| CMC model runs | 1.5e5 | 1.2e5 | 1.2e5 | 1.3e5 |
| bootstrap (logistic) means / sds cov. | 0.78 / 0.11 | 0.67 / 0.00 | 0.67 / 0.00 | 0.44 / 0.00 |
| exact Bayes distance | 0.007 | 0.008 | 0.008 | not run |
| exact Bayes means / sds cov. | 1.00 / 0.89 | 1.00 / 0.50 | 0.94 / 0.39 | - |
| exact Bayes model runs | 1.0e8 | 4.0e8 | 1.6e9 | - |
| likelihood-free distance | 2.93 | 0.57 | 1.74 | 2.08 |
| likelihood-free sims / surrogate evals | 9.6e5 / 8.4e7 | 9.6e5 / 3.3e8 | 9.6e5 / 1.3e9 | 9.6e5 / 5.2e9 |

Larger budgets, NN classifier:
| cells/block | training sims / particles / sweeps | distance | minutes |
|---|---|---|---|
| 2,000 | 32k / 80k / 30 | 0.011-0.015 | 5 |
| 8,000 | 32k / 80k / 100 | 0.010-0.023 | 18 |
| 32,000 | 128k / 160k / 100 | 0.022-0.026 | 63 |

Diagnosis: the ~0.1 floor is settings, not the method. Accuracy is limited by (1) classifier
flexibility (logistic2 matches only per-block means/covariances, floor ~0.07; consistent with
Prop. 2 of the theorem), (2) simulation budget, which must grow with N, and (3) sweeps (EM
converges slowly; still falling at sweep 100 for 32k cells; the number of rejuvenations per
sweep falls as N grows, cause not yet understood). With adequate settings NN-CMC approaches
exact Bayes without a likelihood.

### 4.4 Gillespie models (`bd_ssa*`, `feedback_ssa*`)
| example | method | distance | means cov. | sds cov. | model runs | surrogate evals |
|---|---|---|---|---|---|---|
| birth-death snapshot (tractable) | CMC logistic | 0.044 | - | - | 3.8e6 | - |
| | CMC NN | 0.033 | - | - | 3.8e6 | - |
| | CMC + bootstrap | 0.043 | 9/12 | 0/12 | 3.8e7 | - |
| | exact Bayes | 0.002 | 12/12 | 9/12 | 4.1e8 | - |
| | likelihood-free | 0.44 | 1/12 | 2/12 | 9.6e5 | 3.5e8 |
| birth-death tracked | CMC logistic | 0.002 | - | - | 3.8e6 | - |
| | CMC + bootstrap | 0.002 | 8/12 | 8/12 | 3.8e7 | - |
| | likelihood-free | 0.018 | 7/12 | 6/12 | 1.2e5 | 5.1e7 |
| feedback snapshot (intractable) | CMC logistic | 0.72 | - | - | 2.9e6 | - |
| | CMC + bootstrap | 0.72 | 0/18 | 0/6 | 2.9e7 | - |
| | likelihood-free | 1.16 | 1/18 | 0/6 | 7.2e5 | 2.5e8 |
| feedback tracked | CMC logistic | 0.33 | - | - | 2.9e6 | - |
| | CMC + bootstrap | 0.34 | 7/18 | 3/6 | 2.9e7 | - |
| | likelihood-free | 2.73 | 2/18 | 1/6 | 1.2e5 | 4.3e7 |

CMC is 1.6-8x more accurate than likelihood-free Bayes on the intractable model but used
*more* simulations (3-4e6 vs 1e5-1e6). Feedback is partly unidentified: rates fixed in the
truth keep sd ~0.8 (snapshot) / ~0.5 (tracked). Bootstrap under-covers wherever CMC is biased.

### 4.5 Fish functional response (`fish_sim`, `fish_sim_big`, `fish_real`)
Per-fish (log a, log r), 7 prey densities, Gillespie simulation, likelihood via a per-fish
matrix exponential (exact but costly). Synthetic truth is bimodal (90% voracious).
| data | method | voracious frac. | mean log r | sd log r | distance | min |
|---|---|---|---|---|---|---|
| twin, 95/density | truth | 0.90 | 2.40 | 0.60 | - | - |
| | CMC counts | 0.81 | 2.65 | 0.71 | 0.18 | 1.0 |
| | CMC counts + bootstrap | 0.82 [0.75, 0.88] | 2.61 | 0.66 | 0.23 | 20 |
| | CMC NN | 0.82 | 2.38 | 0.23 | 0.93 | 1.2 |
| | exact Bayes Gaussian | 0.58 [0.51, 0.64] | 2.53 | 0.56 | 0.98 | 40 |
| | exact Bayes DP mixture | 0.82 [0.76, 0.87] | 2.59 | 0.58 | 0.26 | 41 |
| twin, 2000/density | truth | 0.90 | 2.41 | 0.61 | - | - |
| | CMC counts | 0.90 | 2.49 | 0.68 | 0.10 | 1.1 |
| | CMC counts + bootstrap | 0.88 [0.86, 0.89] | 2.50 | 0.70 | 0.10 | 20 |
| real | CMC counts + bootstrap | 0.86 [0.82, 0.92] | 2.54 | 0.82 | - | 17 |
| | exact Bayes DP mixture | 0.93 [0.86, 0.97] | 2.36 | 0.76 | - | 41 |
| | exact Bayes Gaussian | 0.56 [0.49, 0.63] | 2.52 | 0.64 | - | 38 |

The Gaussian mixed-effects model is misspecified (misses bimodality, invents an a-r
correlation, fails the "eats all prey" predictive check). CMC finds the bimodality unaided and
agrees with the DP mixture on real data. Use `logistic_bins` for 1-d counts; the NN collapses
the spread.

### 4.6 mRNA transfection (Arruda et al. 2024 data; `tf_*`)
All 5,488 cells (2,565 eGFP, 2,923 d2eGFP), hourly points 1-30 h, each cell one 30-d tracked
block. ODE model (exact Gaussian likelihood) and a Gillespie version. Synthetic twins use
Arruda et al.'s "true" population. The mean trajectory is symmetric in (delta, gamma), so
labels are canonicalised to delta >= gamma.

| data | method | distance | means cov. | sds cov. | model runs | surrogate evals | min |
|---|---|---|---|---|---|---|---|
| ODE twin | CMC NN | 0.19 | - | - | 2.1e5-3.0e6 (seeds differ; check) | - | 1.5 |
| | CMC logistic | 0.82 | - | - | 3.0e6 | - | 4.9 |
| | CMC logistic + bootstrap | 2.46 | 18/18 | 8/12 | 5.9e7 | - | 77 |
| | exact Bayes Gaussian | 0.34 | 3/18 | 0/12 | 6.5e7 | - | 1.0 |
| | likelihood-free | 16.5 | 2/18 | 1/12 | 1.2e5 | 6.5e7 | 2.5 |
| Gillespie twin | CMC NN | 0.34 | - | - | 2.4e6 | - | 27 |
| | CMC logistic | 2.31 | - | - | 2.4e6 | - | 20 |
| | CMC logistic + bootstrap | 1.05 | 16/24 | 6/15 | 2.4e7 | - | 221 |
| | likelihood-free (old code) | 2.66 | 5/24 | 4/15 | 1.2e5 | 4.5e7 | 3.5 |

Real-data population means (log scale; delta, gamma, km0, t0, offset, sigma):
- eGFP: CMC NN (-2.05, -3.61, 5.93, 0.29, 3.33, -3.21); exact Bayes (-2.20, -3.45, 5.69, 0.19, 2.16, -3.11).
- d2eGFP: CMC NN (-1.63, -1.88, 6.20, 1.27, 2.94, -1.04); exact Bayes (-2.03, -2.48, 5.72, 0.40, 2.43, -2.35).

NN-CMC beats exact Bayes with the exact likelihood on the ODE twin: the truth has one shared
background offset that is barely identified; the Gaussian model gives it sd ~2.2 and becomes
overconfident elsewhere (SAEM/FOCEI fits in Arruda et al. use the same population model, so
may share this). Logistic classifiers fail on 30-d trajectories. eGFP: methods agree; d2eGFP:
faster protein decay as expected, but methods disagree on t0 and sigma (needs predictive
checks and comparison with Arruda et al.'s estimates). Possible issue in Arruda et al.'s code:
their SDE solver appears to call `np.random.seed(42)` inside the Euler-Maruyama function, so
every simulation may share one Brownian path. Read in the code, not verified by running.

### 4.7 The likelihood-free Bayes baseline
Same hierarchical model and sampler as exact Bayes; each cell's likelihood replaced by neural
ratio estimation (Hermans et al. 2020), per block, 40k sims per round, 3 rounds retrained near
current cells. With unbounded parameters the ReLU logit extrapolated without bound and cells
ran away (crashes on `linear_bayes` and `tf_*`). Fixed by returning -inf outside the training
parameter range. After the fix: linear fine (0.002-0.003); transfection converges confidently
to wrong values. It illustrates the surrogate-likelihood failure the theorem describes, but it
is home-built and ran on a smaller simulation budget than CMC, so a referee may call it a
strawman.

## 5. Corrections to earlier statements
- Earlier summaries called likelihood-free "evaluations" simulations. They were mostly
  surrogate network passes. Likelihood-free Bayes actually used **fewer** simulations than
  CMC (often 4-30x). The advantage of CMC over it is accuracy, not simulation count. Tables
  now separate the two (commit "Summaries: report forward-model runs and surrogate
  evaluations separately").
- "Cost independent of the number of cells" is not right as stated (section 4.3).
- The bootstrap has only been run with logistic classifiers; its calibration at converged
  settings is untested.

## 6. Paper state (`../cmc_stochastic_overleaf/main.tex`)
Written: abstract (framing above; mentions package `cmc`, "fully unit tested (TBD)",
deterministic or stochastic, multi-species), section "Relation to hierarchical models"
(`sec:bayes`, `eq:hierlik`, incl. a "Robustness to the classifier" paragraph), appendix
"The true population is a fixed point for any classifier" (`app:robust`, `prop:truth`,
`prop:moments`). Not yet written: results sections for 4.1-4.7. Abstract cost claim must be
reworded. A duplicate `why_iterate.tex` in the paper repo is unresolved.

## 7. To do (roughly in priority order)
1. Write results sections and figures into the paper; reword the cost claim.
2. Bootstrap calibration at converged settings: enable sample weights for `mlp` in
   `ClassifierRatio.fit` (scikit-learn 1.9.1 supports them), run NN bootstrap on the gene
   example at 2k/8k/32k cells with the larger settings. Needs replicates split into separate
   jobs (each converged run is 5-63 min; 6 h job limit).
3. Matched-budget likelihood-free comparison (give NRE the same simulation count as CMC) and a
   published amortised NLME baseline (e.g. Arruda et al. / BayesFlow) on transfection.
4. Reruns on current code: CMC growth fits (missing run counts), `tf_ssa_*` likelihood-free
   fits (predate the trust region), main examples at converged settings; check the NN
   model-run count discrepancy on `tf_sim`.
5. Transfection: offset identifiability (fix per construct or tighter prior); predictive
   checks and comparison with published estimates for d2eGFP.
6. Understand why rejuvenations per sweep fall with N; consider EM acceleration or forced
   rejuvenation each sweep.
7. Later options: SBC-style calibration study, Stan comparison, Sourcerer, simulation reuse
   across updates, warm-started classifiers, comparison with the user's fish Model 2a posterior.
