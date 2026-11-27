"""Amortised NLME baseline: the method of Arruda et al. (2024, ICML), run with their code and their
pretrained network, as in their notebook `Examples/Amortized NLME Inference.ipynb`.

Runs in its own pinned environment (workflow/envs/arruda.yaml; TensorFlow + BayesFlow 1.1.4 +
pyPESTO 0.4.1) against a clone of github.com/arrjon/Amortized-NLME-Models at a fixed commit.

Model: for `transfection_ssa`, their SDE model (chemical Langevin; same 8 parameters, order and
measurement as our Gillespie model) and its pretrained network. As published, their Euler-Maruyama
solver calls np.random.seed(42) before drawing its increments, so every training simulation shares one
Brownian path; we use the network as published. Otherwise their 'simple Fröhlich' model, which is our
`transfection` model (same ODE, measurement,
noise and normal prior). Their network was trained on 180 measurements per cell (every 10 min,
1/6 h to 30 h) and its summary network sees only the sequence of values, not the times, so it is
given the data on that grid:
  - real data (eGFP, d2eGFP): all 180 time points of every cell (the other methods use the hourly
    subset, 30 points);
  - synthetic twin: a fresh data set of the same number of cells, drawn from the same population as
    the twin (constants copied from cmc/models.py, Transfection), simulated on their grid with their
    simulator. The other methods see a different draw of cells at 30 points; the score (distance to
    a large sample of the population) is computed identically by fit.py.

Steps (their defaults): 100 amortised posterior draws per cell; maximise their Monte Carlo estimate
of the population likelihood for a Gaussian population with diagonal covariance (the variance of
log sigma fixed at its lower bound), 10 multi-starts with L-BFGS; then their Bayesian version, an
adaptive Metropolis sampler on the population parameters with their priors, for uncertainty.

Writes an npz that fit.py (method kind `external`) scores like every other method:
pop_draws (S x n_pop x 6, log parameters), seconds, n_train_sims (simulations used to train the
pretrained network), beta_hat, psi_hat."""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--example", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fine", default=None, help="synthetic twin on their grid (make_data_fine.py)")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
ex = cfg["examples"][a.example]
meth = cfg["methods"][a.method]
n_starts = int(meth.get("n_multi_starts", 10))
n_post = int(meth.get("n_samples_opt", 100))
n_mcmc = int(meth.get("n_mcmc", 10000))
n_keep = int(meth.get("n_keep", 50))
n_pop = int(cfg["settings"].get("n_pop", 200))
workers = int(meth.get("workers", 2))

repo = os.path.abspath(a.repo)
sys.path.insert(0, repo)
from scipy import stats  # noqa: E402
from pypesto import Problem, sample  # noqa: E402
from pypesto.objective import AggregatedObjective, NegLogParameterPriors, Objective  # noqa: E402

from inference.helper_functions import create_fixed_params, create_mixed_effect_model_param_names  # noqa: E402
from inference.inference_functions import run_population_optimization  # noqa: E402
from models.froehlich_model_simple import FroehlichModelSimple, batch_simulator  # noqa: E402
from models.froehlich_model_sde import FroehlichModelSDE  # noqa: E402

SDE = ex.get("kind") == "transfection_ssa"  # their SDE model and network (same 8 parameters as ours)

# population of the synthetic twin (cmc/models.py, Transfection: Arruda et al.'s 'true' population)
POP_MEAN = np.array([-0.69364, -7.01357, 6.21662, -0.16357, 2.07944, -3.45388])
POP_VAR = np.array([0.94098, 7.01402, 0.00396, 0.11641, 0.0, 0.0])

rng = np.random.default_rng(1000 * a.seed + 29)
np.random.seed(1000 * a.seed + 31)  # their simulator draws its noise from the global generator
if "data_tracks" in ex:
    rows = [r for r in open(ex["data_tracks"]).read().splitlines() if not r.startswith("#")]
    t = np.array([float(r.split(",")[0]) for r in rows[2:]])
    Y = np.array([[float(x) for x in r.split(",")[1:]] for r in rows[2:]])
    keep = t <= 30
    obs = np.log(Y[keep].T)[:, :, None].astype(np.float32)
elif a.fine:  # twin simulated with our simulator (exact Gillespie for the stochastic model)
    obs = np.load(a.fine)["obs"][:, :, None].astype(np.float32)
else:
    assert not SDE
    theta = POP_MEAN + np.sqrt(POP_VAR) * rng.standard_normal((ex["n_cells"], 6))
    obs = batch_simulator(theta, n_obs=180, with_noise=True)
assert obs.shape[1] == 180, obs.shape
ok = np.all(np.isfinite(obs), axis=(1, 2))
obs = obs[ok]
print(f"{len(obs)} cells x {obs.shape[1]} time points ({(~ok).sum()} dropped as non-finite)")

model = (FroehlichModelSDE if SDE else FroehlichModelSimple)(load_best=True, prior_type="normal")
net_dir = os.path.join(repo, "networks", model.network_name)
model.build_trainer(net_dir)  # restores the latest checkpoint into model.amortizer
hist = sorted(f for f in os.listdir(net_dir) if f.startswith("history_"))
n_train_sims = None
if hist:
    h = pickle.load(open(os.path.join(net_dir, hist[-1]), "rb"))
    tl = h.get("train_losses") if isinstance(h, dict) else None
    if tl is not None:
        n_train_sims = int(len(tl) * model.batch_size)  # one batch of fresh simulations per iteration
print("network", model.network_name, "training simulations", n_train_sims)

t0 = time.time()
names = create_mixed_effect_model_param_names(model.param_names, cov_type="diag")
fix_names = ["var-$\\sigma$"]
fixed_indices, fixed_values = create_fixed_params(fix_names=fix_names, fixed_values=[0] * len(fix_names),
                                                  params_list=names)
fixed_indices, uniq = np.unique(np.array(fixed_indices), return_index=True)
fixed_values = np.array(fixed_values)[uniq]
res, obj, problem = run_population_optimization(
    individual_model=model, data=obs, param_names=names, cov_type="diag",
    n_multi_starts=n_starts, n_samples_opt=n_post,
    x_fixed_indices=fixed_indices, x_fixed_vals=fixed_values,
    pesto_multi_processes=workers, verbose=False)
best = res.optimize_result.x[0]  # pyPESTO sorts the starts by objective value, best first
beta_hat, psi_inv, _, _ = obj.get_params(best)
psi_hat = np.linalg.inv(psi_inv)
t_opt = time.time() - t0
print("optimisation done", round(t_opt / 60, 1), "min; mean", np.round(beta_hat, 3),
      "var", np.round(np.diag(psi_hat), 4))

# their Bayesian version: priors on the population parameters, adaptive Metropolis
inv_var_prior_mean = -np.log(model.prior_cov.diagonal())
inv_var_prior_std = np.ones(len(inv_var_prior_mean)) * 5
P = model.n_params
prior = NegLogParameterPriors(
    [{"index": problem.full_index_to_free_index(i),
      "density_fun": (lambda x, i=i: -stats.norm.logpdf(x, loc=model.prior_mean[i],
                                                       scale=np.sqrt(model.prior_cov.diagonal()[i])))}
     for i in range(P) if i not in fixed_indices]
    + [{"index": problem.full_index_to_free_index(i + P),
        "density_fun": (lambda x, i=i: -stats.norm.logpdf(x, loc=inv_var_prior_mean[i], scale=inv_var_prior_std[i]))}
       for i in range(P) if i + P not in fixed_indices])
bayes_problem = Problem(objective=AggregatedObjective([Objective(obj), prior]),
                        lb=problem.lb_full, ub=problem.ub_full, x_names=problem.x_names,
                        x_scales=problem.x_scales, x_fixed_indices=problem.x_fixed_indices,
                        x_fixed_vals=problem.x_fixed_vals, x_priors_defs=prior)
x0 = problem.get_reduced_vector(best)
res = sample.sample(bayes_problem, n_samples=n_mcmc, sampler=sample.AdaptiveMetropolisSampler(),
                    result=res, x0=x0)
burn = sample.geweke_test(res)
trace = np.asarray(res.sample_result.trace_x[0])[int(burn or 0):]
print("MCMC: burn-in", burn, "kept", len(trace))
idx = np.linspace(0, len(trace) - 1, n_keep).astype(int)
pops = []
for x in trace[idx]:
    full = problem.get_full_vector(x) if len(x) < len(names) else x
    b, pi, _, _ = obj.get_params(np.asarray(full))
    C = np.linalg.inv(pi)
    pops.append(rng.multivariate_normal(b, C, size=n_pop))
seconds = time.time() - t0
np.savez_compressed(a.out, pop_draws=np.array(pops, np.float32), seconds=seconds, t_opt=t_opt,
                    n_train_sims=-1 if n_train_sims is None else n_train_sims,
                    beta_hat=beta_hat, psi_hat=psi_hat, n_cells=len(obs), n_times=obs.shape[1],
                    burn_in=-1 if burn is None else int(burn), network=model.network_name)
print("done", round(seconds / 60, 1), "min")
