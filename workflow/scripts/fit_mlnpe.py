"""Amortised Bayesian multilevel model baseline: multilevel neural posterior estimation (ML-NPE) of
Habermann, Schmitt, Kühmichel, Bulling, Radev & Bürkner (2025, Bayesian Analysis, "Amortized Bayesian
Multilevel Models"), built from BayesFlow 2.1.0 components (Keras 3, torch backend, CPU).

The paper's method (their Sec. 2.3, eqs. 10-14). A two-level model with hyperparameters tau, local
parameters lambda_j of each group j and data x_j factorises the posterior as
    p(tau, {lambda_j} | x) = p(tau | x) prod_j p(lambda_j | tau, x_j),
learned by four networks trained jointly on simulations from the joint model:
  - a local summary network h_local(x_j) (one fixed-length vector per group),
  - a global summary network h_global({h_local(x_j)}) (permutation invariant over groups),
  - a global inference network q(tau | h_global) (normalising flow),
  - one local inference network, shared by all groups, q(lambda_j | tau, h_local(x_j)),
with loss -log q(tau | .) - sum_j log q(lambda_j | tau, .) (their eq. 13); after training,
posterior draws come from ancestral sampling (eq. 14).

Our model in that form. Groups are cells; x_j is cell j's tracked time series (all outputs of the
single tracked block); lambda_j = theta_j, the cell's parameters (model.param_names); tau = (mu, log s),
the mean and log sd of a diagonal Gaussian population on theta (as Arruda et al. 2024, `arruda`):
theta_j ~ N(mu, diag(s^2)), x_j ~ our simulator, cmc.models.make_model(...).simulate_cols, so the
networks are trained on our model. Hyperpriors (the choice of Arruda et al.):
  - mu_k ~ the model's prior on theta_k (normal prior_mean, prior_sd for `transfection`; uniform on
    the box [lower, upper] for the Gillespie models, which have a box prior);
  - log s_k ~ N(log(0.1 c_k), 1.5^2), weakly informative, with c_k the prior sd of theta_k (box width
    / sqrt(12) for a box), i.e. a median population sd of a tenth of the prior sd, with 95% of the mass
    between 0.005 and 1.9 prior sds. (Arruda et al. put N(-log prior var, 5^2) on the log precision,
    i.e. sd 2.5 on log s: too wide to simulate from, the population would cover the whole prior.)
Cells' theta are clipped to the box before simulation (for `transfection`, prior mean +- 4 prior sd,
the box `sourcerer` uses), so that no simulation overflows or runs for ever; the reported populations
are clipped the same way (the clipped Gaussian is the population the networks were trained on).

Architecture and training: their Experiment 1 (the time-series case study), BayesFlow defaults otherwise.
  - local summary: LSTM(64) -> Dense(256, ELU) -> Dense(32) on the cell's (time x species) sequence;
  - global summary: bf.networks.SetTransformer, summary dim 32, 16 inducing points, two attention
    blocks of width 64, 4 heads, dropout 0.01 (BayesFlow 2's set transformer; its key dim is set by the
    block width, the paper's was 32);
  - global and local inference networks: neural spline flows, bf.networks.CouplingFlow(transform=
    "spline"), 6 coupling layers, each with 2 dense layers of 256 ReLU units and 5% dropout;
  - Adam, initial learning rate 5e-4, cosine decay (BayesFlow 2's schedule: 5% linear warm-up from a
    tenth of it, decay to 0; its clipnorm 1.5), batch size 32 datasets;
  - targets standardised (tau with its hyperprior moments, theta with simulated moments) and data
    standardised per time and species (BayesFlow standardises the same way); counts enter as log1p.

Deviations, with reasons:
  1. Many groups. The paper trains on 10-30 groups per data set and applies the networks only within
     the range seen in training. We have ~2000-2900 cells, so training data sets have J ~ DiscreteUniform
     (J_obs / 2, J_obs) cells (one J per batch, every data set in the batch a random subset of J of its
     cells), which covers the observed J. The set transformer pools by attention, which cannot see J,
     so log J is appended to the global summary (as BayesFlow 2.1's experimental MultilevelEstimator
     does with log1p of the group counts), letting the posterior width scale with J.
  2. The local term of the loss is the mean over cells (as BayesFlow 2.1's MultilevelEstimator
     computes it: each network's loss is its mean NLL over rows) rather than the sum, and is estimated
     on `n_local` = 64 random cells per data set (unbiased for the mean). The summaries use all J
     cells; evaluating the local flow on all ~80k cells of a batch would cost ~10x more per step on
     2 CPU cores. The local networks are trained as in the paper but are not used for the output.
  3. Budget. The paper trains offline for 200 epochs on M = 20k-100k simulated data sets (a GPU). On 2
     CPU cores: the ODE model (`transfection`) costs nothing to simulate, so training is online (fresh
     data sets every step, BayesFlow's online mode) for `train_minutes`; the Gillespie models are
     simulated offline into a bank of data sets for `sim_minutes` (2 processes), then trained for
     `train_minutes` epochs over the bank (BayesFlow's offline mode). The number of steps follows from
     the measured step time so that the cosine schedule ends with the budget. A batch holds at most
     `max_batch_cells` cells (memory of the LSTM's backward pass), so with ~2500 cells the batch size
     can drop below 32 (recorded).
  4. Output: S = n_keep (50) draws of (mu, s) from the global network given all observed cells; each
     gives n_pop cells from N(mu, diag s^2), clipped to the box.

Diagnostics recorded: calibration of the global network on held-out simulated data sets at J = J_obs
(coverage of central 50% and 90% intervals for each of mu and log s), training/validation losses.

Writes an npz that fit.py (method kind `external`) scores like every other method: pop_draws
(S x n_pop x p), seconds (simulation + training + inference), n_train_sims (cell simulations, all
phases), n_train_datasets, n_cells, n_times, steps, and the timings of each phase.

Run with --check to simulate a few data sets and exit before importing Keras (a smoke test)."""
import argparse
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import yaml

from cmc.models import make_model

ap = argparse.ArgumentParser()
ap.add_argument("--example", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--check", action="store_true", help="smoke test: simulate a little and stop before training")
a = ap.parse_args()
sys.stdout.reconfigure(line_buffering=True)  # progress reaches the job log even if the job is killed

cfg = yaml.safe_load(open(a.config))
ex = cfg["examples"][a.example]
meth = cfg["methods"][a.method]
train_minutes = float(meth.get("train_minutes", 240))
sim_minutes = float(meth.get("sim_minutes", 80))  # simulation bank (stochastic simulators only)
batch_size = int(meth.get("batch_size", 32))
max_batch_cells = int(meth.get("max_batch_cells", 65536))
n_local = int(meth.get("n_local", 64))
j_lo_frac = float(meth.get("j_lo_frac", 0.5))
max_cells = meth.get("max_cells")  # optional subset of the observed cells (None: all)
lr0 = float(meth.get("learning_rate", 5e-4))
n_keep = int(meth.get("n_keep", 50))
workers = int(meth.get("workers", 2))
threads = int(meth.get("threads", 2))
n_val = int(meth.get("n_val", 128))
max_bank_gb = float(meth.get("max_bank_gb", 3.0))
ls_offset = float(meth.get("log_sd_prior_offset", math.log(0.1)))
ls_sd = float(meth.get("log_sd_prior_sd", 1.5))
n_pop = int(cfg["settings"].get("n_pop", 1000))
if a.check:
    sim_minutes, n_val = 0.05, 4

model = make_model(ex.get("kind", a.example), **ex["model"])
P = model.p
fit_blocks = [j for j in range(len(model.blocks)) if j not in ex["holdout"]]
if len(model.blocks) != 1 or fit_blocks != [0]:
    raise SystemExit("ML-NPE needs grouped data: one tracked block (every cell observed at all times)")
COLS = model.blocks[0]
N_OBS = int(getattr(model, "n_obs", 1))
T = len(COLS) // N_OBS
STOCHASTIC = bool(getattr(model, "stochastic_sim", False))
rng = np.random.default_rng(1000 * a.seed + 41)

# ---- observed cells
Y = np.load(a.data)["block0"].astype(float)
ok = np.all(np.isfinite(Y), axis=1)
Y = Y[ok]
n_dropped = int((~ok).sum())
if max_cells and len(Y) > int(max_cells):
    Y = Y[rng.choice(len(Y), int(max_cells), replace=False)]
J_HI = len(Y)
J_LO = max(2, int(round(j_lo_frac * J_HI)))
print(f"{model.name}: {J_HI} cells x {T} times x {N_OBS} species ({n_dropped} dropped as non-finite); "
      f"training J in [{J_LO}, {J_HI}]")

# ---- hyperprior on tau = (mu, log s) and the box the cells' theta are clipped to
if model.lower is not None:
    LO, HI = np.asarray(model.lower, float), np.asarray(model.upper, float)
    M0, C = (LO + HI) / 2, (HI - LO) / np.sqrt(12)
else:
    M0, C = np.asarray(model.prior_mean, float), np.asarray(model.prior_sd, float)
    LO, HI = M0 - 4 * C, M0 + 4 * C
LS_MEAN = np.log(C) + ls_offset
G_MEAN = np.concatenate([M0, LS_MEAN])
G_SD = np.concatenate([C, np.full(P, ls_sd)])  # exact sd of the uniform too: width / sqrt(12)


def draw_global(n, r):
    mu = r.uniform(LO, HI, (n, P)) if model.lower is not None else M0 + C * r.standard_normal((n, P))
    return np.hstack([mu, LS_MEAN + ls_sd * r.standard_normal((n, P))])


def transform(y):
    return np.log1p(np.maximum(y, 0)) if model.discrete else y


def simulate(g, J, r):
    """data sets with hyperparameters g (B x 2P), J cells each -> theta (B, J, P), x (B, J, T, N_OBS)"""
    B = len(g)
    mu, s = g[:, None, :P], np.exp(g[:, None, P:])
    th = np.clip(mu + s * r.standard_normal((B, J, P)), LO, HI)
    y = transform(model.simulate_cols(th.reshape(-1, P), COLS, r))
    return th.astype(np.float32), y.reshape(B, J, T, N_OBS).astype(np.float32)


def _sim_chunk(args):
    """one task of the simulation bank (module level so that worker processes can run it)"""
    seed, nd = args
    r = np.random.default_rng(seed)
    g = draw_global(nd, r)
    th, y = simulate(g, J_HI, r)
    return g.astype(np.float32), th, y


t_start = time.time()
n_sims = 0  # cell simulations, all phases
if STOCHASTIC:  # offline: a bank of data sets of J_HI cells, simulated for sim_minutes
    per_set = J_HI * (T * N_OBS + P) * 4
    cap = int(max_bank_gb * 1e9 // per_set)
    nd = max(1, 8000 // J_HI)  # data sets per task: ~8000 cells per vectorised Gillespie call
    deadline = t_start + 60 * sim_minutes
    ctx = mp.get_context("fork")
    parts, pending, k = [], [], 0
    with ctx.Pool(workers) as pool:
        while True:
            while len(pending) < 2 * workers and time.time() < deadline and k * nd < cap:
                pending.append(pool.apply_async(_sim_chunk, ((10 ** 7 * a.seed + k, nd),)))
                k += 1
            if not pending:
                break
            parts.append(pending.pop(0).get())
            if len(parts) % 50 == 0:
                print(f"bank: {len(parts) * nd} data sets, {(time.time() - t_start) / 60:.1f} min")
    bank_g = np.concatenate([p_[0] for p_ in parts])
    bank_th = np.concatenate([p_[1] for p_ in parts])
    bank_y = np.concatenate([p_[2] for p_ in parts])
    del parts
    good = np.all(np.isfinite(bank_y.reshape(len(bank_y), -1)), axis=1)
    bank_g, bank_th, bank_y = bank_g[good], bank_th[good], bank_y[good]
    n_sims += int(k * nd * J_HI)
    nv = min(n_val, max(1, len(bank_g) // 10))
    val_g, val_th, val_y = bank_g[-nv:], bank_th[-nv:], bank_y[-nv:]
    bank_g, bank_th, bank_y = bank_g[:-nv], bank_th[:-nv], bank_y[:-nv]
    pil_th, pil_y = bank_th[:256], bank_y[:256]
    print(f"bank: {len(bank_g)} training and {nv} validation data sets of {J_HI} cells "
          f"({(~good).sum()} non-finite dropped), {(time.time() - t_start) / 60:.1f} min")
else:  # online: a pilot for the standardisation and a fixed validation set
    J_pil = min(J_HI, 256)
    pil_th, pil_y = simulate(draw_global(256, rng), J_pil, rng)
    val_g = draw_global(n_val, rng).astype(np.float32)
    val_th, val_y = simulate(val_g, J_HI, rng)
    n_sims += 256 * J_pil + n_val * J_HI
    bank_g = None
t_sim = time.time() - t_start

# standardisation: data per (time, species), local targets per parameter
Y_MEAN = pil_y.reshape(-1, T, N_OBS).mean(0)
Y_SD = pil_y.reshape(-1, T, N_OBS).std(0) + 1e-6
TH_MEAN = pil_th.reshape(-1, P).mean(0)
TH_SD = pil_th.reshape(-1, P).std(0) + 1e-6
y_obs = transform(Y).reshape(1, J_HI, T, N_OBS).astype(np.float32)
print("data standardisation: mean range", np.round([Y_MEAN.min(), Y_MEAN.max()], 3),
      "observed mean range", np.round([y_obs.mean((0, 1)).min(), y_obs.mean((0, 1)).max()], 3))
if a.check:
    print("check: bank" if STOCHASTIC else "check: online", "val", val_y.shape, val_th.shape,
          "pilot", pil_y.shape, f"{t_sim:.1f} s; stopping before training")
    sys.exit(0)

# ---- networks (Keras 3 + BayesFlow 2.1.0, torch backend)
os.environ.setdefault("KERAS_BACKEND", "torch")
import torch  # noqa: E402

torch.set_num_threads(threads)
import keras  # noqa: E402
import bayesflow as bf  # noqa: E402

assert keras.backend.backend() == "torch", keras.backend.backend()
keras.utils.set_random_seed(1000 * a.seed + 43)
B = max(1, min(batch_size, max_batch_cells // J_HI))
K = min(n_local, J_LO)
LOG_J_HI = math.log(J_HI)


class MLNPE(keras.Model):
    def __init__(self):
        super().__init__()
        self.local_summary = keras.Sequential([keras.layers.LSTM(64), keras.layers.Dense(256, activation="elu"),
                                               keras.layers.Dense(32)])
        self.global_summary = bf.networks.SetTransformer(summary_dim=32, embed_dims=(64, 64), num_heads=(4, 4),
                                                         num_inducing_points=16, dropout=0.01)
        nsf = dict(depth=6, transform="spline", subnet_kwargs=dict(widths=(256, 256), activation="relu", dropout=0.05))
        self.global_net = bf.networks.CouplingFlow(**nsf)
        self.local_net = bf.networks.CouplingFlow(**nsf)

    def call(self, y, training=False):
        return self.summaries(y, training)[1]

    def summaries(self, y, training=False):
        nb, nj = y.shape[0], y.shape[1]
        h = self.local_summary(keras.ops.reshape(y, (nb * nj, T, N_OBS)), training=training)
        h = keras.ops.reshape(h, (nb, nj, -1))
        s = self.global_summary(h, training=training)
        log_j = keras.ops.full((nb, 1), math.log(nj) / LOG_J_HI - 1.0)
        return h, keras.ops.concatenate([s, log_j], axis=-1)

    def local_conditions(self, g, h):
        nb, nk = h.shape[0], h.shape[1]
        gk = keras.ops.broadcast_to(keras.ops.expand_dims(g, 1), (nb, nk, 2 * P))
        return keras.ops.reshape(keras.ops.concatenate([gk, h], axis=-1), (nb * nk, -1))

    def losses_(self, y, g, th_k, idx, training):
        h, cg = self.summaries(y, training)
        stage = "training" if training else "validation"
        lg = self.global_net.compute_metrics(g, conditions=cg, stage=stage)["loss"]
        hk = keras.ops.take(h, idx, axis=1)
        ll = self.local_net.compute_metrics(keras.ops.reshape(th_k, (-1, P)), conditions=self.local_conditions(g, hk),
                                            stage=stage)["loss"]
        return lg + ll, lg, ll


def tensor(x):
    return keras.ops.convert_to_tensor(np.asarray(x, np.float32))


def std_y(y):
    return (y - Y_MEAN) / Y_SD


def std_g(g):
    return (g - G_MEAN) / G_SD


def std_th(th):
    return (th - TH_MEAN) / TH_SD


n_online = [0]  # cells simulated by the online training steps


def make_batch(r):
    J = int(r.integers(J_LO, J_HI + 1))
    if STOCHASTIC:
        bi = r.choice(len(bank_g), B, replace=False)
        cells = r.choice(J_HI, J, replace=False)
        g, th, y = bank_g[bi], bank_th[np.ix_(bi, cells)], bank_y[np.ix_(bi, cells)]
    else:
        g = draw_global(B, r)
        th, y = simulate(g, J, r)
        n_online[0] += B * J
        keep = np.all(np.isfinite(y.reshape(B, -1)), axis=1)
        g, th, y = g[keep], th[keep], y[keep]
    idx = r.choice(J, K, replace=False)
    return J, tensor(std_y(y)), tensor(std_g(g)), tensor(std_th(th[:, idx])), keras.ops.convert_to_tensor(idx)


net = MLNPE()
J0, y0, g0, th0, idx0 = make_batch(rng)
net.local_summary.build((None, T, N_OBS))
net.global_summary.build((y0.shape[0], J0, 32))
net.global_net.build((y0.shape[0], 2 * P), (y0.shape[0], 33))
net.local_net.build((y0.shape[0] * K, P), (y0.shape[0] * K, 2 * P + 32))
with torch.no_grad():
    net.losses_(y0, g0, th0, idx0, training=False)
weights = net.trainable_weights
print(f"networks built: {sum(int(np.prod(v.shape)) for v in weights)} trainable weights; batch {B} data sets, "
      f"{K} cells per data set in the local loss")
opt = keras.optimizers.Adam(learning_rate=lr0, clipnorm=1.5)


def lr_at(step, total):
    warm = max(1, int(0.05 * total))
    if step < warm:
        return lr0 * (0.1 + 0.9 * step / warm)
    return 0.5 * lr0 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


def train_step(batch):
    J, y, g, th, idx = batch
    net.zero_grad()
    with torch.enable_grad():
        loss, lg, ll = net.losses_(y, g, th, idx, training=True)
    loss.backward()
    grads = [v.value.grad if v.value.grad is not None else torch.zeros_like(v.value) for v in weights]
    with torch.no_grad():
        opt.apply(grads, weights)
    for v in weights:
        v.value.grad = None
    return float(loss), float(lg), float(ll)


def val_loss():
    out = []
    with torch.no_grad():
        for s0 in range(0, len(val_g), B):
            sl = slice(s0, s0 + B)
            idx = np.arange(K)
            out.append([float(v) for v in net.losses_(tensor(std_y(val_y[sl])), tensor(std_g(val_g[sl])),
                                                      tensor(std_th(val_th[sl][:, idx])),
                                                      keras.ops.convert_to_tensor(idx), training=False)])
    return np.mean(out, axis=0)


def sample_global(y, S):
    """S posterior draws of tau per data set y (n, J, T, N_OBS), unstandardised: (n, S, 2P)"""
    out = []
    with torch.no_grad():
        for s0 in range(0, len(y), B):
            _, cg = net.summaries(tensor(std_y(y[s0:s0 + B])))
            nb = cg.shape[0]
            cond = keras.ops.reshape(keras.ops.repeat(keras.ops.expand_dims(cg, 1), S, axis=1), (nb * S, -1))
            z = keras.ops.convert_to_numpy(net.global_net.sample((nb * S,), conditions=cond))
            out.append(z.reshape(nb, S, 2 * P) * G_SD + G_MEAN)
    return np.concatenate(out)


# ---- training: steps fitted to the time budget (the cosine schedule ends with the budget)
t1 = time.time()
budget = 60 * train_minutes
total, step, epoch_steps = 10 ** 9, 0, (len(bank_g) // B if STOCHASTIC else None)
hist, vals, t_steps = [], [], []
while True:
    if step == 30 or (step > 30 and step % 200 == 0):  # re-fit the number of steps to the remaining time
        per = float(np.mean(t_steps[-200:]))
        total = max(step + 1, step + int(0.97 * (budget - (time.time() - t1)) / per))
        print(f"step {step}: {per:.2f} s per step, total {total} steps")
    if step >= total or time.time() - t1 > budget:
        break
    ts = time.time()
    opt.learning_rate = lr_at(step, total)
    hist.append(train_step(make_batch(rng)))
    t_steps.append(time.time() - ts)
    step += 1
    if step % 100 == 0:
        h = np.mean(hist[-100:], axis=0)
        print(f"step {step}: loss {h[0]:.3f} (global {h[1]:.3f}, local {h[2]:.3f}), lr {lr_at(step, total):.2e}, "
              f"{(time.time() - t1) / 60:.1f} min")
    if step % max(100, total // 20) == 0:
        vals.append([step, *val_loss()])
        print(f"  validation at step {step}: loss {vals[-1][1]:.3f} (global {vals[-1][2]:.3f}, local {vals[-1][3]:.3f})")
t_train = time.time() - t1
n_train_datasets = len(bank_g) if STOCHASTIC else step * B
n_sims += n_online[0]
print(f"training done: {step} steps, {t_train / 60:.1f} min" +
      (f", {step / max(1, epoch_steps):.1f} epochs over the bank" if STOCHASTIC else ""))

# ---- calibration of the global network on the held-out simulated data sets (J = J_obs)
post = sample_global(val_y, 200)
z = (post - val_g[:, None, :])
q = (z < 0).mean(1)  # posterior quantile of the truth
cov50 = np.mean(np.abs(q - 0.5) <= 0.25, axis=0)
cov90 = np.mean(np.abs(q - 0.5) <= 0.45, axis=0)
rmse = np.sqrt(np.mean((post.mean(1) - val_g) ** 2, axis=0))
print("validation coverage 50%:", np.round(cov50, 2), "\nvalidation coverage 90%:", np.round(cov90, 2),
      "\nposterior-mean RMSE / prior sd:", np.round(rmse / G_SD, 3))

# ---- inference on the observed cells
t2 = time.time()
draws = sample_global(y_obs, n_keep)[0]
mu, s = draws[:, :P], np.exp(draws[:, P:])
pops = np.clip(mu[:, None, :] + s[:, None, :] * rng.standard_normal((n_keep, n_pop, P)), LO, HI)
t_inf = time.time() - t2
print("posterior mean of mu:", np.round(mu.mean(0), 3), "\nposterior mean of s:", np.round(s.mean(0), 4),
      f"\ninference {t_inf:.2f} s")
seconds = time.time() - t_start
np.savez_compressed(a.out, pop_draws=pops.astype(np.float32), seconds=seconds, t_sim=t_sim, t_train=t_train,
                    t_infer=t_inf, n_train_sims=n_sims, n_train_datasets=n_train_datasets, n_cells=J_HI,
                    n_cells_dropped=n_dropped, n_times=T, steps=step, batch_size=B, n_local=K, j_lo=J_LO, j_hi=J_HI,
                    mu_draws=mu, sd_draws=s, train_loss=np.array(hist, np.float32), val_loss=np.array(vals, np.float32),
                    val_cov50=cov50, val_cov90=cov90, val_rmse=rmse, offline=STOCHASTIC,
                    hyper_mean=G_MEAN, hyper_sd=G_SD, box_lo=LO, box_hi=HI)
print("done", round(seconds / 60, 1), "min")
