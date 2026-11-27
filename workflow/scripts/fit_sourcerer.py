"""Sourcerer baseline (Vetter et al. 2024, NeurIPS): a sample-based maximum-entropy estimate of the
population ("source distribution"), using their code (github.com/mackelab/sourcerer, pinned commit) for
the source model, the surrogate, the entropy estimate and the sliced-Wasserstein distance.

Their procedure, with their default settings (conf/source/wasserstein.yaml, conf/surrogate/
train_surrogate.yaml, conf/source_model/sampler.yaml):
  1. train a conditional RealNVP surrogate of the simulator, p(y | theta), on simulations with theta
     uniform on a box (the "box domain"; their reference distribution for the entropy);
  2. train a neural sampler (noise -> theta in the box) to minimise
        (1 - lambda) log SW(push-forward through the surrogate, data) - lambda H(source),
     lambda decaying linearly from 1 to 0.3 over 500 steps, then 3000 steps at 0.3, early stopping.

Adaptation to our data, which have several blocks measured on different cells: the surrogate models
all fitted outputs of a cell jointly (one simulation per theta), and the distance term is the mean over
blocks of log SW between the block's columns of the push-forward and that block's data (with one block
this is their objective). Outputs are standardised with the data's mean and sd; counts are dequantised
(log(1 + y + U), U ~ uniform(0, 1)) so that a flow can model them. The box is the model's parameter box,
or prior mean +- 4 prior sd for models with a Gaussian prior.

Writes an npz that fit.py (method kind `external`) scores like every other method."""
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

from cmc.models import make_model

ap = argparse.ArgumentParser()
ap.add_argument("--example", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.repo))
from sourcerer.fit_surrogate import create_train_val_dataloaders, fit_conditional_normalizing_flow  # noqa: E402
from sourcerer.real_nvp import RealNVPs, Sampler  # noqa: E402
from sourcerer.sliced_wasserstein import sliced_wasserstein_distance  # noqa: E402

cfg = yaml.safe_load(open(a.config))
ex = cfg["examples"][a.example]
meth = cfg["methods"][a.method]
n_train = int(meth.get("n_train", 200_000))
n_step = int(meth.get("n_step", 4096))  # cells per block per optimisation step
threads = int(meth.get("threads", 2))
torch.set_num_threads(threads)
torch.manual_seed(1000 * a.seed + 3)
rng = np.random.default_rng(1000 * a.seed + 5)

model = make_model(ex.get("kind", a.example), **ex["model"])
z = np.load(a.data)
fit_blocks = [j for j in range(len(model.blocks)) if j not in ex["holdout"]]
data = {j: z[f"block{j}"].astype(float) for j in fit_blocks}
if model.lower is not None:
    lo, hi = model.lower.astype(float), model.upper.astype(float)
else:
    lo, hi = model.prior_mean - 4 * model.prior_sd, model.prior_mean + 4 * model.prior_sd
cols = np.concatenate([model.blocks[j] for j in fit_blocks])
pos = {j: np.arange(sum(len(model.blocks[i]) for i in fit_blocks[:k]),
                    sum(len(model.blocks[i]) for i in fit_blocks[:k + 1])) for k, j in enumerate(fit_blocks)}


def transform(y, r):
    return np.log1p(y + r.uniform(size=y.shape)) if model.discrete else y


Yd = {j: transform(data[j], rng) for j in fit_blocks}
mu = np.concatenate([Yd[j].mean(0) for j in fit_blocks])
sd = np.concatenate([Yd[j].std(0) for j in fit_blocks]) + 1e-6
Yd = {j: torch.tensor((Yd[j] - mu[pos[j]]) / sd[pos[j]], dtype=torch.float32) for j in fit_blocks}

t0 = time.time()
# 1. surrogate of the simulator on the box
u = rng.uniform(-1, 1, size=(n_train, model.p))
theta = lo + (hi - lo) * (u + 1) / 2
Y = np.vstack([model.simulate_cols(theta[s:s + 20000], cols, rng) for s in range(0, n_train, 20000)])
Y = (transform(Y, rng) - mu) / sd
ok = np.all(np.isfinite(Y), axis=1)
print(f"surrogate training set: {ok.sum()} of {n_train} finite, {Y.shape[1]} outputs", flush=True)
surrogate = RealNVPs(flow_length=8, data_dim=Y.shape[1], context_dim=model.p, hidden_layer_dim=50)
opt = torch.optim.Adam(surrogate.parameters(), lr=1e-4, weight_decay=5e-5)
tr, va = create_train_val_dataloaders(torch.tensor(Y[ok], dtype=torch.float32),
                                      torch.tensor(u[ok], dtype=torch.float32))
fit_conditional_normalizing_flow(network=surrogate, optimizer=opt, training_dataset=tr, validation_dataset=va,
                                 nb_epochs=1000, early_stopping_patience=50)
surrogate.eval()
for prm in surrogate.parameters():
    prm.requires_grad_(False)
t_sur = time.time() - t0

# 2. source: maximum entropy subject to matching the data (their train_source, summed over blocks)
source = Sampler(xdim=model.p, input_noise_dim=model.p, hidden_layer_dim=100, num_hidden_layers=3)
opt = torch.optim.Adam(source.parameters(), lr=1e-4, weight_decay=1e-5)
lam = torch.cat([torch.linspace(1.0, 0.3, 500), 0.3 * torch.ones(3000)])
n_s = min(n_step, min(len(Yd[j]) for j in fit_blocks))
best, wait, hist = np.inf, 0, []
for step in range(len(lam)):
    src = source.sample(n_s)
    entropy_loss = source.reverse_kld(None, 512)
    push = surrogate.sample(context=src)
    y_loss = 0.0
    for j in fit_blocks:
        idx = torch.randperm(len(Yd[j]))[:n_s]
        y_loss = y_loss + torch.log(sliced_wasserstein_distance(push[:, pos[j]], Yd[j][idx], num_projections=500, p=2))
    y_loss = y_loss / len(fit_blocks)
    loss = (1 - lam[step]) * y_loss + lam[step] * entropy_loss
    if torch.isfinite(loss):
        opt.zero_grad()
        loss.backward()
        opt.step()
    hist.append(float(loss))
    if step % 100 == 0:
        print(f"step {step}: loss {float(loss):.4f}, log SW {float(y_loss):.4f}, entropy loss {float(entropy_loss):.4f}",
              flush=True)
    if step >= 500:  # their early stopping, after the lambda decay
        if hist[-1] < best:
            best, wait = hist[-1], 0
        else:
            wait += 1
            if wait > 500:
                print("early stop at step", step)
                break
source.eval()
with torch.no_grad():
    s = source.sample(int(cfg["settings"].get("n_pop", 200)) * 10).numpy()
pop = lo + (hi - lo) * (s + 1) / 2
seconds = time.time() - t0
np.savez_compressed(a.out, pop_draws=pop[None].astype(np.float32), seconds=seconds, t_surrogate=t_sur,
                    n_train_sims=n_train, n_cells=int(sum(len(d) for d in data.values())),
                    n_times=len(cols), steps=len(hist))
print("done", round(seconds / 60, 1), "min")
