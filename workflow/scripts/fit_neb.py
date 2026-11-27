"""Neural Empirical Bayes baseline (Vandegar, Kagan, Wehenkel & Louppe 2021, AISTATS, PMLR 130:2107-2115):
the source distribution (population) is fitted by maximising the marginal likelihood of the observed cells,
    sum_i log p(y_i),   p(y_i) = E_{theta ~ source}[ q(y_i | theta) ],
where q is a conditional normalising-flow surrogate of the simulator, using their biased Monte Carlo estimator
log (1/S) sum_s q(y_i | theta_s) on minibatches of cells.

Code: the NEB parts as adapted into the Sourcerer repository (github.com/mackelab/sourcerer, the same pinned
commit as fit_sourcerer.py): `compute_estimator` and `create_dataloader` (sourcerer/likelihood_estimator.py,
"adapted from github.com/MaximeVandegar/NEB/blob/main/inference/neb/mc_biased.py"), RealNVPs (from NEB's
ml/real_nvps.py), the surrogate training loop (from NEB's ml/ml_helper.py) and the Sampler source model. The
original NEB repository (github.com/MaximeVandegar/NEB, commit 2150e34c0b0b67031b047be585b9ba029082d325, MIT) is
the reference for the defaults below; it is not imported because the Sourcerer copies are the same code with a
cleaner interface, and using them keeps the surrogate identical to the Sourcerer baseline.

Settings. NEB's MC-biased estimator with their example settings (examples/ToyProblems_MonteCarlo_estimators,
also Sourcerer's NEB configuration conf/source/mll.yaml): batches of 128 cells, S = 1024 Monte Carlo draws of
theta per cell, Adam with lr 1e-4, 10% of the cells held out for early stopping. Epochs and patience follow
Sourcerer's NEB configuration (at most 1000 epochs, patience 50 after 10 epochs), since NEB's notebook values
(100 epochs, patience 10) were tuned for 10^4 toy observations. The surrogate is trained exactly as in
fit_sourcerer.py (RealNVP, 8 coupling layers of width 50, on n_train simulations with theta uniform on the box,
Adam lr 1e-4, weight decay 5e-5, at most 1000 epochs, patience 50; dequantised log1p for counts; outputs
standardised by the data).

Deviations, with reasons:
  1. Source model: Sourcerer's Sampler (noise -> MLP with batch norm -> sigmoid onto the box), with Sourcerer's
     source optimiser (weight decay 1e-5), not NEB's unbounded MLP generator (NEB weight decay 0). The surrogate
     is only trained on the box, and an unbounded source can move mass to where the surrogate extrapolates. This
     is also how the Sourcerer paper ran NEB, and it makes the comparison with Sourcerer differ only in the
     objective (marginal likelihood vs entropy-regularised sliced-Wasserstein).
  2. Blocks. In snapshot designs each block is measured on different cells, so the likelihood needs the block
     marginal q(y_b | theta), which a flow over all fitted columns jointly (Sourcerer's surrogate) does not give.
     We use one amortised per-block surrogate: a single conditional RealNVP (same architecture) whose context is
     theta and a one-hot block indicator. Its n_train training simulations are split evenly over the blocks
     (simulation i measures block i mod B only), so the simulation budget and the surrogate's training-set size
     and cost are those of the Sourcerer baseline. The NEB objective is then the sum over all cells of all fitted
     blocks of log p(y_i | block of i), i.e. one pooled data set of (cell, block) pairs, minibatched as in NEB.
     With one block (tracked designs: transfection, *_tracked) there is no indicator and the surrogate is the
     same as Sourcerer's (same seed, simulations and training). Per-block surrogates could equally be used by
     Sourcerer; fit_sourcerer.py is unchanged.
  3. One-dimensional blocks (fish: one count per fish). A RealNVP on one dimension has no coupling and reduces
     to a conditional Gaussian. We append an auxiliary N(0, 1) dimension e, independent of everything, to the
     surrogate's training outputs, and use q(y | theta) = E_{e ~ N(0,1)}[ q(y, e | theta) / phi(e) ], with a
     fresh e for each Monte Carlo draw (so the NEB estimator stays the same biased MC estimator, with e drawn
     alongside theta).
  4. Time budget. NEB's cost is n_cells x S surrogate evaluations per epoch (about 2-3 s per batch of 128 x 1024
     on 2 cores). The fit stops at the first of early stopping, 1000 epochs or `minutes` (default 210) of total
     wall-clock time (at least `min_source_minutes`, default 45, of source training), checked after every batch.
     On the largest examples (gene_n32000: 256k cells) this allows only a few epochs; `stopped` records why
     training ended. As in NEB, the final source (not the best-validation one) is returned.
  5. Pooled cells are shuffled before the 10% validation split (NEB takes the first 10%, which assumes iid
     order; ours are ordered by block).

Writes an npz that fit.py (method kind `external`) scores like every other method."""
import argparse
import math
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
from sourcerer.likelihood_estimator import compute_estimator, create_dataloader  # noqa: E402
from sourcerer.real_nvp import RealNVPs, Sampler  # noqa: E402

cfg = yaml.safe_load(open(a.config))
ex = cfg["examples"][a.example]
meth = cfg["methods"][a.method]
n_train = int(meth.get("n_train", 200_000))
batch_size = int(meth.get("batch_size", 128))
n_mc = int(meth.get("n_mc", 1024))
n_epochs = int(meth.get("n_epochs", 1000))
patience = int(meth.get("patience", 50))
min_epochs = int(meth.get("min_epochs", 10))
minutes = float(meth.get("minutes", 210))
min_source_minutes = float(meth.get("min_source_minutes", 45))
threads = int(meth.get("threads", 2))
torch.set_num_threads(threads)
torch.manual_seed(1000 * a.seed + 3)
rng = np.random.default_rng(1000 * a.seed + 5)

model = make_model(ex.get("kind", a.example), **ex["model"])
z = np.load(a.data)
fit_blocks = [j for j in range(len(model.blocks)) if j not in ex["holdout"]]
B = len(fit_blocks)
data = {j: z[f"block{j}"].astype(float) for j in fit_blocks}
dims = {len(model.blocks[j]) for j in fit_blocks}
if len(dims) != 1:
    raise ValueError(f"the shared per-block surrogate needs blocks of equal size, got {sorted(dims)}")
d = dims.pop()
aux = d == 1  # deviation 3
if model.lower is not None:
    lo, hi = model.lower.astype(float), model.upper.astype(float)
else:
    lo, hi = model.prior_mean - 4 * model.prior_sd, model.prior_mean + 4 * model.prior_sd
cols = np.concatenate([model.blocks[j] for j in fit_blocks])
pos = {j: np.arange(sum(len(model.blocks[i]) for i in fit_blocks[:k]),
                    sum(len(model.blocks[i]) for i in fit_blocks[:k + 1])) for k, j in enumerate(fit_blocks)}


def transform(y, r):
    return np.log1p(y + r.uniform(size=y.shape)) if model.discrete else y


# data preprocessing as fit_sourcerer.py (same random draws, in the same order)
Yd = {j: transform(data[j], rng) for j in fit_blocks}
mu = np.concatenate([Yd[j].mean(0) for j in fit_blocks])
sd = np.concatenate([Yd[j].std(0) for j in fit_blocks]) + 1e-6
Yd = {j: (Yd[j] - mu[pos[j]]) / sd[pos[j]] for j in fit_blocks}

t0 = time.time()
# 1. per-block surrogate q(y_b | theta, b) on the box (deviation 2)
u = rng.uniform(-1, 1, size=(n_train, model.p))
theta = lo + (hi - lo) * (u + 1) / 2
blk = np.arange(n_train) % B
Y = np.empty((n_train, d))
for k, j in enumerate(fit_blocks):
    rows = np.flatnonzero(blk == k)
    Yk = np.vstack([model.simulate_cols(theta[rows[s:s + 20000]], model.blocks[j], rng)
                    for s in range(0, len(rows), 20000)])
    Y[rows] = (transform(Yk, rng) - mu[pos[j]]) / sd[pos[j]]
ok = np.all(np.isfinite(Y), axis=1)
if aux:
    Y = np.hstack([Y, rng.standard_normal((n_train, 1))])
onehot = np.eye(B)[blk] if B > 1 else np.zeros((n_train, 0))
ctx = np.hstack([u, onehot])
print(f"surrogate training set: {ok.sum()} of {n_train} finite, {B} block(s) of {d} output(s)"
      f"{' + 1 auxiliary' if aux else ''}", flush=True)
surrogate = RealNVPs(flow_length=8, data_dim=Y.shape[1], context_dim=ctx.shape[1], hidden_layer_dim=50)
opt = torch.optim.Adam(surrogate.parameters(), lr=1e-4, weight_decay=5e-5)
tr, va = create_train_val_dataloaders(torch.tensor(Y[ok], dtype=torch.float32),
                                      torch.tensor(ctx[ok], dtype=torch.float32))
fit_conditional_normalizing_flow(network=surrogate, optimizer=opt, training_dataset=tr, validation_dataset=va,
                                 nb_epochs=1000, early_stopping_patience=50)
surrogate.eval()
for prm in surrogate.parameters():
    prm.requires_grad_(False)
t_sur = time.time() - t0
print(f"surrogate trained in {t_sur / 60:.1f} min", flush=True)


class BlockLikelihood:
    """log q(y | theta, block) for rows [y, one-hot block]; with an auxiliary dimension, the importance
    estimate log q(y, e | theta) - log phi(e), e ~ N(0, 1), so that E_e[exp(.)] = q(y | theta)."""

    def eval(self):
        surrogate.eval()

    def log_prob(self, yb, context):
        y, oh = yb[:, :d], yb[:, d:]
        c = torch.cat([context, oh], 1) if B > 1 else context
        if not aux:
            return surrogate.log_prob(y, context=c)
        e = torch.randn(len(y), 1)
        return surrogate.log_prob(torch.cat([y, e], 1), context=c) + 0.5 * e[:, 0] ** 2 + 0.5 * math.log(2 * math.pi)


lik = BlockLikelihood()

# 2. source: maximise the marginal likelihood of the cells (NEB, MC-biased estimator)
obs = np.vstack([np.hstack([Yd[j], np.tile(np.eye(B)[k], (len(Yd[j]), 1)) if B > 1 else np.zeros((len(Yd[j]), 0))])
                 for k, j in enumerate(fit_blocks)])
obs = torch.tensor(obs[rng.permutation(len(obs))], dtype=torch.float32)  # deviation 5
n_val = int(0.1 * len(obs))
val, trn = obs[:n_val], obs[n_val:]
assert batch_size <= len(trn), "fewer cells than one batch"
source = Sampler(xdim=model.p, input_noise_dim=model.p, hidden_layer_dim=100, num_hidden_layers=3)
opt = torch.optim.Adam(source.parameters(), lr=1e-4, weight_decay=1e-5)
deadline = t0 + max(60 * minutes, t_sur + 60 * min_source_minutes)
log_s = math.log(n_mc)
train_hist, val_hist, steps, stopped = [], [], 0, "max_epochs"
best, wait = np.inf, 0
for epoch in range(n_epochs):
    source.train()
    losses = []
    for batch in create_dataloader(trn, batch_size):
        loss = -compute_estimator(batch, n_mc, source, lik)  # their estimator omits the constant -log S
        if torch.isfinite(loss):
            opt.zero_grad()
            loss.backward()
            opt.step()
        losses.append(float(loss) + log_s)
        steps += 1
        if time.time() > deadline:
            stopped = "time_budget"
            break
    train_hist.append(np.mean(losses))
    if stopped == "time_budget":
        print(f"time budget reached in epoch {epoch} after {steps} steps", flush=True)
        break
    source.eval()
    with torch.no_grad():
        vl = [float(-compute_estimator(b, n_mc, source, lik)) + log_s for b in create_dataloader(val, batch_size)]
    val_hist.append(np.mean(vl))
    print(f"epoch {epoch}: train -log p(y) {train_hist[-1]:.4f}, val {val_hist[-1]:.4f}, steps {steps}, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if epoch < min_epochs:
        continue
    if val_hist[-1] < best:
        best, wait = val_hist[-1], 0
    else:
        wait += 1
        if wait > patience:
            stopped = "early_stopping"
            print("early stop at epoch", epoch)
            break
source.eval()
with torch.no_grad():
    s = source.sample(int(cfg["settings"].get("n_pop", 200)) * 10).numpy()
pop = lo + (hi - lo) * (s + 1) / 2
seconds = time.time() - t0
np.savez_compressed(a.out, pop_draws=pop[None].astype(np.float32), seconds=seconds, t_surrogate=t_sur,
                    n_train_sims=n_train, n_cells=int(sum(len(v) for v in data.values())),
                    n_times=len(cols), steps=steps, epochs=len(train_hist), stopped=stopped,
                    n_cells_used=len(trn), n_mc=n_mc, train_nll=np.array(train_hist), val_nll=np.array(val_hist))
print("done", round(seconds / 60, 1), "min,", stopped)
