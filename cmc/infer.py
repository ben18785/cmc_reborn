"""Weighted-particle samplers.

iterate_blocks
    Iterated CMC: block-wise Richardson-Lucy / iterative proportional fitting.
    Particles theta_i (from the prior) carry log weights. Each update picks one
    snapshot block j and

      1. draws n_fit particles by weight, simulates their block-j outputs and
         fits a ratio estimator r_j = target_j / pushforward_j;
      2. simulates `n_reps` fresh noisy outputs for every particle and
         multiplies its weight by the average ratio (1/M) sum_m r_j(q_im),
         an unbiased estimate of E[r_j(q) | theta_i] under the noise;
      3. if the effective sample size drops below ess_frac * N, resamples and
         jitters (Liu-West shrinkage kernel, which keeps mean and covariance).

    The training sample (step 1) and the evaluation sample (step 2) are
    independent simulations. Blocks are visited in a random order each sweep.
    With a deterministic map and one block, one update is exactly CMC.

one_step_joint
    The original CMC (or, with noise=True, the augmented update of Butler et
    al. 2020): one importance reweighting of prior draws by
    prod_j target_j(q_j) / pushforward(q) over all fitted blocks jointly.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.special import logsumexp

from .ratio import VineJointRatio, make_ratio


@dataclass
class Settings:
    ratio: str = "kde"  # kde | logistic2 | mlp
    n_particles: int = 20000
    n_reps: int = 4
    n_fit: int = 4000
    n_sweeps: int = 20
    ess_frac: float = 0.5
    shrink: float = 0.95  # Liu-West a
    clip: float = 8.0  # |log r| cap per update
    step: float = 1.0  # damping exponent on the multiplier
    noise: bool = True
    seed: int = 0
    trunc_lvl: int | None = None  # vine baseline only
    threads: int = 1

    def to_dict(self):
        return asdict(self)


def normalise(logw):
    return logw - logsumexp(logw)


def ess(logw):
    w = np.exp(normalise(logw))
    return 1.0 / np.sum(w * w)


def weighted_moments(theta, logw):
    w = np.exp(normalise(logw))
    m = w @ theta
    c = (theta - m).T @ ((theta - m) * w[:, None])
    return m, c


def systematic_resample(logw, n, rng):
    w = np.exp(normalise(logw))
    u = (rng.uniform() + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(w), u)
    return np.minimum(idx, len(w) - 1)


def rejuvenate(model, theta, logw, shrink, rng):
    m, C = weighted_moments(theta, logw)
    idx = systematic_resample(logw, len(theta), rng)
    t = theta[idx]
    Lc = np.linalg.cholesky(C + 1e-10 * np.eye(len(m)))
    z = rng.standard_normal(t.shape) @ Lc.T
    t = shrink * t + (1 - shrink) * m + np.sqrt(1 - shrink**2) * z
    return model.project_to_support(t)


def iterate_blocks(model, data, fit_blocks, s: Settings, callback=None, data_weights=None, init_theta=None):
    """data_weights: optional per-block arrays of per-cell weights (Bayesian bootstrap).
    init_theta: optional starting particles (default: draws from the prior)."""
    rng = np.random.default_rng(s.seed)
    N = s.n_particles
    if init_theta is None:
        theta = model.sample_prior(N, rng)
    else:
        theta = np.asarray(init_theta, float)[rng.integers(len(init_theta), size=N)]
    lat = model.latent(theta)
    # forward-model evaluations: one per particle created (deterministic latent), or one per
    # simulated output (stochastic simulators); scoring not counted
    n_solves = 0 if model.stochastic_sim else N
    logw = np.full(N, -np.log(N))
    targets = {}
    history = []
    n_rejuv = 0
    t0 = time.time()
    for sweep in range(s.n_sweeps):
        for j in rng.permutation(fit_blocks):
            cols = model.blocks[j]
            est = targets.get(j)
            if est is None or s.ratio != "kde":
                est = make_ratio(s.ratio, model.discrete, rng, seed=int(rng.integers(1 << 31)))
                targets[j] = est
            idx = systematic_resample(logw, s.n_fit, rng)
            q_train = model.observe_cols(lat[idx], cols, rng) if s.noise else lat[idx][:, cols]
            if data_weights is None:
                est.fit(data[j], q_train)
            else:
                est.fit(data[j], q_train, weight=data_weights[j])
            reps = 1 if not s.noise else s.n_reps
            lat_rep = np.repeat(lat, reps, axis=0)
            q_eval = model.observe_cols(lat_rep, cols, rng) if s.noise else lat_rep[:, cols]
            if s.noise and model.stochastic_sim:  # every noise draw is a full simulation
                n_solves += len(idx) + len(lat_rep)
            lr = np.clip(est.log_ratio(q_eval), -s.clip, s.clip).reshape(N, reps)
            mult = logsumexp(lr, axis=1) - np.log(reps)
            logw = normalise(logw + s.step * mult)
            if ess(logw) < s.ess_frac * N:
                theta = rejuvenate(model, theta, logw, s.shrink, rng)
                lat = model.latent(theta)
                n_solves += 0 if model.stochastic_sim else N
                logw = np.full(N, -np.log(N))
                n_rejuv += 1
        m, C = weighted_moments(theta, logw)
        rec = dict(sweep=sweep + 1, mean=m.tolist(), sd=np.sqrt(np.diag(C)).tolist(),
                   ess=float(ess(logw)), n_rejuv=n_rejuv, n_solves=n_solves, seconds=time.time() - t0)
        if callback is not None:
            rec.update(callback(theta, logw, rng))
        history.append(rec)
    return dict(theta=theta, logw=logw, history=history)


def one_step_joint(model, data, fit_blocks, s: Settings):
    rng = np.random.default_rng(s.seed)
    N = s.n_particles
    theta = model.sample_prior(N, rng)
    lat = model.latent(theta)
    cols = np.concatenate([model.blocks[j] for j in fit_blocks])
    q = model.observe_cols(lat, cols, rng) if s.noise else lat[:, cols]
    est = VineJointRatio([model.blocks[j] for j in fit_blocks], model.discrete, rng,
                         trunc_lvl=s.trunc_lvl, threads=s.threads)
    t0 = time.time()
    sub = rng.choice(N, size=min(s.n_fit, N), replace=False)
    est.fit([data[j] for j in fit_blocks], q[sub])
    logw = normalise(est.log_ratio(q))
    m, C = weighted_moments(theta, logw)
    history = [dict(sweep=1, mean=m.tolist(), sd=np.sqrt(np.diag(C)).tolist(),
                    ess=float(ess(logw)), n_rejuv=0, seconds=time.time() - t0)]
    return dict(theta=theta, logw=logw, history=history)
