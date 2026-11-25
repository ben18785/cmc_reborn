"""Closed forms for the linear-Gaussian examples.

For q = A theta + e, e ~ N(0, S), current population pi_k = N(m, V) and a
Gaussian target N(mu_q, T) for q, one Richardson-Lucy (EM) step

    pi_{k+1}(theta) = pi_k(theta) * E_{q|theta}[ target(q) / p_k(q) ]
                    = integral of pi_k(theta | q) target(q) dq

is Gaussian: with gain G = V A' (A V A' + S)^{-1},

    m' = m + G (mu_q - A m),     V' = V - G A V + G T G'.

The first step from the prior is what CMC (with noise simulated, i.e. the
augmented-space update of Butler et al. 2020) gives; the fixed point is
any N(m, V) with A m = mu_q and A V A' + S = T.
"""
from __future__ import annotations

import numpy as np


def linear_rl_step(m, V, A, S, mu_q, T):
    A = np.atleast_2d(A)
    P = A @ V @ A.T + S
    G = V @ A.T @ np.linalg.inv(P)
    m_new = m + G @ (mu_q - A @ m)
    V_new = V - G @ A @ V + G @ T @ G.T
    return m_new, 0.5 * (V_new + V_new.T)


def scalar_rl_path(prior_mean, prior_var, sigma2, target_mean, target_var, n_iter):
    """Means and variances of the RL iterates pi_1..pi_n for q = theta + N(0, sigma2)."""
    m, v = float(prior_mean), float(prior_var)
    out = []
    for _ in range(n_iter):
        a = v / (v + sigma2)
        m = m + a * (target_mean - m)
        v = a * a * target_var + v * sigma2 / (v + sigma2)
        out.append((m, v))
    return np.array(out)


def scalar_fixed_point(sigma2, target_mean, target_var):
    """Exact deconvolution N(target_mean, target_var - sigma2) (needs target_var > sigma2)."""
    if target_var <= sigma2:
        raise ValueError("target narrower than the noise: no consistent solution")
    return target_mean, target_var - sigma2


def linear_block_targets(model):
    """Exact Gaussian target (mean, cov) of each block of a LinearGaussian model."""
    out = []
    S = model.sigma**2 * np.eye(model.d)
    for j in range(len(model.blocks)):
        A = model.A[j]
        out.append((A @ model.mu, A @ model.Sigma @ A.T + S))
    return out


def linear_rl_path(model, order, n_sweeps):
    """Exact block-cycling RL iterates for a LinearGaussian model from its prior.

    `order(sweep)` returns the block order for that sweep. Returns a list of
    (mean, cov) after each sweep."""
    m = model.prior_mean.copy()
    V = np.diag(model.prior_sd**2)
    S = model.sigma**2 * np.eye(model.d)
    targets = linear_block_targets(model)
    path = []
    for s in range(n_sweeps):
        for j in order(s):
            m, V = linear_rl_step(m, V, model.A[j], S, *targets[j])
        path.append((m.copy(), V.copy()))
    return path
