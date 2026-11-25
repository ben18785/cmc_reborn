"""Discrepancies used to score fits."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .infer import systematic_resample


def energy_distance(x, y, scale=None, n_max=1500, rng=None):
    """Energy distance between samples x and y (rows), each coordinate divided by `scale`
    (default: pooled sd). Subsamples to n_max rows for cost."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) > n_max:
        x = x[rng.choice(len(x), n_max, replace=False)]
    if len(y) > n_max:
        y = y[rng.choice(len(y), n_max, replace=False)]
    if scale is None:
        scale = np.vstack([x, y]).std(axis=0)
    scale = np.where(np.asarray(scale) > 0, scale, 1.0)
    x = x / scale
    y = y / scale
    return float(2 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean())


def block_discrepancies(model, theta, logw, data, blocks, rng, n=1500):
    """Energy distance between each observed block and the fitted population's
    predicted block (fresh noisy simulation), on log1p scale for counts."""
    idx = systematic_resample(logw, n, rng)
    q = model.simulate(theta[idx], rng, noise=True)
    out = []
    for j in blocks:
        a, b = data[j], q[:, model.blocks[j]]
        if model.discrete:
            a, b = np.log1p(a), np.log1p(b)
        out.append(energy_distance(a, b, rng=rng))
    return out


def resample(theta, logw, n, rng):
    return theta[systematic_resample(logw, n, rng)]
