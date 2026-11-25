"""Vectorised Gillespie stochastic simulation algorithm (exact SSA).

All cells advance together, one reaction per cell per loop iteration; a cell
leaves the loop once its state has been recorded at every observation time.
"""
from __future__ import annotations

import numpy as np


def ssa(x0, stoich, propensity, theta, times, rng, t_start=None):
    """Exact SSA for n cells.

    x0          (n, S) initial copy numbers
    stoich      (R, S) state change of each reaction
    propensity  f(x, theta) -> (n, R) reaction rates for states x (n, S), parameters theta (n, p)
    times       increasing observation times (> 0)
    t_start     optional (n,) time at which each cell's reactions begin (state x0 until then)
    returns     (n, len(times), S) copy numbers at the observation times
    """
    times = np.asarray(times, float)
    n, S = x0.shape
    nt = len(times)
    stoich = np.asarray(stoich, np.int64)
    x = np.asarray(x0, np.int64).copy()
    t = np.zeros(n) if t_start is None else np.asarray(t_start, float).copy()
    nxt = np.zeros(n, int)
    out = np.empty((n, nt, S), np.int64)
    active = np.arange(n)
    while active.size:
        a = propensity(x[active], theta[active])
        a0 = a.sum(axis=1)
        with np.errstate(divide="ignore"):
            tn = t[active] + rng.exponential(size=active.size) / a0  # a0 = 0: no more reactions
        while True:  # record every observation time passed before the next reaction
            k = nxt[active]
            due = (k < nt) & (times[np.minimum(k, nt - 1)] < tn)
            if not due.any():
                break
            ids = active[due]
            out[ids, nxt[ids]] = x[ids]
            nxt[ids] += 1
        go = nxt[active] < nt
        if not go.any():
            break
        ag, ids = a[go], active[go]
        u = rng.uniform(size=ids.size) * a0[go]
        r = np.minimum((np.cumsum(ag, axis=1) < u[:, None]).sum(axis=1), ag.shape[1] - 1)
        x[ids] += stoich[r]
        t[ids] = tn[go]
        active = ids
    return out
