"""Density-ratio estimators r(q) = target(q) / pushforward(q).

All estimators are fitted to two samples: `data` (observed cells of one block,
the target) and `sim` (unweighted simulated outputs of the current population,
already resampled according to the particle weights). `log_ratio(q)` evaluates
log r at new points.

KdeRatio        two Gaussian KDEs (the original CMC choice, per block)
ClassifierRatio probabilistic classifier, data = 1 vs simulations = 0:
                log r = logit c(q) + log(n_sim / n_data)
VineJointRatio  baseline for the joint one-step CMC: target = product of block
                KDEs (independence across snapshots), pushforward = vine copula
                KDE over all fitted output dimensions (kdevine equivalent).

Counts (model.discrete) are dequantised for KDEs: q -> log1p(q + U(0,1)).
Adding the same U(0,1) jitter to both samples leaves the ratio of the pmfs
unchanged, and log1p is a bijection, so the ratio is preserved. Classifiers
use log1p(q) directly.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def _features(q, discrete, rng=None, dequantise=False):
    q = np.asarray(q, float)
    if not discrete:
        return q
    if dequantise:
        q = q + rng.uniform(size=q.shape)
    return np.log1p(q)


class KdeRatio:
    def __init__(self, discrete=False, rng=None, bw=None):
        self.discrete = discrete
        self.rng = rng or np.random.default_rng()
        self.bw = bw
        self._target = None

    def fit_target(self, data):
        x = _features(data, self.discrete, self.rng, dequantise=True)
        self._target = gaussian_kde(x.T, bw_method=self.bw)
        return self

    def fit(self, data, sim):
        if self._target is None:
            self.fit_target(data)
        x = _features(sim, self.discrete, self.rng, dequantise=True)
        self._push = gaussian_kde(x.T, bw_method=self.bw)
        return self

    def log_ratio(self, q):
        x = _features(q, self.discrete, self.rng, dequantise=True)
        return self._target.logpdf(x.T) - self._push.logpdf(x.T)


class ClassifierRatio:
    """kind: 'logistic2' (logistic regression on quadratic features; exact for
    Gaussian blocks) or 'mlp' (two hidden layers, early stopping)."""

    def __init__(self, kind="mlp", discrete=False, seed=0, hidden=(64, 64), alpha=1e-3):
        self.kind, self.discrete, self.seed = kind, discrete, seed
        self.hidden, self.alpha = hidden, alpha

    def _make(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        if self.kind == "logistic2":
            return make_pipeline(
                StandardScaler(),
                PolynomialFeatures(2, include_bias=False),
                StandardScaler(),
                LogisticRegression(C=100.0, max_iter=2000),
            )
        if self.kind == "mlp":
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=self.hidden,
                    alpha=self.alpha,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=10,
                    max_iter=300,
                    random_state=self.seed,
                ),
            )
        raise ValueError(self.kind)

    def fit(self, data, sim, weight=None):
        """`weight`: optional per-cell weights on the data (mean 1), e.g. Bayesian bootstrap."""
        x1 = _features(data, self.discrete)
        x0 = _features(sim, self.discrete)
        X = np.vstack([x1, x0])
        y = np.r_[np.ones(len(x1)), np.zeros(len(x0))]
        clf = self._make()
        kw = {}
        if weight is not None:
            if self.kind != "logistic2":
                raise ValueError("data weights are supported for logistic2 only")
            w = np.asarray(weight, float)
            kw[clf.steps[-1][0] + "__sample_weight"] = np.r_[w * len(w) / w.sum(), np.ones(len(x0))]
        self.clf = clf.fit(X, y, **kw)
        self.offset = np.log(len(x0) / len(x1))
        return self

    def log_ratio(self, q):
        x = _features(q, self.discrete)
        p = self.clf.predict_proba(x)[:, 1]
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p) - np.log1p(-p) + self.offset


class VineJointRatio:
    """Original CMC baseline over all fitted blocks at once (one step)."""

    def __init__(self, blocks, discrete=False, rng=None, trunc_lvl=None, threads=1):
        self.blocks, self.discrete = blocks, discrete
        self.rng = rng or np.random.default_rng()
        self.trunc_lvl, self.threads = trunc_lvl, threads

    def fit(self, data_blocks, sim):
        import pyvinecopulib as pv

        self.targets = [KdeRatio(self.discrete, self.rng).fit_target(d)._target for d in data_blocks]
        x = _features(sim, self.discrete, self.rng, dequantise=True)
        kw = dict(family_set=[pv.BicopFamily.tll], num_threads=self.threads)
        if self.trunc_lvl is not None:
            kw["trunc_lvl"] = int(self.trunc_lvl)
        self.push = pv.Vinedist.from_data(x, controls=pv.FitControlsVinecop(**kw))
        return self

    def log_ratio(self, q):
        x = _features(q, self.discrete, self.rng, dequantise=True)
        lt = np.zeros(len(q))
        start = 0
        for kde, b in zip(self.targets, self.blocks):
            lt += kde.logpdf(x[:, start:start + len(b)].T)
            start += len(b)
        lp = np.asarray(self.push.logpdf(x))
        return lt - lp


def make_ratio(kind, discrete, rng, seed=0):
    if kind == "kde":
        return KdeRatio(discrete=discrete, rng=rng)
    if kind in ("logistic2", "mlp"):
        return ClassifierRatio(kind=kind, discrete=discrete, seed=seed)
    raise ValueError(kind)
