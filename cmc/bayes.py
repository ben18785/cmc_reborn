"""Fully Bayesian population inference: a hierarchical (mixed-effects) model.

Every observed cell i (measured in block j_i, outputs q_i) has its own parameters:

    pi                 ~ prior over populations
    theta_i | pi       ~ pi                     (restricted to the prior box, if any)
    q_i | theta_i      ~ K_{j_i}(. | theta_i)

The population is a mixture of T Gaussians with truncated stick-breaking weights
(a Dirichlet-process mixture, Ishwaran & James 2001); T = 1 is the standard
Gaussian (nonlinear) mixed-effects model. Base measure: component means
N(m0, S0) with m0, S0 the centre and spread of the CMC prior p, component
covariances inverse-Wishart. On a prior box, a cell's density is the Gaussian
times the box indicator; the truncation constant is ignored (negligible when
the population lies well inside the box).

Sampler: data augmentation (Tanner & Wong 1987), the Bayesian counterpart of the
EM / Richardson-Lucy iteration. Each sweep:
  * theta_i: Metropolis-Hastings, vectorised over cells: an independence
    proposal from the cell's component and two random walks shaped by the
    component covariance (scales adapted during burn-in);
  * allocations z_i, stick-breaking weights, (mu_k, Sigma_k): conjugate updates;
  * group moves that shift (mu_k) or linearly transform (Sigma_k) a component
    together with all its cells. These keep the sampler mixing when each cell
    carries little information (snapshots), where plain Gibbs crawls.

Per-cell likelihood:
  exact  model.log_lik(latent, q)
  nre    likelihood-free: per block, a neural ratio estimator (Hermans et al.
         2020) trained on simulated (theta, q) pairs against shuffled pairs.
         Its logit is log K_j(q|theta) - log m_j(q) + const, and m_j(q) is fixed
         for a cell, so it cancels in every Metropolis-Hastings ratio. The
         estimator is retrained on the current cell states (sequential NRE).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.linalg import expm, solve_triangular
from scipy.special import multigammaln
from scipy.stats import invwishart


@dataclass
class BayesSettings:
    lik: str = "exact"  # exact | nre
    n_comp: int = 1  # mixture components (1 = Gaussian mixed effects)
    alpha: float = 1.0  # DP concentration
    n_burn: int = 3000
    n_keep: int = 2000
    thin: int = 10
    n_pop: int = 1000  # population draws stored per saved iteration
    sigma_scale: float = 0.25  # prior guess of within-component sd, as a fraction of the prior sd
    nu_extra: float = 3.0  # inverse-Wishart dof = p + 1 + nu_extra
    n_init: int = 500  # prior particles used to initialise cells
    nre_train: int = 40000
    nre_rounds: int = 3
    nre_hidden: list = field(default_factory=lambda: [128, 128])
    seed: int = 0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- likelihoods
class _Lik:
    def __init__(self, model, data, fit_blocks):
        self.model, self.fit_blocks = model, list(fit_blocks)
        self.q = [np.asarray(data[j], float) for j in self.fit_blocks]
        self.blk = np.concatenate([np.full(len(d), b) for b, d in enumerate(self.q)])
        self.row = np.concatenate([np.arange(len(d)) for d in self.q])
        self.n = len(self.blk)

    def cols(self, b):
        return self.model.blocks[self.fit_blocks[b]]

    n_evals = 0  # forward-model evaluations (one per cell per proposed theta)

    def __call__(self, theta, cells):
        """log K(q_i | theta_i) for the given cell indices (theta row k belongs to cells[k])."""
        cells = np.asarray(cells)
        self.n_evals += len(cells)
        out = np.empty(len(cells))
        b_of = self.blk[cells]
        for b in np.unique(b_of):
            m = b_of == b
            out[m] = self.block_ll(b, theta[m], self.q[b][self.row[cells[m]]])
        return out

    def init_cells(self, particles, rng, chunk=250):
        """Pick, for each cell, one prior particle with probability proportional to its likelihood."""
        M = len(particles)
        idx = np.empty(self.n, int)
        for b in range(len(self.q)):
            cells = np.flatnonzero(self.blk == b)
            for s in range(0, len(cells), chunk):
                c = cells[s:s + chunk]
                th = np.tile(particles, (len(c), 1))
                q = np.repeat(self.q[b][self.row[c]], M, axis=0)
                ll = self.block_ll(b, th, q).reshape(len(c), M)
                self.n_evals += len(th)
                g = ll + rng.gumbel(size=ll.shape)
                idx[c] = np.argmax(g, axis=1)
        return particles[idx].copy()


class ExactLik(_Lik):
    def block_ll(self, b, theta, q):
        f = getattr(self.model, "block_log_lik", None)  # models whose likelihood is not latent-based
        if f is not None:
            return f(theta, self.cols(b), q)
        return self.model.log_lik(self.model.latent_cols(theta, self.cols(b)), q)


class NreLik(_Lik):
    def __init__(self, model, data, fit_blocks, hidden, n_train):
        super().__init__(model, data, fit_blocks)
        self.hidden, self.n_train = tuple(hidden), n_train
        self.nets = [None] * len(self.q)
        self.box = [None] * len(self.q)  # trust region: the range of the training parameters
        self.n_sim = 0  # simulations used to train the ratio estimators

    def _x(self, theta, q):
        q = np.log1p(q) if self.model.discrete else q
        return np.hstack([theta, q])

    def train(self, sample_theta, rng):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        for b in range(len(self.q)):
            th = sample_theta(self.n_train)
            q = self.model.simulate_cols(th, self.cols(b), rng)
            self.n_sim += len(th)
            X = np.vstack([self._x(th, q), self._x(th, q[rng.permutation(len(q))])])
            y = np.r_[np.ones(len(th)), np.zeros(len(th))]
            sc = StandardScaler().fit(X)
            net = MLPClassifier(hidden_layer_sizes=self.hidden, alpha=1e-4, early_stopping=True,
                                validation_fraction=0.1, n_iter_no_change=15, max_iter=400,
                                random_state=int(rng.integers(1 << 31))).fit(sc.transform(X), y)
            self.nets[b] = (sc, net)
            lo, hi = th.min(0), th.max(0)
            self.box[b] = (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

    def block_ll(self, b, theta, q):
        sc, net = self.nets[b]
        h = sc.transform(self._x(theta, q))
        for W, c in zip(net.coefs_[:-1], net.intercepts_[:-1]):
            h = np.maximum(h @ W + c, 0.0)  # relu
        out = (h @ net.coefs_[-1] + net.intercepts_[-1])[:, 0]  # unclipped logit
        # a relu network extrapolates linearly, so its logit can grow without bound away from the
        # training parameters; outside their range the estimate is not trusted
        lo, hi = self.box[b]
        out[~np.all((theta >= lo) & (theta <= hi), axis=1)] = -np.inf
        return out


# ---------------------------------------------------------------- helpers
def _mvn_logpdf(x, mu, L):
    z = solve_triangular(L, (x - mu).T, lower=True)
    return -0.5 * np.sum(z * z, axis=0) - np.sum(np.log(np.diag(L))) - 0.5 * len(mu) * np.log(2 * np.pi)


def _iw_logpdf(S, nu, Psi):
    p = len(S)
    _, ld_S = np.linalg.slogdet(S)
    _, ld_P = np.linalg.slogdet(Psi)
    return (0.5 * nu * ld_P - 0.5 * nu * p * np.log(2) - multigammaln(0.5 * nu, p)
            - 0.5 * (nu + p + 1) * ld_S - 0.5 * np.trace(Psi @ np.linalg.inv(S)))


def _chol(S):
    """Cholesky factor, symmetrising and adding jitter if rounding has made S indefinite."""
    S = 0.5 * (S + S.T)
    jit = 1e-12 * max(1.0, float(np.mean(np.abs(np.diag(S)))))
    for _ in range(8):
        try:
            return np.linalg.cholesky(S + jit * np.eye(len(S)))
        except np.linalg.LinAlgError:
            jit *= 100
    raise np.linalg.LinAlgError("covariance is not positive definite")


# ---------------------------------------------------------------- sampler
def fit_hierarchical(model, data, fit_blocks, s: BayesSettings, callback=None, n_checks=10, init_pop=None):
    """init_pop: optional draws from a population to start from (cells and components);
    default: the prior."""
    rng = np.random.default_rng(s.seed)
    t0 = time.time()
    p = model.p
    if model.lower is not None:
        lo, hi = model.lower, model.upper
        centre, scale = (lo + hi) / 2, (hi - lo) / np.sqrt(12)
        inbox = lambda th: np.all((th >= lo) & (th <= hi), axis=1)
    else:
        centre, scale = model.prior_mean, model.prior_sd
        inbox = lambda th: np.ones(len(th), bool)
    # where the model has an exact symmetry (e.g. two interchangeable rates), restrict every cell
    # to the canonical region so the per-cell posteriors are not multimodal by construction
    _box = inbox
    inbox = lambda th: _box(th) & np.all(model.canonical(th) == th, axis=1)
    prior_draw = lambda n: model.canonical(model.sample_prior(n, rng))
    m0, S0 = centre, np.diag(scale**2)
    S0inv = np.linalg.inv(S0)
    L0 = _chol(S0)
    nu0 = p + 1 + s.nu_extra
    Psi0 = (nu0 - p - 1) * np.diag((s.sigma_scale * scale) ** 2)
    T = s.n_comp

    if s.lik == "exact":
        lik = ExactLik(model, data, fit_blocks)
    elif s.lik == "nre":
        lik = NreLik(model, data, fit_blocks, s.nre_hidden, s.nre_train)
        lik.train(prior_draw, rng)
    else:
        raise ValueError(s.lik)
    N = lik.n
    allc = np.arange(N)

    # ---- initial state
    if init_pop is None:
        theta = lik.init_cells(prior_draw(s.n_init), rng)
        C0 = np.cov(theta.T) + 1e-6 * np.diag(scale**2)
        z = rng.integers(T, size=N) if T > 1 else np.zeros(N, int)
        mu = np.array([theta[rng.integers(N)] if T > 1 else theta.mean(0) for _ in range(T)])
    else:  # every component at the given population's moments; cells drawn given their data
        init_pop = np.asarray(init_pop, float)
        theta = lik.init_cells(init_pop[rng.integers(len(init_pop), size=s.n_init)], rng)
        C0 = np.cov(init_pop.T) + 1e-6 * np.diag(scale**2)
        z = rng.integers(T, size=N) if T > 1 else np.zeros(N, int)
        mu = np.array([init_pop.mean(0) for _ in range(T)])
    ll = lik(theta, allc)
    Sig = np.array([C0.copy() for _ in range(T)])
    w = np.full(T, 1.0 / T)

    log_s = dict(rw_big=np.log(0.5), rw_small=np.log(0.1), shift=np.log(0.1), lin=np.log(0.01))
    mu_hist = [[] for _ in range(T)]
    acc_log = {k: [] for k in ["indep", "rw_big", "rw_small", "shift", "lin"]}
    n_total = s.n_burn + s.n_keep
    retrain_at = set()
    if s.lik == "nre":
        retrain_at = {int(s.n_burn * r / s.nre_rounds) for r in range(1, s.nre_rounds)}
    check_at = set(np.linspace(0, n_total - 1, n_checks).astype(int))
    saved = dict(w=[], mu=[], Sigma=[], pop=[])
    history = []

    def lpop(th, zz):
        out = np.empty(len(th))
        for k in np.unique(zz):
            m = zz == k
            out[m] = _mvn_logpdf(th[m], mu[k], _chol(Sig[k]))
        return out

    def mh_cells(prop, log_extra, key):
        nonlocal theta, ll
        ok = inbox(prop)
        llp = np.full(N, -np.inf)
        if ok.any():
            llp[ok] = lik(prop[ok], allc[ok])
        la = llp - ll + log_extra
        acc = ok & np.isfinite(la) & (np.log(rng.uniform(size=N)) < la)
        theta[acc], ll[acc] = prop[acc], llp[acc]
        acc_log[key].append(acc.mean())
        return acc.mean()

    def sample_pop(n):
        out = np.empty((0, p))
        while len(out) < n:
            k = rng.choice(T, size=2 * n, p=w)
            x = np.empty((2 * n, p))
            for c in np.unique(k):
                m = k == c
                x[m] = mu[c] + rng.standard_normal((m.sum(), p)) @ _chol(Sig[c]).T
            out = np.vstack([out, x[inbox(x)]])
        return out[:n]

    for it in range(n_total):
        burn = it < s.n_burn
        if it in retrain_at:  # sequential NRE: retrain near the current cells
            cur = theta.copy()
            def near(n):
                k = int(0.8 * n)
                x = cur[rng.integers(N, size=k)] + 0.1 * scale * rng.standard_normal((k, p))
                x = model.canonical(model.project_to_support(x))
                return np.vstack([x, prior_draw(n - k)])
            lik.train(near, rng)
            ll = lik(theta, allc)

        Ls = [_chol(Sig[k]) for k in range(T)]
        eps = rng.standard_normal((N, p))
        step = np.empty((N, p))
        for k in range(T):
            m = z == k
            step[m] = eps[m] @ Ls[k].T

        # (1) cells: independence proposal from the component (prior cancels with proposal)
        mh_cells(mu[z] + step, 0.0, "indep")
        # (2, 3) random walks shaped by the component covariance
        for key in ("rw_big", "rw_small"):
            e2 = rng.standard_normal((N, p))
            st = np.empty((N, p))
            for k in range(T):
                m = z == k
                st[m] = e2[m] @ Ls[k].T
            prop = theta + np.exp(log_s[key]) * st
            a = mh_cells(prop, lpop(prop, z) - lpop(theta, z), key)
            if burn:
                log_s[key] = min(log_s[key] + 0.05 * (a - 0.25), np.log(3.0))

        # (4) allocations and weights
        if T > 1:
            lp = np.stack([np.log(w[k] + 1e-300) + _mvn_logpdf(theta, mu[k], Ls[k]) for k in range(T)], 1)
            z = np.argmax(lp + rng.gumbel(size=lp.shape), axis=1)
            n_k = np.bincount(z, minlength=T)
            tail = np.r_[np.cumsum(n_k[::-1])[::-1][1:], 0]
            v = rng.beta(1 + n_k, s.alpha + tail)
            v[-1] = 1.0
            w = v * np.r_[1.0, np.cumprod(1 - v[:-1])]
        n_k = np.bincount(z, minlength=T)

        # (5) conjugate updates of each component
        for k in range(T):
            x = theta[z == k]
            Sinv = np.linalg.inv(Sig[k])
            P = S0inv + n_k[k] * Sinv
            Pc = np.linalg.inv(P)
            mean = Pc @ (S0inv @ m0 + Sinv @ x.sum(0))
            mu[k] = mean + _chol(Pc) @ rng.standard_normal(p)
            R = (x - mu[k]).T @ (x - mu[k])
            B = Psi0 + 0.5 * (R + R.T)
            if not np.isfinite(B).all() or not np.isfinite(mu[k]).all():
                raise FloatingPointError(
                    f"non-finite state in component {k} (n_k={n_k[k]}, it={it}): theta finite "
                    f"{np.isfinite(x).all()}, mean finite {np.isfinite(mean).all()}, Sigma finite "
                    f"{np.isfinite(Sig[k]).all()}, cond(Sigma) {np.linalg.cond(Sig[k]):.3g}, log_s "
                    f"{ {kk: round(float(v), 2) for kk, v in log_s.items()} }")
            Sig[k] = invwishart.rvs(df=nu0 + n_k[k], scale=_chol(B) @ _chol(B).T, random_state=rng)

        # (6) group moves: a component and all its cells together
        for k in range(T):
            cells = np.flatnonzero(z == k)
            if len(cells) == 0:
                continue
            x, llx = theta[cells], ll[cells]
            # shift: theta_i + d and mu + d (population density unchanged)
            H = np.array(mu_hist[k][-500:])
            C = np.cov(H.T) if len(H) > 50 else np.diag((0.01 * scale) ** 2)
            d = np.exp(log_s["shift"]) * (_chol(C + 1e-8 * np.diag(scale**2)) @ rng.standard_normal(p))
            xp = x + d
            if np.isfinite(xp).all() and inbox(xp).all():
                llp = lik(xp, cells)
                la = llp.sum() - llx.sum() + _mvn_logpdf((mu[k] + d)[None], m0, L0)[0] - _mvn_logpdf(mu[k][None], m0, L0)[0]
                acc = bool(np.isfinite(la) and np.log(rng.uniform()) < la)
            else:
                acc = False
            if acc:
                theta[cells], ll[cells], mu[k] = xp, llp, mu[k] + d
                x, llx = xp, llp
            acc_log["shift"].append(float(acc))
            if burn:
                log_s["shift"] = min(log_s["shift"] + 0.05 * (float(acc) - 0.25), np.log(3.0))
            # linear map: theta_i -> mu + A (theta_i - mu), Sigma -> A Sigma A'
            M = rng.standard_normal((p, p))
            e = np.exp(log_s["lin"])
            A = expm(e * M)
            xp = mu[k] + (x - mu[k]) @ A.T
            Sp = A @ Sig[k] @ A.T
            if (np.isfinite(xp).all() and np.isfinite(Sp).all() and inbox(xp).all()
                    and np.linalg.cond(Sp) < 1e8 and np.all(np.linalg.eigvalsh(0.5 * (Sp + Sp.T)) > 0)):
                llp = lik(xp, cells)
                la = (llp.sum() - llx.sum() + _iw_logpdf(Sp, nu0, Psi0) - _iw_logpdf(Sig[k], nu0, Psi0)
                      + (p + 1) * e * np.trace(M))
                acc = bool(np.isfinite(la) and np.log(rng.uniform()) < la)
            else:
                acc = False
            if acc:
                theta[cells], ll[cells], Sig[k] = xp, llp, Sp
            acc_log["lin"].append(float(acc))
            if burn:
                log_s["lin"] = min(log_s["lin"] + 0.05 * (float(acc) - 0.25), np.log(0.3))
            if burn:  # the shift proposal adapts during burn-in only
                mu_hist[k].append(mu[k].copy())

        if not burn and (it - s.n_burn) % s.thin == 0:
            saved["w"].append(w.copy())
            saved["mu"].append(mu.copy())
            saved["Sigma"].append(Sig.copy())
            saved["pop"].append(sample_pop(s.n_pop).astype(np.float32))
        if it in check_at:
            rec = dict(sweep=it + 1, seconds=time.time() - t0, phase="burn" if burn else "keep",
                       accept={k: float(np.mean(v[-100:])) if v else None for k, v in acc_log.items()},
                       n_active=int((n_k > 0).sum()), mean_ll=float(ll.mean()), n_solves=int(lik.n_evals))
            if callback is not None:
                pop = sample_pop(2000)
                rec.update(callback(pop, np.full(len(pop), -np.log(len(pop))), rng))
            history.append(rec)

    return dict(pop_draws=np.array(saved["pop"]), w=np.array(saved["w"]), mu=np.array(saved["mu"]),
                Sigma=np.array(saved["Sigma"]), history=history, seconds=time.time() - t0, n_cells=N,
                n_solves=int(lik.n_evals), n_solves_train=int(getattr(lik, "n_sim", 0)))
