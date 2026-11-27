"""Forward models for population inference from snapshot data.

Every model works on an unconstrained parameter vector theta (log scale where
the physical parameter is positive). A model splits the forward map into

    latent(theta)            deterministic part, shape (n, n_out), cached by the samplers
    observe(latent, rng)     stochastic part (measurement or intrinsic noise)

so `observe(latent(theta), rng)` is one draw from the stochastic forward map
K(q | theta), and `latent(theta)` alone is the noise-free map used by the
original CMC. For a model whose noise cannot be split off like this (e.g. a
Gillespie simulation), latent can return theta itself and observe can run the
simulation.

Outputs are grouped into blocks: one block per snapshot (time point, dose),
holding the quantities measured on the same cells. Different blocks come
from different cells, so only per-block distributions are observed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ssa import ssa


@dataclass
class Model:
    name: str
    param_names: list[str]
    lower: np.ndarray  # prior box (uniform prior on theta) or None for Gaussian prior
    upper: np.ndarray
    blocks: list[np.ndarray]  # output column indices of each block
    block_labels: list[str]
    prior_mean: np.ndarray | None = None  # Gaussian prior (used if lower is None)
    prior_sd: np.ndarray | None = None
    discrete: bool = False  # outputs are counts
    meta: dict = field(default_factory=dict)

    @property
    def p(self) -> int:
        return len(self.param_names)

    @property
    def n_out(self) -> int:
        return int(sum(len(b) for b in self.blocks))

    # ---- prior
    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.lower is not None:
            return rng.uniform(self.lower, self.upper, size=(n, self.p))
        return self.prior_mean + self.prior_sd * rng.standard_normal((n, self.p))

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        if self.lower is not None:
            inside = np.all((theta >= self.lower) & (theta <= self.upper), axis=1)
            return np.where(inside, 0.0, -np.inf)
        z = (theta - self.prior_mean) / self.prior_sd
        return -0.5 * np.sum(z * z, axis=1)

    def project_to_support(self, theta: np.ndarray) -> np.ndarray:
        """Reflect points back into the prior box (used after jittering)."""
        if self.lower is None:
            return theta
        lo, hi = self.lower, self.upper
        width = hi - lo
        t = (theta - lo) % (2 * width)
        t = np.where(t > width, 2 * width - t, t)
        return lo + t

    # ---- forward map (overridden)
    def latent(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def observe(self, latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def latent_cols(self, theta: np.ndarray, cols) -> np.ndarray:
        """latent(theta)[:, cols]; models may override to compute only those outputs."""
        return self.latent(theta)[:, cols]

    # True when observe() runs a full stochastic simulation (e.g. Gillespie), so that every
    # noise draw costs a simulation; False when it only adds noise to the cached latent.
    stochastic_sim = False

    def observe_cols(self, latent: np.ndarray, cols, rng) -> np.ndarray:
        """Noisy outputs `cols` given the full cached latent(theta)."""
        return self.observe(latent[:, cols], rng)

    def simulate_cols(self, theta: np.ndarray, cols, rng) -> np.ndarray:
        """One draw of the outputs `cols` for each row of theta."""
        return self.observe(self.latent_cols(theta, cols), rng)

    def log_lik(self, latent: np.ndarray, q: np.ndarray) -> np.ndarray:
        """log K(q | theta) given latent(theta), summed over the last axis (broadcasts).
        Only needed by the exact-likelihood Bayesian sampler (cmc.bayes)."""
        raise NotImplementedError

    def simulate(self, theta, rng, noise=True):
        lat = self.latent(theta)
        return self.observe(lat, rng) if noise else lat

    def canonical(self, theta: np.ndarray) -> np.ndarray:
        """Map parameters to a canonical representative where the model has an exact symmetry."""
        return theta

    # ---- data-generating population (overridden)
    def sample_population(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------
class ScalarGaussian(Model):
    """q = theta + N(0, sigma^2); population N(mu, s^2); prior N(0, prior_var).

    Everything is Gaussian, so the Richardson-Lucy iterates have a closed form
    (cmc.gaussian.scalar_rl_path)."""

    def __init__(self, sigma=1.0, pop_mean=1.0, pop_sd=1.0, prior_var=10.0):
        super().__init__(
            name="scalar",
            param_names=["theta"],
            lower=None,
            upper=None,
            prior_mean=np.zeros(1),
            prior_sd=np.sqrt([prior_var]),
            blocks=[np.array([0])],
            block_labels=["q"],
            meta=dict(sigma=sigma, pop_mean=pop_mean, pop_sd=pop_sd, prior_var=prior_var),
        )
        self.sigma, self.pop_mean, self.pop_sd = sigma, pop_mean, pop_sd

    def latent(self, theta):
        return theta[:, :1].copy()

    def observe(self, latent, rng):
        return latent + self.sigma * rng.standard_normal(latent.shape)

    def log_lik(self, latent, q):
        s = self.sigma
        return np.sum(-0.5 * ((q - latent) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi), axis=-1)

    def sample_population(self, n, rng):
        return self.pop_mean + self.pop_sd * rng.standard_normal((n, 1))


# --------------------------------------------------------------------------
class LinearGaussian(Model):
    """q_j = A_j theta + N(0, sigma^2 I) for J blocks of dimension d.

    The population is N(mu, Sigma) with correlations; with J*d(d+1)/2 >= p(p+1)/2
    the population is identified, so recovery can be checked against the truth.
    The Gaussian prior keeps Richardson-Lucy iterates Gaussian
    (cmc.gaussian.linear_rl_step)."""

    def __init__(self, p=5, n_blocks=24, d=2, sigma=0.5, prior_sd=3.0, seed=11):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n_blocks, d, p)) / np.sqrt(p)
        mu = rng.normal(0, 1, p)
        sds = rng.uniform(0.4, 1.0, p)
        L = rng.standard_normal((p, p))
        C = L @ L.T + p * np.eye(p)
        C = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))  # correlation matrix
        Sigma = C * np.outer(sds, sds)
        blocks = [np.arange(j * d, (j + 1) * d) for j in range(n_blocks)]
        super().__init__(
            name="linear",
            param_names=[f"theta{i+1}" for i in range(p)],
            lower=None,
            upper=None,
            prior_mean=np.zeros(p),
            prior_sd=np.full(p, prior_sd),
            blocks=blocks,
            block_labels=[f"snapshot{j+1}" for j in range(n_blocks)],
            meta=dict(sigma=sigma),
        )
        self.A, self.mu, self.Sigma, self.sigma, self.d = A, mu, Sigma, sigma, d

    def latent(self, theta):
        # (n, J, d) -> (n, J*d)
        return np.einsum("jdp,np->njd", self.A, theta).reshape(theta.shape[0], -1)

    def observe(self, latent, rng):
        return latent + self.sigma * rng.standard_normal(latent.shape)

    def log_lik(self, latent, q):
        s = self.sigma
        return np.sum(-0.5 * ((q - latent) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi), axis=-1)

    def sample_population(self, n, rng):
        return rng.multivariate_normal(self.mu, self.Sigma, size=n)


# --------------------------------------------------------------------------
def _linear2_solution(a, b, c, d, s, t):
    """x(t) for x' = M x + (s, 0), x(0) = 0, M = [[-a, b], [c, -d]], vectorised.

    Uses e^{Mt} = [(l1 e^{l2 t} - l2 e^{l1 t}) I + (e^{l1 t} - e^{l2 t}) M] / (l1 - l2)
    and x(t) = (e^{Mt} - I) M^{-1} (s, 0)."""
    tr = -(a + d)
    det = a * d - b * c
    disc = np.sqrt((a - d) ** 2 + 4 * b * c)
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    # M^{-1} (s, 0) = (1/det) [[-d, -b], [-c, -a]] (s, 0)
    y0 = -d * s / det
    y1 = -c * s / det
    e1 = np.exp(l1 * t)
    e2 = np.exp(l2 * t)
    f0 = (l1 * e2 - l2 * e1) / (l1 - l2)
    f1 = (e1 - e2) / (l1 - l2)
    # (e^{Mt} - I) y = (f0 - 1) y + f1 M y
    My0 = -a * y0 + b * y1
    My1 = c * y0 - d * y1
    return (f0 - 1) * y0 + f1 * My0, (f0 - 1) * y1 + f1 * My1


class GrowthFactor(Model):
    """Growth factor receptor model (Dixit et al. 2018; Lambert et al. 2021, eqs 14-15):

        dR/dt = RT kdeg - k1 L R + km1 P - kdeg R
        dP/dt = k1 L R - km1 P - kdegs P,      R(0) = P(0) = 0.

    Outputs: (log R, log P) at each (dose L, time t), with independent log-normal
    measurement noise of sd `noise_sd` on every measurement. Population: RT and k1
    vary log-normally across cells, the other three rates are shared (as in the paper).

    `design` sets which outputs are measured on the same cells (one block each):
      "snapshot": each (dose, time) on its own cells (destructive measurement);
      "pairs":    consecutive times at a dose tracked on the same cells (overlapping);
      "tracked":  every time at a dose tracked on the same cells (live-cell imaging)."""

    names = ["log_RT", "log_k1", "log_km1", "log_kdeg", "log_kdegs"]

    def __init__(self, doses=(2.0, 10.0), times=(2, 5, 10, 20, 40, 80), noise_sd=0.1,
                 pop_sd=(0.15, 0.2), design="snapshot"):
        lower = np.log([2e5, 0.1, 1.0, 0.003, 0.05])
        upper = np.log([2e6, 10.0, 50.0, 0.1, 1.0])
        conds = [(L, t) for L in doses for t in times]
        cols = lambda k: [2 * k, 2 * k + 1]  # (log R, log P) of condition k
        nt = len(times)
        if design == "snapshot":
            groups = [[k] for k in range(len(conds))]
        elif design == "pairs":
            groups = [[d * nt + i, d * nt + i + 1] for d in range(len(doses)) for i in range(nt - 1)]
        elif design == "tracked":
            groups = [[d * nt + i for i in range(nt)] for d in range(len(doses))]
        else:
            raise ValueError(f"unknown design {design!r}")
        blocks = [np.array([c for k in g for c in cols(k)]) for g in groups]
        labels = [f"L={conds[g[0]][0]:g},t=" + "+".join(f"{conds[k][1]:g}" for k in g) for g in groups]
        super().__init__(
            name="growth",
            param_names=list(self.names),
            lower=lower,
            upper=upper,
            blocks=blocks,
            block_labels=labels,
            meta=dict(noise_sd=noise_sd, doses=list(doses), times=list(times), design=design),
        )
        self.conds = conds
        self.noise_sd = noise_sd
        self.pop_mean = np.log([6.5e5, 1.7, 8.0, 0.015, 0.25])
        self.pop_sd = np.array([pop_sd[0], pop_sd[1], 0.0, 0.0, 0.0])

    def latent(self, theta):
        RT, k1, km1, kdeg, kdegs = np.exp(theta).T
        out = np.empty((theta.shape[0], 2 * len(self.conds)))
        for k, (L, t) in enumerate(self.conds):
            R, P = _linear2_solution(k1 * L + kdeg, km1, k1 * L, km1 + kdegs, RT * kdeg, t)
            out[:, 2 * k] = np.log(np.maximum(R, 1e-12))
            out[:, 2 * k + 1] = np.log(np.maximum(P, 1e-12))
        return out

    def observe(self, latent, rng):
        return latent + self.noise_sd * rng.standard_normal(latent.shape)

    def log_lik(self, latent, q):
        s = self.noise_sd
        return np.sum(-0.5 * ((q - latent) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi), axis=-1)

    def latent_cols(self, theta, cols):
        cols = np.asarray(cols)
        RT, k1, km1, kdeg, kdegs = np.exp(theta).T
        full = {}
        for k in sorted({int(c) // 2 for c in cols}):
            L, t = self.conds[k]
            R, P = _linear2_solution(k1 * L + kdeg, km1, k1 * L, km1 + kdegs, RT * kdeg, t)
            full[2 * k] = np.log(np.maximum(R, 1e-12))
            full[2 * k + 1] = np.log(np.maximum(P, 1e-12))
        return np.stack([full[int(c)] for c in cols], axis=1)

    def sample_population(self, n, rng):
        return self.pop_mean + self.pop_sd * rng.standard_normal((n, self.p))


# --------------------------------------------------------------------------
class GeneExpression(Model):
    """Induced transcription of three genes with cell-to-cell rate variation.

    mRNA of gene g follows a birth-death process from zero at induction, so
    X_g(t) ~ Poisson(k_g / gamma_g (1 - exp(-gamma_g t))) exactly (intrinsic
    noise). Capture is binomial with efficiency `capture` (as in scRNA-seq), so
    the measured count is Poisson(capture * lambda). Each snapshot measures all
    three genes on the same cells at one time. Transcription rates are
    correlated across genes within a cell (shared extrinsic factor)."""

    def __init__(self, times=(0.5, 1, 2, 3, 4, 6, 8, 12), capture=0.3, corr=0.6):
        lower = np.log([1.0, 1.0, 1.0, 0.05, 0.05, 0.05])
        upper = np.log([300.0, 300.0, 300.0, 2.0, 2.0, 2.0])
        blocks = [np.array([3 * k, 3 * k + 1, 3 * k + 2]) for k in range(len(times))]
        super().__init__(
            name="gene",
            param_names=["log_k1", "log_k2", "log_k3", "log_g1", "log_g2", "log_g3"],
            lower=lower,
            upper=upper,
            blocks=blocks,
            block_labels=[f"t={t:g}h" for t in times],
            discrete=True,
            meta=dict(capture=capture, times=list(times), corr=corr),
        )
        self.times = np.asarray(times, float)
        self.capture = capture
        self.pop_mean = np.log([40.0, 20.0, 80.0, 0.3, 0.5, 0.2])
        sds = np.array([0.3, 0.3, 0.3, 0.2, 0.2, 0.2])
        C = np.eye(6)
        C[:3, :3] = corr + (1 - corr) * np.eye(3)
        self.pop_cov = C * np.outer(sds, sds)

    def latent(self, theta):
        k = np.exp(theta[:, :3])
        g = np.exp(theta[:, 3:])
        lam = (k / g)[:, None, :] * (1 - np.exp(-g[:, None, :] * self.times[None, :, None]))
        return (self.capture * lam).reshape(theta.shape[0], -1)

    def observe(self, latent, rng):
        return rng.poisson(latent).astype(float)

    def log_lik(self, latent, q):
        from scipy.special import gammaln
        return np.sum(q * np.log(latent) - latent - gammaln(q + 1), axis=-1)

    def sample_population(self, n, rng):
        return rng.multivariate_normal(self.pop_mean, self.pop_cov, size=n)


# --------------------------------------------------------------------------
class SSAModel(Model):
    """A Gillespie model observed at `times`. Outputs are time-major: the observed species
    at time 1, then at time 2, and so on. The latent is theta itself, and every call to
    observe runs the simulator, so each output draw costs a full simulation.

    design: "snapshot" (each time on fresh cells, one block per time) or
            "tracked"  (every time on the same cells: one block, one trajectory per cell)."""

    stochastic_sim = True

    def _setup(self, name, names, lower, upper, times, obs_species, species, design, meta):
        self.times = np.asarray(times, float)
        self.obs_species = list(obs_species)
        self.n_obs = len(obs_species)
        nt = len(times)
        groups = [[k] for k in range(nt)] if design == "snapshot" else [list(range(nt))]
        if design not in ("snapshot", "tracked"):
            raise ValueError(f"unknown design {design!r}")
        blocks = [np.array([self.n_obs * k + s for k in g for s in range(self.n_obs)]) for g in groups]
        labels = ["t=" + "+".join(f"{times[k]:g}" for k in g) for g in groups]
        Model.__init__(self, name=name, param_names=names, lower=np.log(lower), upper=np.log(upper),
                       blocks=blocks, block_labels=labels, discrete=True,
                       meta=dict(meta, times=list(times), design=design,
                                 observed=[species[s] for s in obs_species]))

    def latent(self, theta):
        return np.array(theta, float, copy=True)

    def observe_cols(self, latent, cols, rng):
        cols = np.asarray(cols)
        tk = sorted({int(c) // self.n_obs for c in cols})
        traj = ssa(self.x0(len(latent)), self.stoich, self.propensity, np.exp(latent), self.times[tk], rng)
        obs = traj[:, :, self.obs_species].reshape(len(latent), -1).astype(float)
        pos = {tk[i] * self.n_obs + s: i * self.n_obs + s for i in range(len(tk)) for s in range(self.n_obs)}
        return obs[:, [pos[int(c)] for c in cols]]

    def observe(self, latent, rng):
        return self.observe_cols(latent, np.arange(self.n_out), rng)

    def simulate_cols(self, theta, cols, rng):
        return self.observe_cols(theta, cols, rng)


class BirthDeathSSA(SSAModel):
    """Two genes, each with constant transcription k_g and first-order decay gamma_g, from zero
    copies, simulated with Gillespie. Transcription rates are correlated across cells. The
    snapshot likelihood is exact: X_g(t) ~ Poisson(k_g / gamma_g (1 - exp(-gamma_g t))), so the
    exact hierarchical sampler can serve as a control."""

    stoich = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    def __init__(self, times=(0.5, 1, 2, 3, 4, 6, 8, 12), design="snapshot", corr=0.6):
        self._setup("bd_ssa", ["log_k1", "log_k2", "log_g1", "log_g2"], [1.0, 1.0, 0.05, 0.05],
                    [300.0, 300.0, 2.0, 2.0], times, [0, 1], ["X1", "X2"], design, dict(corr=corr))
        self.pop_mean = np.log([40.0, 20.0, 0.3, 0.5])
        sds = np.array([0.3, 0.3, 0.2, 0.2])
        C = np.eye(4)
        C[0, 1] = C[1, 0] = corr
        self.pop_cov = C * np.outer(sds, sds)

    def x0(self, n):
        return np.zeros((n, 2), np.int64)

    @staticmethod
    def propensity(x, r):
        return np.stack([r[:, 0], r[:, 2] * x[:, 0], r[:, 1], r[:, 3] * x[:, 1]], axis=1)

    def latent_cols(self, theta, cols):
        """Poisson means of outputs `cols` (exact snapshot likelihood)."""
        k, g = np.exp(theta[:, :2]), np.exp(theta[:, 2:])
        cols = np.asarray(cols)
        t = self.times[cols // 2]
        s = cols % 2
        return (k[:, s] / g[:, s]) * (1 - np.exp(-g[:, s] * t))

    def log_lik(self, latent, q):
        if self.meta["design"] != "snapshot":
            raise NotImplementedError("snapshot likelihood; tracked cells use block_log_lik")
        from scipy.special import gammaln
        return np.sum(q * np.log(latent) - latent - gammaln(q + 1), axis=-1)

    def block_log_lik(self, theta, cols, q):
        """Exact log-likelihood of outputs `cols` for each cell. Snapshots: Poisson. Tracked cells:
        each gene is an immigration-death (M/M/infinity) chain, so over an interval of length d the
        x molecules present survive independently with probability p = exp(-gamma d) and the new
        ones are Poisson(k/gamma (1 - p)):
            P(x' | x) = sum_j Binom(j; x, p) Pois(x' - j; k/gamma (1 - p)),
        with x = 0 at t = 0. The trajectory likelihood is the product over intervals."""
        if self.meta["design"] == "snapshot":
            return self.log_lik(self.latent_cols(theta, cols), q)
        from scipy.special import gammaln, logsumexp

        cols = np.asarray(cols)
        t_idx, s_idx = cols // self.n_obs, cols % self.n_obs
        k, g = np.exp(theta[:, :2]), np.exp(theta[:, 2:4])
        out = np.zeros(len(theta))
        for s in range(self.n_obs):
            sel = np.where(s_idx == s)[0]
            sel = sel[np.argsort(t_idx[sel])]
            ts = self.times[t_idx[sel]]
            x_prev, t_prev = np.zeros(len(theta)), 0.0
            for c, t in zip(sel, ts):
                x = q[:, c]
                p = np.exp(-g[:, s] * (t - t_prev))
                lam = k[:, s] / g[:, s] * (1 - p)
                J = int(min(x_prev.max(), x.max())) + 1
                j = np.arange(J)[None, :]
                ok = (j <= x_prev[:, None]) & (j <= x[:, None])
                jj = np.where(ok, j, 0)
                lb = (gammaln(x_prev[:, None] + 1) - gammaln(jj + 1) - gammaln(x_prev[:, None] - jj + 1)
                      + jj * np.log(p)[:, None] + (x_prev[:, None] - jj) * np.log1p(-p)[:, None])
                m = x[:, None] - jj
                lp = m * np.log(lam)[:, None] - lam[:, None] - gammaln(m + 1)
                out += logsumexp(np.where(ok, lb + lp, -np.inf), axis=1)
                x_prev, t_prev = x, t
        return out

    def sample_population(self, n, rng):
        return rng.multivariate_normal(self.pop_mean, self.pop_cov, size=n)


class FeedbackSSA(SSAModel):
    """Telegraph gene with negative autoregulation, simulated with Gillespie; mRNA (M) and
    protein (P) are measured, the promoter state is not. Reactions:

        off -> on   k_on / (1 + (P/K)^2)        on -> off   k_off
        on -> on + M   k_m                      M -> 0      gamma_m M
        M -> M + P     k_p M                    P -> 0      gamma_p P  (gamma_p = 0.2, known)

    Starts with the promoter off and no molecules. The chemical master equation has no closed
    form, so there is no exact per-cell likelihood. Transcription and translation rates vary
    across cells (correlated); the other rates are shared."""

    stoich = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    gamma_p = 0.2

    def __init__(self, times=(1, 2, 4, 8, 16, 24), design="snapshot", corr=0.5):
        self._setup("feedback_ssa", ["log_kon", "log_koff", "log_km", "log_gm", "log_kp", "log_K"],
                    [0.1, 0.1, 2.0, 0.2, 0.5, 5.0], [10.0, 10.0, 200.0, 5.0, 20.0, 500.0],
                    times, [1, 2], ["G", "M", "P"], design, dict(corr=corr, gamma_p=self.gamma_p))
        self.pop_mean = np.log([1.0, 1.0, 20.0, 1.0, 2.0, 50.0])
        sds = np.array([0.0, 0.0, 0.3, 0.0, 0.3, 0.0])
        C = np.eye(6)
        C[2, 4] = C[4, 2] = corr
        self.pop_cov = C * np.outer(sds, sds)

    def x0(self, n):
        return np.zeros((n, 3), np.int64)

    def propensity(self, x, r):
        G, M, P = x[:, 0], x[:, 1], x[:, 2]
        on = r[:, 0] / (1 + (P / r[:, 5]) ** 2)
        return np.stack([on * (1 - G), r[:, 1] * G, r[:, 2] * G, r[:, 3] * M, r[:, 4] * M,
                         self.gamma_p * P], axis=1)

    def sample_population(self, n, rng):
        v = np.diag(self.pop_cov) > 0  # shared rates are exactly fixed
        out = np.tile(self.pop_mean, (n, 1))
        out[:, v] = rng.multivariate_normal(self.pop_mean[v], self.pop_cov[np.ix_(v, v)], size=n)
        return out


# --------------------------------------------------------------------------
def _fr_pmf(n0, a, r, chunk=2000):
    """pmf of prey eaten (0..n0) in a unit-length trial, from the master equation, for each
    (a, r) pair. States: searching with i eaten (i = 0..n0), handling with i eaten."""
    from scipy.linalg import expm

    m = n0 + 1
    i = np.arange(m)
    out = np.empty((len(a), m))
    for s in range(0, len(a), chunk):
        aa, rr = a[s:s + chunk], r[s:s + chunk]
        A = np.zeros((len(aa), 2 * m, 2 * m))
        A[:, i, i] = -(n0 - i)[None, :] * aa[:, None]  # searching i -> handling i + 1
        A[:, m + i[1:], i[:-1]] = (n0 - i[:-1])[None, :] * aa[:, None]
        A[:, i[1:], m + i[1:]] = rr[:, None]  # handling i -> searching i
        A[:, m + i[1:], m + i[1:]] = -rr[:, None]
        p = expm(A)[:, :, 0]  # start: searching, none eaten
        out[s:s + chunk] = np.clip(p[:, :m] + p[:, m:], 0.0, 1.0)
    return out


class FunctionalResponse(Model):
    """Predator foraging trial (stickleback functional response; Newport et al. 2026). A fish
    switches between searching, catching each of the n0 - i remaining prey at rate a, and
    handling, which ends at rate r. A trial lasts one time unit and starts with the fish
    searching. Each fish is observed once, at one initial prey number: one block per n0,
    with one output (prey eaten).

    Simulation is Gillespie (cheap: at most 2 n0 events). The exact likelihood needs a matrix
    exponential of size 2(n0 + 1) for every fish and every parameter value.

    Synthetic population (from the fitted mixture model of Newport et al.): a fraction
    `vorac` of fish are voracious (very fast attack, log a ~ N(log 300, 0.3^2)), the rest
    attack slowly (log a ~ N(-2.2, 0.3^2)); log r ~ N(2.4, 0.6^2) in both."""

    stochastic_sim = True

    def __init__(self, n0s=(5, 10, 15, 20, 25, 30, 40), vorac=0.9):
        super().__init__(name="fish", param_names=["log_a", "log_r"], lower=np.log([0.01, 0.5]),
                         upper=np.log([1000.0, 200.0]), blocks=[np.array([k]) for k in range(len(n0s))],
                         block_labels=[f"n0={n}" for n in n0s], discrete=True,
                         meta=dict(n0s=list(n0s), vorac=vorac))
        self.n0s, self.vorac = list(n0s), vorac

    def latent(self, theta):
        return np.array(theta, float, copy=True)

    def observe_cols(self, latent, cols, rng):
        rates = np.exp(latent)
        out = np.empty((len(latent), len(cols)))
        stoich = np.array([[1, 1], [0, -1]])  # (eaten, handling)
        for k, c in enumerate(cols):
            n0 = self.n0s[int(c)]
            prop = lambda x, th, n0=n0: np.stack([th[:, 0] * (n0 - x[:, 0]) * (1 - x[:, 1]), th[:, 1] * x[:, 1]], 1)
            out[:, k] = ssa(np.zeros((len(latent), 2), np.int64), stoich, prop, rates, [1.0], rng)[:, 0, 0]
        return out

    def observe(self, latent, rng):
        return self.observe_cols(latent, np.arange(self.n_out), rng)

    def simulate_cols(self, theta, cols, rng):
        return self.observe_cols(theta, cols, rng)

    def block_log_lik(self, theta, cols, q):
        n0 = self.n0s[int(np.asarray(cols)[0])]
        P = _fr_pmf(n0, np.exp(theta[:, 0]), np.exp(theta[:, 1]))
        return np.log(np.maximum(P[np.arange(len(q)), np.asarray(q)[:, 0].astype(int)], 1e-300))

    def sample_population(self, n, rng):
        v = rng.uniform(size=n) < self.vorac
        la = np.where(v, np.log(300.0), -2.2) + 0.3 * rng.standard_normal(n)
        lr = 2.4 + 0.6 * rng.standard_normal(n)
        return np.column_stack([la, lr])


# --------------------------------------------------------------------------
# mRNA transfection (Fröhlich et al. 2018; the single-cell benchmark of Arruda et al. 2024).
# Each cell is a tracked trajectory: log GFP fluorescence at 1, 2, ..., 30 h (one block).
TF_TIMES = np.arange(1.0, 31.0)


def _sort_rates(theta):
    """The mean trajectory is symmetric in (delta, gamma): label so that delta >= gamma."""
    th = np.array(theta, float, copy=True)
    hi, lo = np.maximum(th[:, 0], th[:, 1]), np.minimum(th[:, 0], th[:, 1])
    th[:, 0], th[:, 1] = hi, lo
    return th


def _relax(x, tau):
    """(1 - exp(-x tau)) / x, stable as x -> 0."""
    z = x * tau
    small = np.abs(z) < 1e-8
    return np.where(small, tau, -np.expm1(-np.where(small, 1.0, z)) / np.where(small, 1.0, x))


class Transfection(Model):
    """ODE model of Arruda et al. (2024, 'simple Fröhlich model'): after transfection at t0,
    mRNA decays at rate delta and is translated into GFP, which decays at rate gamma:
        p(t) = km0 exp(-gamma tau) (1 - exp(-(delta - gamma) tau)) / (delta - gamma),  tau = t - t0 > 0,
    observed as log(p + offset) + N(0, sigma^2). theta = log(delta, gamma, km0, t0, offset, sigma).
    p is symmetric in (delta, gamma), so the labels are fixed afterwards by delta >= gamma.
    Prior: the normal prior of Arruda et al. Population for synthetic data: their 'true'
    synthetic population (log-normal; offset and sigma shared)."""

    def __init__(self, times=tuple(TF_TIMES)):
        super().__init__(name="transfection", param_names=["log_delta", "log_gamma", "log_km0", "log_t0",
                                                            "log_offset", "log_sigma"],
                         lower=None, upper=None, prior_mean=np.array([-3.0, -3.0, 5.0, 0.0, 0.0, -1.0]),
                         prior_sd=np.sqrt([5.0, 5.0, 11.0, 2.0, 6.0, 2.0]),
                         blocks=[np.arange(len(times))], block_labels=["trajectory"],
                         meta=dict(times=list(times)))
        self.times = np.asarray(times, float)
        self.pop_mean = np.array([-0.69364, -7.01357, 6.21662, -0.16357, 2.07944, -3.45388])
        self.pop_var = np.array([0.94098, 7.01402, 0.00396, 0.11641, 0.0, 0.0])

    def mean(self, theta):
        d, g, km, t0, off = (np.exp(theta[:, k])[:, None] for k in range(5))
        tau = np.maximum(self.times[None, :] - t0, 0.0)
        lo, hi = np.minimum(d, g), np.maximum(d, g)  # symmetric in (delta, gamma): stable form
        p = km * np.exp(-lo * tau) * _relax(hi - lo, tau)
        return np.log(p + off)

    def latent(self, theta):
        """Noise-free outputs, plus the cell's noise sd as an extra last column."""
        return np.hstack([self.mean(theta), np.exp(theta[:, 5:6])])

    def observe_cols(self, latent, cols, rng):
        return latent[:, cols] + latent[:, -1:] * rng.standard_normal((len(latent), len(cols)))

    def observe(self, latent, rng):
        return self.observe_cols(latent, np.arange(self.n_out), rng)

    def simulate_cols(self, theta, cols, rng):
        return self.observe_cols(self.latent(theta), cols, rng)

    def block_log_lik(self, theta, cols, q):
        mu = self.mean(theta)[:, cols]
        s = np.exp(theta[:, 5:6])
        return np.sum(-0.5 * ((q - mu) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi), axis=1)

    def canonical(self, theta):
        return _sort_rates(theta)

    def sample_population(self, n, rng):
        return self.pop_mean + np.sqrt(self.pop_var) * rng.standard_normal((n, 6))


class TransfectionSSA(SSAModel):
    """Stochastic version (exact Gillespie; Arruda et al. use the chemical Langevin SDE): from
    transfection at t0, m0 mRNA molecules decay (delta) and are translated (k) into protein, which
    decays (gamma). Observed: log(scale P + offset) + N(0, sigma^2) at 1, ..., 30 h.
    theta = log(delta, gamma, k, m0, scale, t0, offset, sigma). No tractable likelihood.
    Prior: a box (bounds keep the number of reactions per simulation manageable). Synthetic
    population: Arruda et al.'s 'true' SDE population, clipped to the box."""

    stoich = np.array([[-1, 0], [0, 1], [0, -1]])  # (M, P): mRNA decay, translation, protein decay

    def __init__(self, times=tuple(TF_TIMES)):
        self.times = np.asarray(times, float)
        self.obs_species, self.n_obs = [1], 1
        Model.__init__(self, name="transfection_ssa",
                       param_names=["log_delta", "log_gamma", "log_k", "log_m0", "log_scale", "log_t0",
                                    "log_offset", "log_sigma"],
                       lower=np.array([-4.0, -12.0, -3.0, 4.0, -2.0, -3.0, 0.0, -5.0]),
                       upper=np.array([1.5, 0.0, 1.5, 6.5, 3.0, 2.0, 4.0, 0.0]),
                       blocks=[np.arange(len(times))], block_labels=["trajectory"],
                       meta=dict(times=list(times)))
        self.pop_mean = np.array([-0.69364, -7.01357, 0.02702, 5.70375, 0.75142, -0.16357, 2.07944, -3.45388])
        self.pop_var = np.array([0.94098, 7.01402, 0.67524, 6e-05, 0.0, 0.11641, 0.0, 0.0])

    def x0(self, n):
        return np.zeros((n, 2), np.int64)

    @staticmethod
    def propensity(x, r):
        return np.stack([r[:, 0] * x[:, 0], r[:, 2] * x[:, 0], r[:, 1] * x[:, 1]], axis=1)

    def observe_cols(self, latent, cols, rng):
        cols = np.asarray(cols)
        r = np.exp(latent)
        x0 = self.x0(len(latent))
        x0[:, 0] = np.rint(r[:, 3]).astype(np.int64)  # m0 molecules at t0
        last = int(cols.max())
        P = ssa(x0, self.stoich, self.propensity, r, self.times[:last + 1], rng, t_start=r[:, 5])[:, :, 1]
        y = np.log(r[:, 4:5] * P[:, cols] + r[:, 6:7])
        return y + r[:, 7:8] * rng.standard_normal(y.shape)

    def sample_population(self, n, rng):
        th = self.pop_mean + np.sqrt(self.pop_var) * rng.standard_normal((n, 8))
        return np.clip(th, self.lower, self.upper)

    def canonical(self, theta):
        return _sort_rates(theta)


MODELS = {
    "scalar": ScalarGaussian,
    "linear": LinearGaussian,
    "growth": GrowthFactor,
    "gene": GeneExpression,
    "bd_ssa": BirthDeathSSA,
    "feedback_ssa": FeedbackSSA,
    "fish": FunctionalResponse,
    "transfection": Transfection,
    "transfection_ssa": TransfectionSSA,
}


def make_model(name: str, **kwargs) -> Model:
    return MODELS[name](**kwargs)


def make_snapshot_data(model: Model, n_cells: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Each block is measured on its own fresh cells; within a block, all outputs come from
    the same cells (one output pair per block = destructive snapshots)."""
    data = []
    for b in model.blocks:
        theta = model.sample_population(n_cells, rng)
        q = model.simulate(theta, rng, noise=True)
        data.append(q[:, b])
    return data
