"""Exact-likelihood NPMLE of a mixing distribution on an adaptive candidate grid.

Used as a reference ('oracle') for the fixed point that iterated CMC targets when the per-cell
likelihood is available in closed form. See workflow/scripts/fit_npmle.py."""
import numpy as np

from cmc.bayes import ExactLik


class Observations:
    """Distinct observations per block, with multiplicities (count data repeat)."""

    def __init__(self, model, data, fit_blocks):
        self.model, self.fit_blocks = model, list(fit_blocks)
        self.Q, C = [], []
        for j in self.fit_blocks:
            u, cnt = np.unique(np.asarray(data[j], float), axis=0, return_counts=True)
            self.Q.append(u)
            C.append(cnt.astype(float))
        self.c = np.concatenate(C)
        self.lik = ExactLik(model, {j: data[j] for j in self.fit_blocks}, self.fit_blocks)
        self.pairwise = getattr(model, "block_log_lik", None) is not None

    def loglik_matrix(self, theta, chunk=200):
        """log K(y_i | theta_k), distinct observations i x candidates k, float32."""
        out = np.empty((len(self.c), len(theta)), np.float32)
        r0 = 0
        for b, q in enumerate(self.Q):
            cols = self.lik.cols(b)
            for s in range(0, len(theta), chunk):
                th = theta[s:s + chunk]
                if self.pairwise:
                    ll = self.lik.block_ll(b, np.repeat(th, len(q), axis=0),
                                           np.tile(q, (len(th), 1))).reshape(len(th), len(q))
                else:  # latent once per candidate, broadcast over the observations
                    lat = self.model.latent_cols(th, cols)
                    ll = self.model.log_lik(lat[:, None, :], q[None, :, :])
                out[r0:r0 + len(q), s:s + len(th)] = ll.T
            r0 += len(q)
        return out


def solve_weights(logL, c, w0=None, gtol=1e-4, max_iter=20000):
    """Weights maximising sum_i c_i log sum_k w_k exp(logL_ik): EM (Laird 1978; Vardi et al. 1985)
    with SQUAREM acceleration (Varadhan & Roland 2008), stopped by the optimality condition of the
    NPMLE on the grid (Lindsay 1983): D_k = sum_i c_i L_ik / f_i / sum_i c_i <= 1 for all k, with
    equality on the support; stop when max_k D_k <= 1 + gtol. Returns (w, log-likelihood, EM steps)."""
    m = logL.max(axis=1, keepdims=True).astype(float)
    m = np.where(np.isfinite(m), m, 0.0)
    L = np.exp(logL - m.astype(np.float32))
    w = np.full(L.shape[1], 1.0 / L.shape[1]) if w0 is None else np.asarray(w0, float) / np.sum(w0)
    cs = np.asarray(c, float) / np.sum(c)

    def em(w):
        return w * (L.T @ (cs / np.maximum(L @ w, 1e-300)))

    def obj(w):
        return float(cs @ np.log(np.maximum(L @ w, 1e-300)))

    it = 0
    while it < max_iter:
        D = L.T @ (cs / np.maximum(L @ w, 1e-300))
        if D.max() <= 1 + gtol:
            break
        w1 = w * D
        w2 = em(w1)
        r, v = w1 - w, w2 - 2 * w1 + w
        alpha = -np.sqrt((r @ r) / max(v @ v, 1e-300))
        wn = np.maximum(w - 2 * alpha * r + alpha * alpha * v, 0.0)
        wn = em(wn / wn.sum()) if wn.sum() > 0 else w2
        o2, on = obj(w2), obj(wn)
        if on < o2:  # the extrapolation did not improve: plain EM step
            wn, on = w2, o2
        w, it = wn / wn.sum(), it + 3
    ll = float(np.asarray(c, float) @ (np.log(np.maximum(L @ w, 1e-300)) + m[:, 0]))
    return w, ll, it


def npmle(obs, rng, n_grid=4000, n_new=3000, n_rounds=8, keep=1e-4, h0=0.2, minutes=None, log=print, **kw):
    """Adaptive-grid NPMLE (cf. NPAG): prior candidates, then rounds of jittered candidates around the
    current support with a halving jitter scale; refinement stops early (recorded in the history) once
    `minutes` have elapsed. Returns (theta, w, logL, history, n_lik_evals)."""
    import time
    t0 = time.time()
    model = obs.model
    scale = (model.upper - model.lower) / np.sqrt(12) if model.lower is not None else model.prior_sd
    theta = model.sample_prior(n_grid, rng)
    logL = obs.loglik_matrix(theta)
    n_lik = logL.size
    w, ll, it = solve_weights(logL, obs.c, **kw)
    hist = [dict(round=0, n_cand=len(theta), loglik=ll, em_steps=it)]
    log(f"round 0: {len(theta)} candidates, log-lik {ll:.2f}, {it} EM steps")
    h = h0
    for r in range(1, n_rounds + 1):
        if minutes is not None and time.time() - t0 > 60 * minutes:
            log(f"time budget reached after {r - 1} refinement rounds")
            hist[-1]["stopped_early"] = True
            break
        sup = w > keep * w.max()
        theta, logL, w = theta[sup], logL[:, sup], w[sup]
        parent = rng.choice(len(theta), size=n_new, p=w / w.sum())
        new = model.project_to_support(theta[parent] + h * scale * rng.standard_normal((n_new, model.p)))
        if model.lower is None:
            new = np.clip(new, model.prior_mean - 6 * model.prior_sd, model.prior_mean + 6 * model.prior_sd)
        lnew = obs.loglik_matrix(new)
        n_lik += lnew.size
        theta, logL = np.vstack([theta, new]), np.hstack([logL, lnew])
        w, ll, it = solve_weights(logL, obs.c, np.r_[w, np.full(n_new, 1e-3 * w.min() + 1e-12)], **kw)
        n_sup = int((w > keep * w.max()).sum())
        hist.append(dict(round=r, n_cand=len(theta), loglik=ll, em_steps=it, h=h, n_support=n_sup))
        log(f"round {r}: h {h:.4f}, {len(theta)} candidates, log-lik {ll:.2f}, {it} EM steps, support {n_sup}")
        h /= 2
    return theta, w, logL, hist, n_lik
