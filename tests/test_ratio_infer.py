import numpy as np
import pytest
from scipy.stats import norm

from cmc.gaussian import scalar_rl_path
from cmc.infer import Settings, iterate_blocks, weighted_moments
from cmc.models import ScalarGaussian, make_snapshot_data
from cmc.ratio import ClassifierRatio, KdeRatio


def _two_gaussians(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(1.0, 1.0, (n, 1))
    sim = rng.normal(0.0, 2.0, (n, 1))
    grid = np.linspace(-1, 3, 9)[:, None]
    truth = norm.logpdf(grid[:, 0], 1, 1) - norm.logpdf(grid[:, 0], 0, 2)
    return data, sim, grid, truth


@pytest.mark.parametrize("est", [KdeRatio(), ClassifierRatio("logistic2")])
def test_ratio_estimators_recover_gaussian_log_ratio(est):
    data, sim, grid, truth = _two_gaussians()
    lr = est.fit(data, sim).log_ratio(grid)
    assert np.max(np.abs(lr - truth)) < 0.25


def test_discrete_kde_ratio_preserves_pmf_ratio():
    rng = np.random.default_rng(1)
    data = rng.poisson(5, (6000, 1)).astype(float)
    sim = rng.poisson(8, (6000, 1)).astype(float)
    k = np.array([[3.0], [5.0], [8.0]])
    truth = (np.log(5) - np.log(8)) * k[:, 0] - 5 + 8
    lr = KdeRatio(discrete=True, rng=rng).fit(data, sim).log_ratio(k)
    assert np.max(np.abs(lr - truth)) < 0.35


def test_one_update_is_cmc_for_deterministic_map():
    """noise=False, one block, one update: weights = target / prior pushforward."""
    model = ScalarGaussian()
    rng = np.random.default_rng(2)
    data = make_snapshot_data(model, 3000, rng)
    s = Settings(ratio="logistic2", n_particles=8000, n_sweeps=1, noise=False, ess_frac=0.0, seed=3)
    out = iterate_blocks(model, data, [0], s)
    m, C = weighted_moments(out["theta"], out["logw"])
    tv = model.pop_sd**2 + model.sigma**2  # noise ignored: CMC reproduces the noisy spread
    assert abs(m[0] - model.pop_mean) < 0.1 and abs(C[0, 0] - tv) < 0.3


@pytest.mark.slow
def test_iterated_scalar_tracks_closed_form():
    model = ScalarGaussian()
    rng = np.random.default_rng(4)
    data = make_snapshot_data(model, 4000, rng)
    s = Settings(ratio="logistic2", n_particles=10000, n_sweeps=15, n_fit=4000, seed=5)
    out = iterate_blocks(model, data, [0], s)
    path = scalar_rl_path(0.0, model.meta["prior_var"], model.sigma**2, model.pop_mean,
                          model.pop_sd**2 + model.sigma**2, 15)
    sd_hat = np.array([h["sd"][0] for h in out["history"]])
    assert np.allclose(sd_hat[[0, 4, 14]], np.sqrt(path[[0, 4, 14], 1]), atol=0.12)
