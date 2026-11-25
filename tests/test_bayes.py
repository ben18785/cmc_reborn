import numpy as np

from cmc.bayes import BayesSettings, ExactLik, NreLik, fit_hierarchical
from cmc.models import LinearGaussian, ScalarGaussian, make_snapshot_data


def test_gaussian_mixed_effects_recovers_linear_population():
    model = LinearGaussian(p=2, n_blocks=4, d=2, sigma=0.3, seed=3)
    rng = np.random.default_rng(0)
    data = make_snapshot_data(model, 400, rng)
    s = BayesSettings(n_comp=1, n_burn=600, n_keep=400, thin=10, n_pop=500, seed=1)
    res = fit_hierarchical(model, data, range(4), s)
    pop = res["pop_draws"].reshape(-1, 2)
    assert np.all(np.abs(pop.mean(0) - model.mu) < 0.15)
    assert np.all(np.abs(pop.std(0) / np.sqrt(np.diag(model.Sigma)) - 1) < 0.25)
    # posterior over the population sd is a spread, not a point
    sds = res["pop_draws"].std(axis=1)
    assert np.all(sds.std(0) > 0)


def test_dp_mixture_runs_and_is_close():
    model = LinearGaussian(p=2, n_blocks=4, d=2, sigma=0.3, seed=3)
    data = make_snapshot_data(model, 300, np.random.default_rng(1))
    s = BayesSettings(n_comp=5, n_burn=400, n_keep=200, thin=10, n_pop=500, seed=2)
    pop = fit_hierarchical(model, data, range(4), s)["pop_draws"].reshape(-1, 2)
    assert np.all(np.abs(pop.mean(0) - model.mu) < 0.25)


def test_nre_logit_matches_exact_likelihood_differences():
    model = ScalarGaussian(sigma=1.0)
    rng = np.random.default_rng(0)
    data = [rng.normal(1.0, 1.4, (50, 1))]
    nre = NreLik(model, data, [0], hidden=[64, 64], n_train=20000)
    nre.train(lambda n: rng.normal(0, 2, (n, 1)), rng)
    ex = ExactLik(model, data, [0])
    th1, th2 = np.full((50, 1), 0.0), np.full((50, 1), 1.5)
    cells = np.arange(50)
    d_nre = nre(th2, cells) - nre(th1, cells)
    d_ex = ex(th2, cells) - ex(th1, cells)
    assert np.corrcoef(d_nre, d_ex)[0, 1] > 0.95
    assert np.median(np.abs(d_nre - d_ex)) < 0.3
