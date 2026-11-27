import numpy as np
import pytest
import yaml

from cmc.models import make_model
from cmc.npmle import Observations, npmle, solve_weights

CFG = yaml.safe_load(open("config/v2.yaml"))
EXACT = ["linear_bayes", "growth_snapshot", "growth_tracked", "gene_n2000", "bd_ssa", "bd_ssa_tracked",
         "fish_sim", "tf_sim"]


@pytest.mark.parametrize("example", EXACT)
def test_loglik_matrix_matches_pairwise(example):
    ex = CFG["examples"][example]
    model = make_model(ex.get("kind", example), **ex["model"])
    rng = np.random.default_rng(0)
    pop = model.sample_population(30, rng)
    blocks = list(range(len(model.blocks)))[:3]
    data = {j: model.simulate_cols(pop, model.blocks[j], rng) for j in blocks}
    obs = Observations(model, data, blocks)
    theta = model.sample_population(7, rng)
    L = obs.loglik_matrix(theta)
    r0 = 0
    for b, q in enumerate(obs.Q):
        for k in range(len(theta)):
            ref = obs.lik.block_ll(b, np.repeat(theta[k:k + 1], len(q), axis=0), q)
            np.testing.assert_allclose(L[r0:r0 + len(q), k], ref, rtol=1e-4, atol=1e-3)
        r0 += len(q)


def test_solver_kkt_and_recovery():
    # 1-D Gaussian location mixture: true support {-2, 2} with weights 0.3 / 0.7
    rng = np.random.default_rng(1)
    n = 4000
    x = np.where(rng.uniform(size=n) < 0.3, -2.0, 2.0) + rng.standard_normal(n)
    grid = np.linspace(-5, 5, 201)
    logL = (-0.5 * (x[:, None] - grid[None]) ** 2).astype(np.float32)
    w, ll, it = solve_weights(logL, np.ones(n), gtol=1e-4, max_iter=100000)
    # Lindsay's gradient condition: D(theta) = mean_i L_i(theta) / f_i <= 1 everywhere, = 1 on the support
    L = np.exp(logL.astype(float))
    D = (L / (L @ w)[:, None]).mean(0)
    assert D.max() < 1 + 1e-3
    assert abs(w[grid < 0].sum() - 0.3) < 0.03
    assert abs(np.sum(w * grid) - (0.3 * -2 + 0.7 * 2)) < 0.1


def test_npmle_bd_snapshot_small():
    ex = CFG["examples"]["bd_ssa"]
    model = make_model(ex.get("kind", "bd_ssa"), **ex["model"])
    rng = np.random.default_rng(2)
    pop = model.sample_population(300, rng)
    data = {j: model.simulate_cols(pop, model.blocks[j], rng) for j in range(len(model.blocks))}
    obs = Observations(model, data, range(len(model.blocks)))
    theta, w, _, hist, _ = npmle(obs, rng, n_grid=500, n_new=300, n_rounds=3, log=lambda s: None)
    lls = [h["loglik"] for h in hist]
    assert all(b >= a - 1e-6 * abs(a) for a, b in zip(lls, lls[1:]))  # refinement never lowers the optimum
    assert np.isclose(w.sum(), 1)
