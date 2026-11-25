import numpy as np

from cmc.gaussian import (linear_block_targets, linear_rl_path, linear_rl_step, scalar_fixed_point,
                          scalar_rl_path)
from cmc.models import LinearGaussian


def test_scalar_path_converges_to_deconvolution():
    path = scalar_rl_path(0.0, 10.0, 1.0, 1.0, 2.0, 2000)
    assert np.allclose(path[-1], scalar_fixed_point(1.0, 1.0, 2.0), atol=1e-3)
    # first step = augmented one-step update: theta|q mixed over the target
    a = 10 / 11
    assert np.isclose(path[0, 1], a * a * 2.0 + 10 * 1 / 11)
    # pushforward of the first step is too wide (overstates heterogeneity)
    assert path[0, 1] + 1.0 > 2.0


def test_multivariate_step_matches_scalar():
    m, v = linear_rl_step(np.zeros(1), np.eye(1) * 10.0, np.eye(1), np.eye(1), np.ones(1), np.eye(1) * 2.0)
    s = scalar_rl_path(0.0, 10.0, 1.0, 1.0, 2.0, 1)[0]
    assert np.allclose([m[0], v[0, 0]], s)


def test_fixed_point_is_stationary():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2, 3))
    S = 0.3 * np.eye(2)
    m0, V0 = rng.standard_normal(3), np.eye(3) * 0.5
    mu_q, T = A @ m0, A @ V0 @ A.T + S
    m1, V1 = linear_rl_step(m0, V0, A, S, mu_q, T)
    assert np.allclose(m1, m0) and np.allclose(V1, V0)


def test_block_cycling_recovers_identified_population():
    model = LinearGaussian()
    path = linear_rl_path(model, lambda s: range(len(model.blocks)), 400)
    m, V = path[-1]
    targets = linear_block_targets(model)
    S = model.sigma**2 * np.eye(model.d)
    for j, (mu_q, T) in enumerate(targets):
        A = model.A[j]
        assert np.allclose(A @ m, mu_q, atol=1e-3)
        assert np.allclose(A @ V @ A.T + S, T, atol=2e-2)
    assert np.allclose(m, model.mu, atol=1e-2)
    assert np.allclose(V, model.Sigma, atol=5e-2)
