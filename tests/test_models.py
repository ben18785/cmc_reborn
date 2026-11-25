import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cmc.models import GeneExpression, GrowthFactor, LinearGaussian, ScalarGaussian, make_snapshot_data


def test_growth_closed_form_matches_ode():
    m = GrowthFactor()
    theta = m.pop_mean[None, :] + np.array([[0.1, -0.2, 0.3, 0.1, -0.1]])
    RT, k1, km1, kdeg, kdegs = np.exp(theta[0])
    lat = m.latent(theta)[0]
    for k, (L, t) in enumerate(m.conds):
        f = lambda _t, x: [RT * kdeg - k1 * L * x[0] + km1 * x[1] - kdeg * x[0],
                           k1 * L * x[0] - km1 * x[1] - kdegs * x[1]]
        sol = solve_ivp(f, (0, t), [0.0, 0.0], rtol=1e-10, atol=1e-6, method="LSODA")
        assert np.allclose(lat[2 * k:2 * k + 2], np.log(sol.y[:, -1]), atol=1e-5)


def test_gene_poisson_mean():
    m = GeneExpression()
    rng = np.random.default_rng(0)
    theta = np.repeat(m.pop_mean[None, :], 20000, axis=0)
    lat = m.latent(theta)
    q = m.observe(lat, rng)
    assert np.allclose(q.mean(0), lat[0], rtol=0.05, atol=0.05)
    k, g = np.exp(m.pop_mean[0]), np.exp(m.pop_mean[3])
    assert np.isclose(lat[0, 0], m.capture * k / g * (1 - np.exp(-g * m.times[0])))


def test_linear_shapes_and_blocks():
    m = LinearGaussian(p=4, n_blocks=6, d=2)
    th = m.sample_prior(10, np.random.default_rng(1))
    assert m.latent(th).shape == (10, 12)
    assert np.allclose(m.latent(th)[:, m.blocks[3]], th @ m.A[3].T)
    assert np.all(np.linalg.eigvalsh(m.Sigma) > 0)


def test_reflection_stays_in_box():
    m = GrowthFactor()
    rng = np.random.default_rng(2)
    t = m.lower + (m.upper - m.lower) * rng.uniform(-3, 4, size=(1000, m.p))
    r = m.project_to_support(t)
    assert np.all(r >= m.lower - 1e-12) and np.all(r <= m.upper + 1e-12)
    inside = m.sample_prior(50, rng)
    assert np.allclose(m.project_to_support(inside), inside)


@pytest.mark.parametrize("cls", [ScalarGaussian, LinearGaussian, GrowthFactor, GeneExpression])
def test_snapshot_data(cls):
    m = cls()
    data = make_snapshot_data(m, 50, np.random.default_rng(3))
    assert len(data) == len(m.blocks)
    assert all(d.shape == (50, len(b)) for d, b in zip(data, m.blocks))
    assert all(np.all(np.isfinite(d)) for d in data)
