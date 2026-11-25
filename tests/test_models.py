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


def test_growth_designs_share_outputs_and_cells():
    from cmc.models import GrowthFactor, make_snapshot_data
    snap, pairs, tracked = (GrowthFactor(design=d) for d in ("snapshot", "pairs", "tracked"))
    assert [len(b) for b in snap.blocks] == [2] * 12
    assert [len(b) for b in pairs.blocks] == [4] * 10
    assert [len(b) for b in tracked.blocks] == [12, 12]
    assert sorted(np.concatenate(tracked.blocks).tolist()) == list(range(24))
    # tracked blocks: one cell's record across times is correlated (same theta)
    data = make_snapshot_data(tracked, 3000, np.random.default_rng(0))
    c = np.corrcoef(data[0][:, 0], data[0][:, 10])[0, 1]
    assert c > 0.2


def test_gillespie_birth_death_matches_poisson_and_tracks_cells():
    from scipy.stats import poisson

    from cmc.models import BirthDeathSSA

    rng = np.random.default_rng(0)
    m = BirthDeathSSA()
    th = m.sample_population(4000, rng)
    q = m.simulate(th, rng)
    lam = m.latent_cols(th, np.arange(m.n_out))
    # given theta, counts are Poisson: mean and variance of q - lam match lam
    assert np.allclose(q.mean(0) / lam.mean(0), 1, atol=0.03)
    assert np.allclose(((q - lam) ** 2).mean(0) / lam.mean(0), 1, atol=0.1)
    assert np.allclose(m.log_lik(lam[:5, :2], q[:5, :2]),
                       poisson.logpmf(q[:5, :2], lam[:5, :2]).sum(1))
    # a block's columns can be simulated on their own
    assert m.observe_cols(m.latent(th[:10]), m.blocks[3], rng).shape == (10, 2)
    # tracked design: one block, one trajectory per cell, so counts correlate over time
    mt = BirthDeathSSA(design="tracked")
    assert len(mt.blocks) == 1 and len(mt.blocks[0]) == 16
    th1 = np.tile(mt.pop_mean, (3000, 1))  # same rates in every cell: only intrinsic noise
    qt = mt.simulate(th1, rng)
    # gene 1 at t = 6 h (col 10) and t = 8 h (col 12): immigration-death gives
    # corr = exp(-g dt) sqrt(lam6 / lam8)
    k, g = np.exp(mt.pop_mean[0]), np.exp(mt.pop_mean[2])
    lam6, lam8 = k / g * (1 - np.exp(-6 * g)), k / g * (1 - np.exp(-8 * g))
    expected = np.exp(-2 * g) * np.sqrt(lam6 / lam8)
    assert abs(np.corrcoef(qt[:, 10], qt[:, 12])[0, 1] - expected) < 0.05


def test_gillespie_feedback_runs_and_represses():
    from cmc.models import FeedbackSSA

    rng = np.random.default_rng(1)
    m = FeedbackSSA()
    th = m.sample_population(2000, rng)
    q = m.simulate(th, rng)
    assert q.shape == (2000, 12) and (q >= 0).all()
    # stronger feedback (smaller K) lowers protein at the last time
    weak, strong = th.copy(), th.copy()
    weak[:, 5], strong[:, 5] = np.log(500.0), np.log(5.0)
    assert m.simulate(strong, rng)[:, -1].mean() < 0.7 * m.simulate(weak, rng)[:, -1].mean()


def test_fish_gillespie_matches_master_equation():
    from cmc.models import FunctionalResponse, _fr_pmf

    rng = np.random.default_rng(2)
    m = FunctionalResponse()
    th = m.sample_population(3000, rng)
    for c in (0, 3, 6):
        n0 = m.n0s[c]
        q = m.observe_cols(th, [c], rng)[:, 0].astype(int)
        P = _fr_pmf(n0, np.exp(th[:, 0]), np.exp(th[:, 1]))
        assert np.allclose(P.sum(1), 1, atol=1e-8)
        assert np.abs(np.bincount(q, minlength=n0 + 1) / len(q) - P.mean(0)).max() < 0.03
        ll = m.block_log_lik(th[:50], [c], q[:50, None])
        assert np.allclose(ll, np.log(P[np.arange(50), q[:50]]))


def test_count_classifier_recovers_discrete_ratio():
    from cmc.ratio import ClassifierRatio

    rng = np.random.default_rng(3)
    data = rng.poisson(3, (5000, 1)).astype(float)
    sim = rng.poisson(5, (5000, 1)).astype(float)
    grid = np.arange(0, 9)[:, None].astype(float)
    from scipy.stats import poisson
    truth = poisson.logpmf(grid[:, 0], 3) - poisson.logpmf(grid[:, 0], 5)
    lr = ClassifierRatio("logistic_bins", discrete=True).fit(data, sim).log_ratio(grid)
    assert np.max(np.abs(lr - truth)[1:8]) < 0.2  # counts 0 and 8 have only ~35-40 cells in one sample


def test_transfection_models():
    from cmc.models import Transfection, TransfectionSSA

    rng = np.random.default_rng(4)
    m = Transfection()
    th = m.sample_population(500, rng)
    q = m.simulate(th, rng)
    assert q.shape == (500, 30) and np.isfinite(q).all()
    # exact likelihood is Gaussian around the ODE solution; labels canonical: delta >= gamma
    sw = th.copy()
    sw[:, [0, 1]] = sw[:, [1, 0]]
    assert np.allclose(m.block_log_lik(th, np.arange(30), q), m.block_log_lik(sw, np.arange(30), q))
    assert (m.canonical(sw)[:, 0] >= m.canonical(sw)[:, 1]).all()
    # Gillespie version: before t0 the cell shows only the offset
    s = TransfectionSSA()
    ts = s.sample_population(300, rng)
    ts[:, 5] = np.log(20.0)  # transfection at 20 h
    ts[:, 7] = np.log(1e-6)  # (almost) no measurement noise
    y = s.simulate(ts, rng)
    assert np.allclose(y[:, :19], np.log(np.exp(ts[:, 6:7])), atol=1e-4)
    assert (y[:, -1] > y[:, 0]).mean() > 0.95
