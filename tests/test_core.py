"""Fast unit tests: physics of the OPF, Kelly math, calibration, and the look-ahead guard."""
import numpy as np
import pytest

from gridalpha.sim.grid import build_grid
from gridalpha.sim.market import solve_opf
from gridalpha.trade.kelly import growth, kelly_portfolio, kelly_single, sample_from_quantiles
from gridalpha.train.calibrate import apply_conformal, coverage, fit_conformal

TAUS = [0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98]


def test_ptdf_balances_flows():
    g = build_grid(7)
    P = g.ptdf()
    inj = np.random.default_rng(0).normal(size=g.n)
    inj -= inj.mean()
    flow = P @ inj
    # Kirchhoff: net flow out of every bus equals its injection (slack absorbs the rest)
    A = np.zeros((g.m, g.n))
    A[np.arange(g.m), g.lines[:, 0]] = 1
    A[np.arange(g.m), g.lines[:, 1]] = -1
    net = A.T @ flow
    keep = [i for i in range(g.n) if i != g.slack]
    assert np.allclose(net[keep], inj[keep], atol=1e-6)


def _case(g, rating_scale):
    G = len(g.gens["cap"])
    Mg = np.zeros((g.n, G))
    Mg[g.gens["bus"], np.arange(G)] = 1
    cost = np.where(np.isnan(g.gens["fixed"]), g.gens["hr"] * 3.0 + g.gens["vom"], g.gens["fixed"])
    avail = g.gens["cap"] * np.where(g.gens["fuel"] == "wind", 0.7, np.where(g.gens["fuel"] == "solar", 0.0, 1.0))
    load = g.load_share * 11000
    return solve_opf(g.ptdf(), Mg, avail, cost, load, g.rating * rating_scale, np.ones(g.m, bool), 5000.0)


def test_opf_congestion_comes_only_from_binding_lines():
    g = build_grid(7)
    free = _case(g, 100.0)                      # no line can bind
    assert np.allclose(free.lmp, free.energy, atol=1e-6)
    assert np.allclose(free.shadow, 0.0)
    tight = _case(g, 0.35)                      # some lines must bind
    assert (np.abs(tight.shadow) > 1e-6).any()
    assert np.ptp(tight.lmp) > 1.0
    # LMP decomposition: congestion = -sum_l shadow_l * PTDF_l,i
    P = g.ptdf()
    assert np.allclose(tight.lmp - tight.energy, -(P.T @ tight.shadow), atol=1e-6)


def test_kelly_single_matches_gaussian_rule():
    rng = np.random.default_rng(0)
    X = rng.normal(2.0, 10.0, 20000)
    C = 50.0
    f = kelly_single(X, C)
    assert f == pytest.approx(2.0 / 100.0 * C, rel=0.15)        # f* ~ mu / sigma^2 * C
    assert growth(f, X, C)[0] > growth(2 * f, X, C)[0]
    assert kelly_single(-np.abs(X), C) == 0.0


def test_fat_left_tail_caps_the_bet():
    X = np.full(1000, 3.0)
    X[0] = -500.0
    f = kelly_single(X, 50.0)
    assert f < 50.0 / 500.0


def test_portfolio_kelly_cuts_correlated_bets():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(5000, 1))
    X = 1.0 + 8.0 * np.hstack([z, z + 0.05 * rng.normal(size=(5000, 1))])
    fs = kelly_single(X[:, 0], 50.0)
    fp = kelly_portfolio(X, 50.0)
    assert fp.sum() == pytest.approx(fs, rel=0.25)          # two copies of one bet ~ one bet


def test_quantile_sampler_round_trip():
    q = np.array([[-20, -10, -5, -1, 0, 2, 6, 12, 25]], float)
    u = np.linspace(0.001, 0.999, 4001)[:, None]
    s = sample_from_quantiles(q, TAUS, u)[:, 0]
    for t, v in zip(TAUS, q[0]):
        assert np.mean(s <= v) == pytest.approx(t, abs=0.01)


def test_conformal_fixes_overconfident_bands():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 3, (60, 24, 10))
    base = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0])          # far too narrow
    q = np.broadcast_to(base, y.shape + (9,)).copy()
    assert coverage(q, y, TAUS) < 0.5
    s = fit_conformal(q[:40], y[:40], TAUS)
    assert abs(coverage(apply_conformal(q[40:], s), y[40:], TAUS) - 0.8) < 0.05
