"""Distributional and portfolio Kelly sizing on the model's own predictive distribution.

Payoff per MWh X (net of costs) is sampled from the predictive quantiles. With C the
capital at risk per MWh, the growth-optimal fraction maximizes E[log(1 + f X / C)].
The constraint 1 + f X_s / C > 0 for every sample means one sampled price spike caps
the bet, which is the right behavior for heavy-tailed DA-RT spreads.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import ndtr, ndtri


def sample_from_quantiles(q: np.ndarray, taus, u: np.ndarray) -> np.ndarray:
    """Inverse CDF of a piecewise-linear quantile function with exponential tails.

    q [M,Q] monotone; u [S,M] uniforms -> samples [S,M].
    """
    taus = np.asarray(taus, dtype=float)
    q = np.asarray(q, dtype=float)
    S, M = u.shape
    out = np.empty((S, M))
    b_lo = np.maximum((q[:, 1] - q[:, 0]) / np.log(taus[1] / taus[0]), 1e-6)
    b_hi = np.maximum((q[:, -1] - q[:, -2]) / np.log((1 - taus[-2]) / (1 - taus[-1])), 1e-6)
    for j in range(M):
        uj = u[:, j]
        x = np.interp(uj, taus, q[j])
        lo = uj < taus[0]
        hi = uj > taus[-1]
        x[lo] = q[j, 0] - b_lo[j] * np.log(taus[0] / np.maximum(uj[lo], 1e-9))
        x[hi] = q[j, -1] + b_hi[j] * np.log((1 - taus[-1]) / np.maximum(1 - uj[hi], 1e-9))
        out[:, j] = x
    return out


def growth(f: float | np.ndarray, X: np.ndarray, C: float) -> np.ndarray:
    """Expected log growth for fractions f (scalar or grid) on payoff samples X [S]."""
    f = np.atleast_1d(f)
    r = 1.0 + np.outer(f, X) / C
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where((r > 0).all(1), np.log(np.maximum(r, 1e-300)).mean(1), -np.inf)
    return g


def kelly_single(X: np.ndarray, C: float) -> float:
    """Growth-optimal fraction for one trade (0 if the mean payoff is not positive)."""
    if X.mean() <= 0:
        return 0.0
    neg = X[X < 0]
    fmax = 0.999 * C / (-neg.min()) if len(neg) else 10.0
    fmax = min(fmax, 10.0)
    res = minimize_scalar(lambda f: -growth(f, X, C)[0], bounds=(0.0, fmax), method="bounded",
                          options={"xatol": 1e-4})
    return float(max(res.x, 0.0))


def kelly_portfolio(X: np.ndarray, C: float, f0: np.ndarray | None = None, ub: np.ndarray | None = None) -> np.ndarray:
    """Multivariate Kelly: maximize mean_s log(1 + sum_i f_i X_si / C), f >= 0.

    X [S,M] joint payoff samples (same scenario draw across trades).
    """
    S, M = X.shape
    if M == 0:
        return np.zeros(0)
    Xc = X / C
    ub = ub if ub is not None else np.full(M, 5.0)
    f0 = np.clip(f0 if f0 is not None else np.full(M, 0.01), 0, ub)

    def obj(f):
        r = 1.0 + Xc @ f
        if (r <= 1e-6).any():
            bad = np.minimum(r, 1e-6)
            return 1e6 * float(((1e-6 - bad) ** 2).sum()) + 10.0, 1e6 * 2 * (Xc.T @ (bad - 1e-6))
        val = -np.log(r).mean()
        grad = -(Xc / r[:, None]).mean(0)
        return val, grad

    # start from a feasible point
    while obj(f0)[0] >= 10.0 and f0.max() > 1e-8:
        f0 = f0 * 0.5
    res = minimize(obj, f0, jac=True, method="L-BFGS-B", bounds=list(zip(np.zeros(M), ub)),
                   options={"maxiter": 300})
    f = np.clip(res.x, 0, ub)
    if (1.0 + Xc @ f <= 0).any():
        f = f0
    return f


def copula_uniforms(corr: np.ndarray, S: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian-copula uniforms [S,M] with the given correlation matrix."""
    M = corr.shape[0]
    if M == 0:
        return np.zeros((S, 0))
    w, V = np.linalg.eigh(corr)
    Lc = V * np.sqrt(np.clip(w, 1e-8, None))
    z = rng.standard_normal((S, M)) @ Lc.T
    return ndtr(z)


def node_correlation(pit_vals: np.ndarray, shrink: float = 0.2) -> np.ndarray:
    """Cross-node correlation of normal scores from calibration-window PITs [D,H,N]."""
    z = ndtri(np.clip(pit_vals, 1e-4, 1 - 1e-4))
    Z = z.reshape(-1, z.shape[-1])
    Z = Z[np.isfinite(Z).all(1)]
    if len(Z) < 50:
        return np.eye(z.shape[-1])
    c = np.corrcoef(Z.T)
    c = np.nan_to_num(c)
    np.fill_diagonal(c, 1.0)
    return (1 - shrink) * c + shrink * np.eye(len(c))


def candidate_corr(node_corr: np.ndarray, nodes: np.ndarray, hours: np.ndarray, sides: np.ndarray,
                   decay: float) -> np.ndarray:
    """Correlation across candidate trades: node correlation x hour decay, signed by side."""
    c = node_corr[np.ix_(nodes, nodes)] * decay ** np.abs(hours[:, None] - hours[None, :])
    s = np.where(sides > 0, 1.0, -1.0)
    # side flips the payoff sign; the copula is on the spread, so correlation of spreads is kept
    # and sampling is done on the spread, then multiplied by the side. Return spread correlation.
    _ = s
    np.fill_diagonal(c, 1.0)
    return c
