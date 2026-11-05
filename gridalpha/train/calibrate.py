"""Ensembling and conformal recalibration of predictive quantiles."""
from __future__ import annotations

import numpy as np


def ensemble(members: list[np.ndarray]) -> np.ndarray:
    """Quantile averaging (Vincentization) across ensemble members, kept monotone."""
    q = np.mean(np.stack(members), axis=0)
    return np.sort(q, axis=-1)


def fit_conformal(q_val: np.ndarray, y_val: np.ndarray, taus, by_hour: bool = True) -> np.ndarray:
    """Per-quantile shift c_tau so that P(y <= q_tau + c_tau) = tau on the calibration window.

    q_val [D,H,N,Q], y_val [D,H,N]. Returns shifts [H,Q] (or [1,Q]).
    """
    taus = np.asarray(taus)
    groups = range(q_val.shape[1]) if by_hour else [slice(None)]
    shifts = []
    for g in groups:
        qq = q_val[:, g].reshape(-1, len(taus))
        yy = y_val[:, g].reshape(-1)
        m = np.isfinite(yy)
        r = yy[m][:, None] - qq[m]
        shifts.append([np.quantile(r[:, k], t) if m.sum() > 50 else 0.0 for k, t in enumerate(taus)])
    s = np.array(shifts)
    if by_hour:   # light smoothing across hours to avoid overfitting the calibration window
        s = 0.5 * s + 0.25 * np.roll(s, 1, 0) + 0.25 * np.roll(s, -1, 0)
    return s


def apply_conformal(q: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    adj = q + (shifts[None, :, None, :] if shifts.shape[0] == q.shape[1] else shifts[None, None])
    return np.sort(adj, axis=-1)


def coverage(q: np.ndarray, y: np.ndarray, taus, lo: float = 0.1, hi: float = 0.9) -> float:
    taus = list(taus)
    i, j = taus.index(lo), taus.index(hi)
    m = np.isfinite(y)
    return float(((y >= q[..., i]) & (y <= q[..., j]))[m].mean())


def pit(q: np.ndarray, y: np.ndarray, taus) -> np.ndarray:
    """Probability integral transform of y under the piecewise-linear quantile CDF."""
    taus = np.asarray(taus)
    out = np.full(y.shape, np.nan)
    qf = q.reshape(-1, len(taus))
    yf = y.reshape(-1)
    of = out.reshape(-1)
    for i in range(len(yf)):
        if np.isfinite(yf[i]):
            of[i] = np.interp(yf[i], qf[i], taus, left=taus[0] / 2, right=1 - (1 - taus[-1]) / 2)
    return out
