"""Baselines: a naive empirical-quantile forecaster and a LightGBM quantile model.

Both produce the same quantile grid as the GNN so the backtest and Kelly sizing treat
every model identically.
"""
from __future__ import annotations

import numpy as np

from ..data.build import GLOBAL_FEATURES, NODE_FEATURES

LGBM_TAUS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def interp_quantiles(qk: np.ndarray, taus_from, taus_to) -> np.ndarray:
    """Linear interpolation in tau with linear tail extension; qk [...,K] -> [...,Q]."""
    tf, tt = np.asarray(taus_from), np.asarray(taus_to)
    out = np.empty(qk.shape[:-1] + (len(tt),), dtype=np.float32)
    lo_slope = (qk[..., 1] - qk[..., 0]) / (tf[1] - tf[0])
    hi_slope = (qk[..., -1] - qk[..., -2]) / (tf[-1] - tf[-2])
    for j, t in enumerate(tt):
        if t < tf[0]:
            out[..., j] = qk[..., 0] - lo_slope * (tf[0] - t) * 1.5
        elif t > tf[-1]:
            out[..., j] = qk[..., -1] + hi_slope * (t - tf[-1]) * 1.5
        else:
            k = np.searchsorted(tf, t, side="right") - 1
            k = min(k, len(tf) - 2)
            w = (t - tf[k]) / (tf[k + 1] - tf[k])
            out[..., j] = (1 - w) * qk[..., k] + w * qk[..., k + 1]
    return np.sort(out, axis=-1)


def naive_quantiles(F: dict, days: np.ndarray, taus, window: int = 30, lag: int = 2) -> np.ndarray:
    """Empirical quantiles of each node-hour's spread over the last `window` known days."""
    y = F["y_spread"]
    out = np.zeros((len(days), y.shape[1], y.shape[2], len(taus)), np.float32)
    for i, d in enumerate(days):
        hi = d - lag + 1
        lo = max(hi - window, 0)
        if hi - lo < 5:
            continue
        out[i] = np.moveaxis(np.nanquantile(y[lo:hi], taus, axis=0), 0, -1)
    return out


def _neighbor_mean(x: np.ndarray, lines: np.ndarray, n: int) -> np.ndarray:
    A = np.zeros((n, n))
    A[lines[:, 0], lines[:, 1]] = 1
    A[lines[:, 1], lines[:, 0]] = 1
    A = A / np.maximum(A.sum(1, keepdims=True), 1)
    return np.einsum("nm,...m->...n", A, x)


def lgbm_rows(F: dict, days: np.ndarray) -> np.ndarray:
    """Tabular rows [D*H*N, F]: node + global features, history summaries, neighbor aggregates, ids."""
    node = F["node"][days]                                   # [D,H,N,fn]
    glob = F["glob"][days]                                   # [D,H,fg]
    hist = F["hist"][days]                                   # [D,K,H,N,3]
    Dn, H, N, _ = node.shape
    g = np.broadcast_to(glob[:, :, None, :], (Dn, H, N, glob.shape[-1]))
    hs = np.concatenate([hist.mean(1), hist[:, 0], hist.std(1)], axis=-1)       # [D,H,N,9]
    oc = NODE_FEATURES.index("op_cong")
    nf = NODE_FEATURES.index("net_fcst")
    nb = np.stack([_neighbor_mean(node[..., oc], F["lines"], N), _neighbor_mean(node[..., nf], F["lines"], N),
                   _neighbor_mean(hist.mean(1)[..., 0], F["lines"], N)], -1)
    ids = np.broadcast_to(np.arange(N)[None, None, :, None], (Dn, H, N, 1)).astype(np.float32)
    zone = np.broadcast_to(F["zone_of"][None, None, :, None], (Dn, H, N, 1)).astype(np.float32)
    X = np.concatenate([node, g, hs, nb, ids, zone], axis=-1)
    return X.reshape(-1, X.shape[-1]).astype(np.float32)


def lgbm_feature_names() -> list[str]:
    hs = [f"hist_{s}_{k}" for s in ["mean", "last", "std"] for k in ["spread", "rt_cong", "da_cong"]]
    return NODE_FEATURES + GLOBAL_FEATURES + hs + ["nb_op_cong", "nb_net_fcst", "nb_spread"] + ["node_id", "zone"]


def fit_lgbm(F: dict, train_days: np.ndarray, seed: int = 0, max_rows: int = 400_000, rounds: int = 250, log=print):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    X = lgbm_rows(F, train_days)
    y = F["y_spread"][train_days].reshape(-1)
    ok = np.isfinite(y)
    idx = np.where(ok)[0]
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    names = lgbm_feature_names()
    models = []
    for t in LGBM_TAUS:
        m = lgb.LGBMRegressor(objective="quantile", alpha=t, n_estimators=rounds, learning_rate=0.05,
                              num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=2)
        m.fit(X[idx], y[idx], feature_name=names, categorical_feature=["node_id", "zone"])
        models.append(m)
        log(f"    lightgbm tau={t} done")
    return models


def predict_lgbm(models, F: dict, days: np.ndarray, taus) -> np.ndarray:
    X = lgbm_rows(F, days)
    qk = np.stack([m.booster_.predict(X) for m in models], -1)
    Dn, H, N = len(days), F["node"].shape[1], F["node"].shape[2]
    qk = np.sort(qk.reshape(Dn, H, N, -1), -1)
    return interp_quantiles(qk, LGBM_TAUS, taus)
