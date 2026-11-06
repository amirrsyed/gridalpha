"""Turn calibrated predictive distributions into complete, sized trade tickets."""
from __future__ import annotations

import numpy as np

from .kelly import candidate_corr, copula_uniforms, growth, kelly_portfolio, kelly_single, sample_from_quantiles


def ptdf_from_meta(meta: dict) -> np.ndarray | None:
    if meta.get("source") != "sim":
        return None
    lines = np.array([[l["from"], l["to"]] for l in meta["grid"]["lines"]])
    b = 1.0 / np.array(meta["reactance"])
    n = len(meta["grid"]["nodes"])
    A = np.zeros((len(lines), n))
    A[np.arange(len(lines)), lines[:, 0]] = 1
    A[np.arange(len(lines)), lines[:, 1]] = -1
    Bbus = A.T @ (b[:, None] * A)
    slack = 0
    keep = [i for i in range(n) if i != slack]
    Binv = np.zeros((n, n))
    Binv[np.ix_(keep, keep)] = np.linalg.inv(Bbus[np.ix_(keep, keep)])
    return (b[:, None] * A) @ Binv


def line_name(meta: dict, l: int) -> str:
    ln = meta["grid"]["lines"][l]
    if ln.get("name"):
        return ln["name"]
    nodes = meta["grid"]["nodes"]
    return f"{nodes[ln['from']]['name']}–{nodes[ln['to']]['name']}"


def impact_per_mwh(cfg: dict, node_name: str, mw: float) -> float:
    """$/MWh our own order moves the DA price against us: linear in size, smaller at liquid hubs and zones."""
    tc = cfg["trade"]
    k = tc.get("impact_per_mw", 0.0)
    if node_name.startswith(("HB_", "LZ_")):
        k *= tc.get("impact_hub_factor", 0.2)
    return k * mw


def size_day(q: np.ndarray, da_hat: np.ndarray, members_med: np.ndarray | None, cfg: dict, node_corr: np.ndarray,
             zone_of: np.ndarray, capital: float, rng: np.random.Generator, kappas=(1.0, 0.5, 0.25),
             node_names: list[str] | None = None) -> dict:
    """Select candidates for one day and size them.

    q [H,N,Q] calibrated spread quantiles (RT - DA, $/MWh); da_hat [H,N].
    Returns candidate arrays and MW for each kelly fraction in `kappas`.
    """
    tc = cfg["trade"]
    taus = np.asarray(cfg["model"]["quantiles"])
    H, N, Q = q.shape
    mid = list(taus).index(0.5)
    # quick screen with a small independent sample per node-hour
    u = rng.random((400, H * N))
    S = sample_from_quantiles(q.reshape(-1, Q), taus, u)
    if tc.get("edge_estimator", "mean") == "body":
        # direction and edge from the body of the forecast, where it is stable; the extreme tails are
        # learned from rare spikes and are noisy, so they only shrink or block positions via Kelly + risk caps
        lo, hi = tc.get("body_quantiles", [0.05, 0.95])
        qq = q.reshape(-1, Q)
        body = np.clip(S, qq[:, list(taus).index(lo)],
                       qq[:, list(taus).index(hi)]).mean(0)
        side = np.where(body >= 0, 1, -1).reshape(H, N)
        Xs = S * side.reshape(-1)[None, :] - tc["cost_per_mwh"]
        mean_edge = body * side.reshape(-1) - tc["cost_per_mwh"]
    else:
        side = np.where(q[..., mid] >= 0, 1, -1)                  # +1 DEC, -1 INC
        Xs = S * side.reshape(-1)[None, :] - tc["cost_per_mwh"]
        mean_edge = Xs.mean(0)
    p_profit = (Xs > 0).mean(0)
    ok = (mean_edge > tc["min_edge"]) & (p_profit > tc["min_p_profit"])
    cand = np.where(ok)[0]
    if len(cand) > 2 * tc["max_tickets_per_day"]:
        score = mean_edge[cand] / (Xs[:, cand].std(0) + 1e-6)
        cand = cand[np.argsort(-score)[: 2 * tc["max_tickets_per_day"]]]
    res = {"cand": cand, "hours": cand // N, "nodes": cand % N, "side": side.reshape(-1)[cand]}
    M = len(cand)
    if M == 0:
        res.update({"mw": {k: np.zeros(0) for k in kappas}, "f_star": np.zeros(0), "f_single": np.zeros(0),
                    "X": np.zeros((tc["samples"], 0)), "mean_edge": np.zeros(0), "p_profit": np.zeros(0)})
        return res
    hours, nodes, sides = res["hours"], res["nodes"], res["side"]
    corr = candidate_corr(node_corr, nodes, hours, sides, tc["hour_corr_decay"])
    U = copula_uniforms(corr, tc["samples"], rng)
    spread_s = sample_from_quantiles(q[hours, nodes], taus, U)
    X = spread_s * sides[None, :] - tc["cost_per_mwh"]
    C = tc["risk_per_mwh"]
    f_single = np.array([kelly_single(X[:, i], C) for i in range(M)])
    f_star = kelly_portfolio(X, C, f0=f_single / max(M, 1), ub=np.maximum(f_single, 1e-6) * 1.5)
    full_edge = X.mean(0)
    res.update({"f_star": f_star, "f_single": f_single, "X": X,
                "mean_edge": np.minimum(full_edge, mean_edge[cand]) if tc.get("edge_estimator") == "body" else full_edge,
                "p_profit": (X > 0).mean(0), "corr": corr})
    caps = capital if isinstance(capital, dict) else {k: capital for k in kappas}
    names = node_names or [""] * q.shape[1]
    res["impact_k"] = np.array([impact_per_mwh(cfg, names[n], 1.0) for n in nodes])
    res["mw"] = {k: apply_sizing(res, k, caps[k], cfg, zone_of) for k in kappas}
    return res


def apply_sizing(sized: dict, kappa: float, capital: float, cfg: dict, zone_of: np.ndarray) -> np.ndarray:
    """Fractional Kelly -> MW, then node-hour, zone-hour and daily-VaR caps."""
    tc = cfg["trade"]
    C = tc["risk_per_mwh"]
    if len(sized["f_star"]) == 0:
        return np.zeros(0)
    m = kappa * sized["f_star"] * capital / C
    m = np.minimum(m, tc["max_mw_node_hour"])
    # price impact: with edge e and impact k per MW, profit m(e - k m) peaks at m = e / 2k
    kimp = sized.get("impact_k")
    if kimp is not None:
        m = np.where(kimp > 0, np.minimum(m, np.maximum(sized["mean_edge"], 0) / (2 * np.maximum(kimp, 1e-9))), m)
    z = zone_of[sized["nodes"]]
    hours = sized["hours"]
    for key in set(zip(z.tolist(), hours.tolist())):
        sel = (z == key[0]) & (hours == key[1])
        tot = m[sel].sum()
        if tot > tc["max_mw_zone_hour"]:
            m[sel] *= tc["max_mw_zone_hour"] / tot
    # collateral: ERCOT holds credit against every MWh bid or offered; total posted <= available capital
    coll = tc.get("collateral_per_mwh", 0.0)
    if coll > 0 and m.sum() * coll > tc.get("collateral_share", 1.0) * capital:
        m *= tc.get("collateral_share", 1.0) * capital / (m.sum() * coll)
    pnl = sized["X"] @ m
    var95 = -np.quantile(pnl, 0.05)
    cap = tc["max_daily_var95"] * capital
    if var95 > cap > 0:
        m *= cap / var95
    m[m < 0.5] = 0.0
    return np.round(m, 1)


def make_tickets(day: str, sized: dict, q: np.ndarray, da_hat: np.ndarray, bind: np.ndarray, attrib: np.ndarray | None,
                 group_names: list[str], meta: dict, cfg: dict, kappa: float, model_version: str,
                 ptdf: np.ndarray | None, members_med: np.ndarray | None = None) -> list[dict]:
    tc = cfg["trade"]
    taus = list(cfg["model"]["quantiles"])
    mid = taus.index(0.5)
    nodes_meta = meta["grid"]["nodes"]
    lines = meta["grid"]["lines"]
    out = []
    mw = sized["mw"].get(kappa, np.zeros(0))
    order = np.argsort(-(mw * sized["mean_edge"])) if len(mw) else []
    for rank, i in enumerate(order):
        if mw[i] <= 0 or len(out) >= tc["max_tickets_per_day"]:
            continue
        h, n, s = int(sized["hours"][i]), int(sized["nodes"][i]), int(sized["side"][i])
        qq = q[h, n]
        X = sized["X"][:, i] - sized.get("impact_k", np.zeros(len(mw)))[i] * mw[i]
        exp_da = float(da_hat[h, n])
        exp_rt = exp_da + float(qq[mid])
        edge_need = tc["cost_per_mwh"] + tc["min_edge"]
        limit = exp_rt - edge_need if s > 0 else exp_rt + edge_need
        pnl = X * mw[i]
        # constraint exposure: PTDF x binding probability (sim) or incident constraints (real)
        if ptdf is not None:
            expo = ptdf[:, n] * bind[h]
        else:
            expo = np.array([bind[h, l] if n in (ln["from"], ln["to"]) else 0.0 for l, ln in enumerate(lines)])
        top_l = int(np.argmax(np.abs(expo)))
        drivers = []
        if attrib is not None:
            a = attrib[h, n] * s                       # positive = supports this side
            for g in np.argsort(-np.abs(a))[:3]:
                drivers.append({"group": group_names[g], "usd": round(float(a[g]), 2)})
        fs = float(sized["f_single"][i])
        curve_f = np.linspace(0, max(2.0 * fs, 1e-3), 25)
        out.append({
            "id": f"{day}-HE{h + 1:02d}-{nodes_meta[n]['name']}",
            "market": "ERCOT DAM (simulated)" if meta.get("source") == "sim" else "ERCOT DAM",
            "instrument": "Virtual DEC (buy DA, sell RT)" if s > 0 else "Virtual INC (sell DA, buy RT)",
            "side": "DEC" if s > 0 else "INC",
            "date": day, "he": h + 1, "node": nodes_meta[n]["name"], "node_id": n, "zone": nodes_meta[n]["zone"],
            "dc_node": bool(nodes_meta[n].get("dc")),
            "mw": float(mw[i]), "price_limit": round(float(limit), 2),
            "exp_da": round(exp_da, 2), "exp_rt": round(exp_rt, 2),
            "q": [round(float(x), 2) for x in qq], "mean_edge": round(float(X.mean()), 2),
            "p_profit": round(float((X > 0).mean()), 3),
            "kelly_single": round(fs, 4), "kelly_portfolio": round(float(sized["f_star"][i]), 4),
            "kelly_applied": round(float(kappa * sized["f_star"][i]), 4),
            "var95_usd": round(float(-np.quantile(pnl, 0.05)), 0),
            "loss_q01_usd": round(float(-np.quantile(pnl, 0.01)), 0),
            "exp_pnl_usd": round(float(pnl.mean()), 0),
            "constraint": line_name(meta, top_l), "constraint_exposure": round(float(expo[top_l]), 3),
            "drivers": drivers,
            "ens_disagreement": round(float(members_med[h, n].std()), 2) if members_med is not None else None,
            "growth_curve": [[round(float(f), 4), round(float(g), 6)] for f, g in zip(curve_f, growth(curve_f, X, tc["risk_per_mwh"]))],
            "valid_until": f"{day} -1d 10:00 CT",
            "model_version": model_version,
        })
    return out
