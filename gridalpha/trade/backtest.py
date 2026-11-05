"""Walk-forward trading backtest and live signal generation.

For every out-of-sample day each model's calibrated distribution is turned into sized
tickets exactly as in production. A ticket's virtual clears only if the realized DA
price is inside its price limit; P&L is side x (RT - DA) - costs, per MW. Capital
compounds daily, separately for each Kelly fraction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..export.store import load_assembled
from ..train.calibrate import apply_conformal, ensemble, fit_conformal
from ..train.train import build_model, device_of, predict, attributions
from ..train.walk_forward import plan_folds
from .kelly import node_correlation
from .tickets import make_tickets, ptdf_from_meta, size_day

KAPPAS = (1.0, 0.5, 0.25)
THEMES = ["Data-center pocket", "Evening ramp", "Negative price", "Scarcity", "Other"]


def _dc_neighborhood(meta: dict) -> set:
    dc = {n["id"] for n in meta["grid"]["nodes"] if n.get("dc")}
    near = set(dc)
    for ln in meta["grid"]["lines"]:
        if ln["from"] in dc or ln["to"] in dc:
            near.update([ln["from"], ln["to"]])
    return near


def theme_of(n: int, h: int, da_act: float, rt_act: float, dc_near: set) -> str:
    if rt_act > 300 or da_act > 300:
        return "Scarcity"
    if da_act < 0 or rt_act < 0:
        return "Negative price"
    if n in dc_near:
        return "Data-center pocket"
    if 17 <= h <= 20:
        return "Evening ramp"
    return "Other"


def _metrics(daily: np.ndarray, capital0: float, trades: list, compound: bool = False) -> dict:
    eq = capital0 + np.cumsum(daily)
    peak = np.maximum.accumulate(np.concatenate([[capital0], eq]))[1:]
    dd = float(np.max((peak - eq) / peak)) if len(eq) else 0.0
    rets = daily / (np.concatenate([[capital0], eq[:-1]]) if compound else capital0)
    sd = rets.std()
    cleared = [t for t in trades if t["cleared"]]
    return {
        "pnl": float(daily.sum()), "return": float(daily.sum() / capital0),
        "sharpe": float(rets.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0,
        "max_drawdown": dd, "worst_day": float(daily.min()) if len(daily) else 0.0,
        "n_tickets": len(trades), "n_cleared": len(cleared),
        "hit_rate": float(np.mean([t["pnl"] > 0 for t in cleared])) if cleared else 0.0,
        "mwh": float(sum(t["mw"] for t in cleared)),
        "days_traded": int((daily != 0).sum()),
    }


def backtest_model(ap, name: str, live_day: int) -> dict:
    cfg, F = ap.cfg, ap.F
    meta = ap.meta()
    tc = cfg["trade"]
    taus = cfg["model"]["quantiles"]
    mid = taus.index(0.5)
    res = load_assembled(ap.A / "preds" / f"{name}_assembled")
    gnn = res if name == "gnn" else load_assembled(ap.A / "preds" / "gnn_assembled")
    zone_of = F["zone_of"]
    corr = [node_correlation(p) for p in res["pit_val"]]
    rng = np.random.default_rng(cfg["project"]["seed"])
    dc_near = _dc_neighborhood(meta)
    cap = {k: tc["capital"] for k in KAPPAS}
    daily = {k: [] for k in KAPPAS}
    trades = {k: [] for k in KAPPAS}
    counts = {k: {"n_tickets": 0, "n_cleared": 0, "wins": 0, "mwh": 0.0, "at_cap": 0} for k in KAPPAS}
    dates = []
    for i, d in enumerate(res["days"]):
        if d == live_day:
            continue
        q = res["q_cal"][i]
        da_hat = gnn["da"][i]
        sized = size_day(q, da_hat, None, cfg, corr[int(res["fold"][i])], zone_of, dict(cap), rng, KAPPAS)
        da_act, rt_act = F["y_da"][d], F["y_da"][d] + F["y_spread"][d]
        dates.append(str(F["dates"][d]))
        for k in KAPPAS:
            mw = sized["mw"][k]
            pnl_day = 0.0
            for j in np.where(mw > 0)[0]:
                h, n, s = int(sized["hours"][j]), int(sized["nodes"][j]), int(sized["side"][j])
                exp_rt = da_hat[h, n] + q[h, n, mid]
                need = tc["cost_per_mwh"] + tc["min_edge"]
                limit = exp_rt - need if s > 0 else exp_rt + need
                dA, rT = float(da_act[h, n]), float(rt_act[h, n])
                cleared = (dA <= limit) if s > 0 else (dA >= limit)
                pnl = (s * (rT - dA) - tc["cost_per_mwh"]) * mw[j] if cleared else 0.0
                pnl_day += pnl
                c = counts[k]
                c["n_tickets"] += 1
                c["at_cap"] += int(mw[j] >= tc["max_mw_node_hour"] - 0.05)
                if cleared:
                    c["n_cleared"] += 1
                    c["wins"] += int(pnl > 0)
                    c["mwh"] += float(mw[j])
                if name == "gnn" or k == tc["kelly_fraction"]:
                    trades[k].append({
                        "date": dates[-1], "he": h + 1, "n": n, "side": "DEC" if s > 0 else "INC", "mw": float(mw[j]),
                        "limit": round(float(limit), 2), "exp_da": round(float(da_hat[h, n]), 2),
                        "exp_rt": round(float(exp_rt), 2), "q": [round(float(x), 2) for x in q[h, n]],
                        "edge": round(float(sized["mean_edge"][j]), 2), "pp": round(float(sized["p_profit"][j]), 3),
                        "kf": round(float(k * sized["f_star"][j]), 4), "ks": round(float(sized["f_single"][j]), 4),
                        "cleared": bool(cleared), "da": round(dA, 2), "rt": round(rT, 2), "pnl": round(float(pnl), 2),
                        "theme": theme_of(n, h, dA, rT, dc_near),
                    })
            daily[k].append(pnl_day)
            if tc.get("compound", False):
                cap[k] += pnl_day
    out = {}
    for k in KAPPAS:
        dv = np.array(daily[k])
        m = _metrics(dv, tc["capital"], trades[k], tc.get("compound", False))
        c = counts[k]
        m.update({"n_tickets": c["n_tickets"], "n_cleared": c["n_cleared"], "mwh": c["mwh"],
                  "hit_rate": c["wins"] / c["n_cleared"] if c["n_cleared"] else 0.0,
                  "share_at_cap": c["at_cap"] / c["n_tickets"] if c["n_tickets"] else 0.0})
        by_theme = {t: float(sum(x["pnl"] for x in trades[k] if x["theme"] == t)) for t in THEMES}
        out[str(k)] = m | {"by_theme": by_theme}
    return {"metrics": out, "daily": {str(k): [round(x, 2) for x in daily[k]] for k in KAPPAS}, "dates": dates,
            "trades": trades[tc["kelly_fraction"]]}


def run_backtests(ap) -> dict:
    live_day = int(ap.F["node"].shape[0] - 1)
    out_dir = ap.A / "backtest"
    out_dir.mkdir(exist_ok=True)
    summary = {}
    for name in ["gnn", "lgbm", "naive"]:
        ap.log(f"  backtest: {name}")
        r = backtest_model(ap, name, live_day)
        (out_dir / f"{name}.json").write_text(json.dumps(r, default=float))
        summary[name] = r["metrics"]
    return summary


# ---------------------------------------------------------------------------------------------
# Live signals: predict the newest operating day with the latest fold's ensemble
# ---------------------------------------------------------------------------------------------

def live_signals(ap) -> dict:
    cfg, F = ap.cfg, ap.F
    meta = ap.meta()
    taus = cfg["model"]["quantiles"]
    mid = taus.index(0.5)
    folds = plan_folds(F, cfg)
    last = folds[-1]
    day = np.array([F["node"].shape[0] - 1])
    dev = device_of(cfg)
    qs, das, binds, val_qs, meds = [], [], [], [], []
    att, names = None, None
    for s in cfg["train"]["ensemble_seeds"]:
        ck = torch.load(ap.A / "models" / f"gnn_fold{last['k']}_seed{s}.pt", map_location=dev)
        model = build_model(F, cfg).to(dev)
        model.load_state_dict(ck["state"])
        p = predict(model, F, day, cfg)
        qs.append(p["q"])
        das.append(p["da"])
        binds.append(p["bind"])
        meds.append(p["q"][..., mid])
        val_qs.append(np.load(ap.A / "preds" / f"gnn_fold{last['k']}_seed{s}.npz")["val_q"])
        if att is None:
            att, names = attributions(model, F, day, cfg, last["train"])
    shifts = fit_conformal(ensemble(val_qs), F["y_spread"][last["val"]], taus)
    q = apply_conformal(ensemble(qs), shifts)[0]
    da_hat = np.mean(das, 0)[0]
    bind = np.mean(binds, 0)[0]
    members_med = np.stack(meds, -1)[0]
    gnn = load_assembled(ap.A / "preds" / "gnn_assembled")
    corr = node_correlation(gnn["pit_val"][-1])
    bt_path = ap.A / "backtest" / "gnn.json"
    capital = cfg["trade"]["capital"]
    k = cfg["trade"]["kelly_fraction"]
    if bt_path.exists() and cfg["trade"].get("compound", False):
        bt = json.loads(bt_path.read_text())
        capital += float(np.sum(bt["daily"][str(k)]))
    rng = np.random.default_rng(cfg["project"]["seed"] + 1)
    sized = size_day(q, da_hat, members_med, cfg, corr, F["zone_of"], capital, rng, (k,))
    date = str(F["dates"][day[0]])
    version = f"gnn-{date}-ens{len(cfg['train']['ensemble_seeds'])}-{cfg.get('_profile', 'default')}"
    tickets = make_tickets(date, sized, q, da_hat, bind, att[0], names, meta, cfg, k, version, ptdf_from_meta(meta),
                           members_med)
    out_dir = ap.A / "signals"
    out_dir.mkdir(exist_ok=True)
    payload = {"operating_day": date, "diagnostic": any(st.get("diagnostic_override") for st in ap.state["stages"].values()), "capital": capital, "kelly_fraction": k, "model_version": version,
               "tickets": tickets,
               "grid_forecast": {"median_spread": np.round(q[..., mid], 2).tolist(),
                                 "q10": np.round(q[..., taus.index(0.1)], 2).tolist(),
                                 "q90": np.round(q[..., taus.index(0.9)], 2).tolist(),
                                 "da_hat": np.round(da_hat, 2).tolist(), "bind": np.round(bind, 3).tolist()}}
    (out_dir / "latest.json").write_text(json.dumps(payload, default=float))
    import csv
    with open(out_dir / "latest.csv", "w", newline="") as fh:
        cols = ["id", "side", "node", "zone", "he", "mw", "price_limit", "exp_da", "exp_rt", "mean_edge", "p_profit",
                "kelly_applied", "var95_usd", "constraint"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(tickets)
    return {"operating_day": date, "n_tickets": len(tickets), "total_mw": float(sum(t["mw"] for t in tickets)),
            "exp_pnl_usd": float(sum(t["exp_pnl_usd"] for t in tickets)), "capital": capital}
