"""Package everything the Blueprint.js dashboard shows into one ``signals.json``."""
from __future__ import annotations

import datetime as dt
import json
import shutil

import numpy as np

from ..train.calibrate import pit
from .store import load_assembled

TRADE_FIELDS = ["date", "he", "n", "side", "mw", "limit", "exp_da", "exp_rt", "q", "edge", "pp", "kf", "ks",
                "cleared", "da", "rt", "pnl", "theme"]


def _r(a, nd=2):
    return np.round(np.nan_to_num(np.asarray(a, dtype=float)), nd).tolist()


def export_dashboard(ap) -> dict:
    cfg, F, A = ap.cfg, ap.F, ap.A
    meta = ap.meta()
    taus = cfg["model"]["quantiles"]
    mid = taus.index(0.5)
    out = {"meta": {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": meta.get("source"), "profile": cfg.get("_profile", "default"),
        "quantiles": taus, "kelly_fraction": cfg["trade"]["kelly_fraction"], "kappas": [1.0, 0.5, 0.25],
        "capital0": cfg["trade"]["capital"], "cost_per_mwh": cfg["trade"]["cost_per_mwh"],
        "min_edge": cfg["trade"]["min_edge"], "min_p_profit": cfg["trade"]["min_p_profit"],
        "risk_per_mwh": cfg["trade"]["risk_per_mwh"], "max_mw_node_hour": cfg["trade"]["max_mw_node_hour"],
        "max_daily_var95": cfg["trade"]["max_daily_var95"], "ensemble_seeds": cfg["train"]["ensemble_seeds"],
        "shadow": bool(ap.state["stages"].get("backtest", {}).get("shadow")),
        "shadow_failed": (ap.state["stages"].get("backtest", {}).get("gate") or {}).get("failed", []),
        "diagnostic": [
            {"stage": s, "failed": (ap.state["stages"][s].get("gate") or {}).get("failed", [])}
            for s, st in ap.state["stages"].items() if st.get("diagnostic_override") and not st.get("shadow")
        ],
    }, "grid": meta["grid"], "zones": meta["zones"]}

    live = A / "signals" / "latest.json"
    if live.exists():
        out["live"] = json.loads(live.read_text())
        out["meta"]["model_version"] = out["live"]["model_version"]

    # backtest
    bt = {"models": {}}
    for m in ["gnn", "lgbm", "naive"]:
        p = A / "backtest" / f"{m}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        bt["dates"] = r["dates"]
        bt["models"][m] = {"metrics": r["metrics"], "daily": r["daily"]}
        if m == "gnn":
            bt["trade_fields"] = TRADE_FIELDS
            bt["trades"] = [[t[f] for f in TRADE_FIELDS] for t in r["trades"]]
    out["backtest"] = bt

    # forecast vs realized maps (last 30 test days) + calibration
    g = load_assembled(A / "preds" / "gnn_assembled")
    days = g["days"]
    last = days[-31:-1] if len(days) > 31 else days[:-1]
    sel = np.isin(days, last)
    out["maps"] = {
        "dates": [str(F["dates"][d]) for d in last],
        "pred": [_r(x, 1) for x in g["q_cal"][sel][..., mid]],
        "real": [_r(x, 1) for x in F["y_spread"][last]],
        "bind": [_r(x, 2) for x in g["bind"][sel]],
    }
    if "attrib" in g:
        out["maps"]["attrib_names"] = g.get("attrib_names")
    cal = {"taus": taus}
    y = F["y_spread"][days[:-1]]
    for m in ["gnn", "lgbm", "naive"]:
        p = A / "preds" / f"{m}_assembled.npz"
        if not p.exists():
            continue
        r = load_assembled(A / "preds" / f"{m}_assembled")
        qc = r["q_cal"][:-1]
        cal[m] = [float(np.nanmean(y <= qc[..., k])) for k in range(len(taus))]
        if m == "gnn":
            qr = r["q_raw"][:-1]
            cal["gnn_raw"] = [float(np.nanmean(y <= qr[..., k])) for k in range(len(taus))]
            sub = np.random.default_rng(0).choice(y.size, size=min(40000, y.size), replace=False)
            pv = pit(qc.reshape(-1, len(taus))[sub][:, None, None, :], y.reshape(-1)[sub][:, None, None], taus).ravel()
            cal["pit_hist"] = np.histogram(pv[np.isfinite(pv)], bins=20, range=(0, 1))[0].tolist()
    out["calibration"] = cal

    # model health
    model = {"folds": {}, "curves": g.get("curves"), "pipeline": {}}
    for m in ["gnn", "lgbm", "naive"]:
        p = A / "preds" / f"{m}_assembled.json"
        if p.exists():
            r = json.loads(p.read_text())
            model["folds"][m] = r["stats"]
            model[f"{m}_overall"] = r["overall"]
    for s, st in ap.state["stages"].items():
        model["pipeline"][s] = {k: st.get(k) for k in ["status", "ended", "secs", "gate", "error"]}
    model["pipeline"]["publish"] = {"status": "passed", "ended": out["meta"]["generated"], "secs": None, "gate": None,
                                    "error": None}   # this export is the publish stage itself
    import torch
    pre = []
    for f in ap.folds():
        pth = A / "models" / f"fold{f['k']}_pretrain.pt"
        if pth.exists():
            pre.append(torch.load(pth, map_location="cpu")["info"]["pretrain_loss"])
    model["pretrain_curves"] = pre
    if "adaptive_adj" in g:
        adj = g["adaptive_adj"]
        edges = []
        for i in range(adj.shape[0]):
            for j in np.argsort(-adj[i])[:2]:
                if i != j and adj[i, j] > 0.05:
                    edges.append([int(i), int(j), round(float(adj[i, j]), 3)])
        model["learned_edges"] = edges
    out["model"] = model

    site = A / "site"
    site.mkdir(exist_ok=True)
    path = site / "signals.json"
    path.write_text(json.dumps(out, separators=(",", ":"), default=float))
    dash = ap.root / "dashboard" / "public"
    if dash.parent.exists():
        dash.mkdir(exist_ok=True)
        shutil.copy(path, dash / "signals.json")
    return {"bytes": path.stat().st_size, "trades": len(bt.get("trades", [])),
            "live_tickets": len(out.get("live", {}).get("tickets", []))}
