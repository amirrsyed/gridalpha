"""Point-in-time feature tensors for the GNN and the baselines.

For target operating day d (bids due d-1 10:00):
  * forecasts for d (published d-1) are allowed,
  * realized prices / constraints are allowed only through day d-2,
  * static network data and announced (planned) outages are allowed.
``leak_test`` checks this by perturbing everything realized after d-2 and confirming
the features of day d do not move.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NODE_FEATURES = [
    "load_fcst", "wind_fcst", "solar_fcst", "net_fcst", "temp_fcst", "dc_mw", "dc_growth_60d",
    "thermal_avail", "planned_gen_out", "op_cong", "op_lmp", "cap_wind", "cap_solar", "cap_gas", "cap_base",
]
GLOBAL_FEATURES = [
    "sys_load_fcst", "sys_wind_fcst", "sys_solar_fcst", "margin_fcst", "gas", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "weekend", "lines_out", "net_load_ramp",
]
EDGE_FEATURES = ["rating", "planned_out", "reactance", "is_tie", "bind_freq_30d", "shadow_7d", "op_shadow"]
HIST_FEATURES = ["spread", "rt_cong", "da_cong"]

# feature groups used for occlusion attribution ("drivers" on each ticket)
DRIVER_GROUPS = {
    "Load forecast": (["load_fcst", "temp_fcst"], ["sys_load_fcst"]),
    "Wind forecast": (["wind_fcst"], ["sys_wind_fcst"]),
    "Solar / evening ramp": (["solar_fcst"], ["sys_solar_fcst", "net_load_ramp"]),
    "Data-center load": (["dc_mw", "dc_growth_60d"], []),
    "Outages": (["planned_gen_out", "thermal_avail"], ["lines_out"]),
    "Own OPF view": (["op_cong", "op_lmp"], []),
    "Recent spreads": ([], []),          # history tensor
    "Scarcity / gas": ([], ["margin_fcst", "gas"]),
}


def asinh_scale(x, s=10.0):
    return np.arcsinh(np.asarray(x) / s).astype(np.float32)


def load_raw(raw_dir: Path) -> tuple[dict, dict]:
    raw = dict(np.load(raw_dir / "market.npz"))
    meta = json.loads((raw_dir / "meta.json").read_text())
    return raw, meta


def _bus_caps(meta: dict, n: int) -> dict:
    g = meta["gens"]
    bus = np.array(g["bus"], dtype=int)
    fuel = np.array(g["fuel"], dtype=str)
    cap = np.array(g["cap"], dtype=float)
    out = {}
    for name, fuels in {"wind": ["wind"], "solar": ["solar"], "gas": ["gas_cc", "gas_ct"],
                        "base": ["nuclear", "coal"]}.items():
        m = np.isin(fuel, fuels)
        out[name] = np.bincount(bus[m], weights=cap[m], minlength=n)
    out["thermal_mask"] = np.isin(fuel, ["gas_cc", "gas_ct", "nuclear", "coal"])
    return out


def build_features(raw: dict, meta: dict, cfg: dict) -> dict:
    fc = cfg["features"]
    D, H, N = raw["da_lmp"].shape
    L = raw["rt_shadow"].shape[2]
    zone_of = np.array(meta["zone_of"])
    lag = fc["history_lag_days"]
    nh = fc["history_days"]
    first = max(lag + nh, fc["bind_freq_window"] + lag)
    caps = _bus_caps(meta, N)
    gbus = np.array(meta["gens"]["bus"], dtype=int)
    gcap = np.array(meta["gens"]["cap"], dtype=float)

    da_cong = raw["da_lmp"] - raw["da_energy"][..., None]
    rt_cong = raw["rt_lmp"] - raw["rt_energy"][..., None]
    spread = raw["rt_lmp"] - raw["da_lmp"]

    # --- node features [D,H,N,F]
    wind_bus = caps["wind"][None, None, :] * raw["wind_fcst_cf"][:, :, zone_of]
    solar_bus = caps["solar"][None, None, :] * raw["solar_fcst_cf"][:, :, zone_of]
    load_f = raw["load_bus_fcst"]
    temp_bus = raw["temp_fcst"][:, :, zone_of]
    dc = raw.get("dc_mw", np.zeros((D, N)))
    dc_g = dc - np.concatenate([np.repeat(dc[:1], 60, 0), dc[:-60]])[:D]
    if "planned_gen" in raw:
        pg = raw["planned_gen"].astype(float) * gcap[None, :]
        pg_bus = np.stack([np.bincount(gbus, weights=pg[d], minlength=N) for d in range(D)])
        th = np.bincount(gbus[caps["thermal_mask"]], weights=gcap[caps["thermal_mask"]], minlength=N)
        th_av = th[None, :] - pg_bus
    else:
        pg_bus = np.zeros((D, N))
        th_av = np.repeat((caps["gas"] + caps["base"])[None, :], D, 0)
    if "op_lmp" in raw:
        op_c = raw["op_lmp"] - raw["op_energy"][..., None]
        op_l = raw["op_lmp"]
    else:
        op_c = np.zeros((D, H, N))
        op_l = np.zeros((D, H, N))
    rep = lambda a: np.repeat(a[:, None, :], H, axis=1)
    stat = lambda a: np.broadcast_to(a[None, None, :], (D, H, N))
    node = np.stack([
        load_f / 1000.0, wind_bus / 1000.0, solar_bus / 1000.0, (wind_bus + solar_bus - load_f) / 1000.0,
        temp_bus / 30.0, rep(dc) / 1000.0, rep(dc_g) / 1000.0, rep(th_av) / 1000.0, rep(pg_bus) / 1000.0,
        asinh_scale(op_c, 10.0), asinh_scale(op_l, 30.0),
        stat(caps["wind"]) / 1000.0, stat(caps["solar"]) / 1000.0, stat(caps["gas"]) / 1000.0, stat(caps["base"]) / 1000.0,
    ], axis=-1).astype(np.float32)

    # --- global features [D,H,G]
    dates = np.array(meta["dates"], dtype="datetime64[D]")
    dow = ((dates.astype("int64") - 4) % 7).astype(float)          # 1970-01-01 was a Thursday
    doy = (dates - dates.astype("datetime64[Y]")).astype(int).astype(float)
    hrs = np.arange(H, dtype=float)
    sys_load = load_f.sum(-1)
    net = sys_load - wind_bus.sum(-1) - solar_bus.sum(-1)
    ramp = np.concatenate([np.diff(net, axis=1), np.zeros((D, 1))], axis=1)
    margin = raw.get("da_margin", np.zeros((D, H)))
    hb = lambda a: np.broadcast_to(a[None, :], (D, H))
    db = lambda a: np.broadcast_to(a[:, None], (D, H))
    glob = np.stack([
        sys_load / 10000.0, wind_bus.sum(-1) / 5000.0, solar_bus.sum(-1) / 5000.0, margin,
        db(raw["gas"]) / 3.0, hb(np.sin(2 * np.pi * hrs / 24)), hb(np.cos(2 * np.pi * hrs / 24)),
        db(np.sin(2 * np.pi * dow / 7)), db(np.cos(2 * np.pi * dow / 7)),
        db(np.sin(2 * np.pi * doy / 365)), db(np.cos(2 * np.pi * doy / 365)), db((dow >= 5).astype(float)),
        db(raw["planned_line"].sum(-1) / 5.0), ramp / 1000.0,
    ], axis=-1).astype(np.float32)

    # --- history [D, nh, H, N, 3] using days d-lag .. d-lag-nh+1
    hist_src = np.stack([asinh_scale(spread), asinh_scale(rt_cong), asinh_scale(da_cong)], axis=-1)
    hist = np.zeros((D, nh, H, N, 3), np.float32)
    for k in range(nh):
        src = np.arange(D) - lag - k
        ok = src >= 0
        hist[ok, k] = hist_src[src[ok]]

    # --- edge features [D,H,L,F]
    reac = np.array(meta["reactance"], dtype=float)
    rating = np.array([ln["rating"] for ln in meta["grid"]["lines"]], dtype=float)
    tie = np.array([ln["tie"] for ln in meta["grid"]["lines"]], dtype=float)
    bind = (np.abs(raw["rt_shadow"]) > 0.01).astype(float)       # [D,H,L]
    bind_day = bind.mean(1)
    shadow_day = np.abs(raw["rt_shadow"]).mean(1)
    w = fc["bind_freq_window"]
    cs = np.concatenate([np.zeros((1, L)), np.cumsum(bind_day, 0)])
    cs2 = np.concatenate([np.zeros((1, L)), np.cumsum(shadow_day, 0)])
    bf = np.zeros((D, L))
    sh7 = np.zeros((D, L))
    for d in range(D):
        hi = d - lag + 1                                   # through day d-lag inclusive
        if hi <= 0:
            continue
        bf[d] = (cs[hi] - cs[max(hi - w, 0)]) / max(min(w, hi), 1)
        sh7[d] = (cs2[hi] - cs2[max(hi - 7, 0)]) / max(min(7, hi), 1)
    op_sh = raw.get("op_shadow", np.zeros((D, H, L)))
    eb = lambda a: np.broadcast_to(a[None, None, :], (D, H, L))
    edge = np.stack([
        eb(rating / 1000.0), np.repeat(raw["planned_line"][:, None, :].astype(float), H, 1), eb(reac / reac.mean()),
        eb(tie), np.repeat(bf[:, None, :], H, 1), np.repeat(asinh_scale(sh7, 10)[:, None, :], H, 1),
        asinh_scale(op_sh, 10.0),
    ], axis=-1).astype(np.float32)

    labels = {
        "y_spread": spread.astype(np.float32),
        "y_da": raw["da_lmp"].astype(np.float32),
        "y_rtcong": rt_cong.astype(np.float32),
        "y_bind": bind.astype(np.float32),
    }
    if "rt_inj" in raw:
        labels["y_inj"] = raw["rt_inj"].astype(np.float32)

    lines = np.array([[ln["from"], ln["to"]] for ln in meta["grid"]["lines"]], dtype=np.int64)
    return {
        "node": node, "glob": glob, "hist": hist, "edge": edge, **labels,
        "lines": lines, "first_day": np.array(first), "dates": np.array(meta["dates"]),
        "zone_of": zone_of, "reactance": reac, "rating": rating,
    }


def save_features(feats: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "features.npz", **feats)
    return out_dir / "features.npz"


def load_features(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


REALIZED = ["da_lmp", "rt_lmp", "da_energy", "rt_energy", "rt_shadow", "da_shadow", "rt_inj"]


def leak_test(raw: dict, meta: dict, cfg: dict, n_checks: int = 5, seed: int = 0) -> dict:
    """Perturb realized data from day d-1 onward and require identical features for day d."""
    rng = np.random.default_rng(seed)
    base = build_features(raw, meta, cfg)
    D = raw["da_lmp"].shape[0]
    first = int(base["first_day"])
    days = rng.integers(first + 1, D, size=n_checks)
    worst = 0.0
    for d in days:
        pert = {k: v.copy() for k, v in raw.items()}
        for k in REALIZED:
            if k in pert:
                pert[k][d - 1:] = pert[k][d - 1:] + rng.normal(0, 50, pert[k][d - 1:].shape)
        f2 = build_features(pert, meta, cfg)
        for key in ["node", "glob", "hist", "edge"]:
            worst = max(worst, float(np.abs(f2[key][d] - base[key][d]).max()))
    return {"passed": worst == 0.0, "max_feature_change": worst, "days_checked": [int(x) for x in days]}
