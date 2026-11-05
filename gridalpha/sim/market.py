"""Scenario generator and day-ahead / real-time market clearing.

Each operating day gets weather, load, renewables, gas prices and outages. The
day-ahead market clears a DC optimal power flow on the *forecasts* published the day
before (with planned outages); the real-time market clears on what actually happened
(with forced outages and a scarcity adder). LMPs come from the LP duals, so every
congestion price is caused by a binding line, exactly as in a real nodal market.

Built-in, realistic sources of predictable DA-RT dislocation (what a model can learn):
  * the ISO's load forecast learns new data-center load slowly (smoothed history),
  * wind is over-forecast at night in high-wind hours; ramps are sometimes mistimed,
  * solar is over-forecast on cloudy days,
  * load is under-forecast in heat waves and winter storms,
  * the DA market under-weights the RT scarcity adder,
  * forced outages are unknown in DA but more likely in storms and on loaded lines.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .grid import COAL_PRICE, Grid, build_grid

ZONE_TEMP = {"FAR_WEST": 19.0, "WEST": 18.0, "PANHANDLE": 15.0, "NORTH": 19.0,
             "CENTRAL": 20.5, "HOUSTON": 21.0, "SOUTH": 22.5}
SYS_BASE_MW = 9900.0
THERMAL = ("gas_cc", "gas_ct", "coal", "nuclear")


# ---------------------------------------------------------------------------
# Scenario generation (vectorized over the whole calendar)
# ---------------------------------------------------------------------------

def generate_scenarios(grid: Grid, start: str, end: str, seed: int = 7, gas_start: float = 2.8,
                       dc_growth: bool = True) -> dict:
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range(start, end, freq="D")
    D, H, Z, N = len(dates), 24, len(grid.zone_names), grid.n
    doy = dates.dayofyear.values
    month = dates.month.values
    hours = np.arange(H)

    # --- extreme weather events -------------------------------------------------
    storm = np.zeros(D)
    heat = np.zeros(D)
    for yr in sorted(set(dates.year)):
        win = np.where(((dates.year == yr) & (month <= 2)))[0]
        if len(win) > 10 and rng.random() < 0.85:
            s = rng.choice(win[:-5])
            storm[s:s + 4] = np.array([0.6, 1.0, 1.0, 0.5])[: len(storm[s:s + 4])]
        summ = np.where((dates.year == yr) & ((month == 7) | (month == 8)))[0]
        if len(summ) > 10:
            s = rng.choice(summ[:-7])
            heat[s:s + 6] = 1.0

    # --- temperature ------------------------------------------------------------
    anom = np.zeros((D, Z))
    common = 0.0
    for d in range(1, D):
        common = 0.75 * common + rng.normal(0, 2.4)
        anom[d] = 0.75 * anom[d - 1] + 0.25 * common + rng.normal(0, 1.4, Z)
    base_t = np.array([ZONE_TEMP[z] for z in grid.zone_names])
    t_day = base_t[None, :] + 10.0 * np.cos(2 * np.pi * (doy[:, None] - 200) / 365.0) + anom
    t_day += -16.0 * storm[:, None] + 5.0 * heat[:, None]
    diurnal = 5.5 * np.sin(2 * np.pi * (hours - 9) / 24.0)
    temp = t_day[:, None, :] + diurnal[None, :, None]                       # [D,H,Z]
    temp_fcst = temp + rng.normal(0, 1.5, (D, 1, Z))

    # --- load ---------------------------------------------------------------------
    summer_shape = 0.80 + 0.20 * np.exp(-((hours - 17) ** 2) / 18.0)
    winter_shape = 0.80 + 0.12 * np.exp(-((hours - 8) ** 2) / 6.0) + 0.16 * np.exp(-((hours - 19) ** 2) / 8.0)
    wsum = np.clip(np.cos(2 * np.pi * (doy - 200) / 365.0), 0, 1)[:, None]
    shape = wsum * summer_shape[None, :] + (1 - wsum) * winter_shape[None, :]
    wkd = np.where(dates.dayofweek.values >= 5, 0.94, 1.0)[:, None]
    growth = (1.02 ** ((dates - dates[0]).days.values / 365.0))[:, None]
    sens = 1 + 0.032 * np.clip(temp - 24, 0, None) + 0.022 * np.clip(14 - temp, 0, None)
    zone_share = np.array([grid.load_share[grid.zone_of == z].sum() for z in range(Z)])
    load_zone = SYS_BASE_MW * zone_share[None, None, :] * (shape * wkd * growth)[:, :, None] * sens
    load_zone *= 1 + rng.normal(0, 0.01, load_zone.shape)

    # ISO DA load forecast error: AR over hours + heat-wave / storm under-forecast
    e = np.zeros((D, H, Z))
    for h in range(H):
        prev = e[:, h - 1] if h else 0
        e[:, h] = 0.9 * prev + rng.normal(0, 0.009, (D, Z))
    tf = temp_fcst
    bias = -0.02 * (tf > 35) - 0.045 * storm[:, None, None]
    load_zone_fcst = load_zone * (1 + e + bias)

    within = np.zeros(N)
    for z in range(Z):
        idx = grid.zone_of == z
        within[idx] = grid.load_share[idx] / grid.load_share[idx].sum()
    load_bus = load_zone[:, :, grid.zone_of] * within[None, None, :]
    load_bus_fcst = load_zone_fcst[:, :, grid.zone_of] * within[None, None, :]

    # --- data centers: actual load and the ISO's smoothed forecast of it -------------
    dc_mw = np.zeros((D, N))
    for k, (zone, mw0, mw1, t0, t1) in enumerate(grid.dc_spec):
        t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
        frac = np.clip((dates - t0).days.values / max((t1 - t0).days, 1), 0, 1)
        on = dates >= t0
        mw = np.where(on, mw0 + (mw1 - mw0) * frac, 0.0) if dc_growth else np.full(D, mw1)
        dc_mw[:, grid.dc_bus[k]] += mw
    dc_act = dc_mw[:, None, :] * (0.92 + rng.normal(0, 0.02, (D, H, N))).clip(0.8, 1.0)
    dc_daily = dc_act.mean(axis=1)
    alpha = 1 - 0.5 ** (1 / 45.0)
    dc_fc = np.zeros((D, N))
    s = dc_daily[0]
    for d in range(D):
        lag = max(d - 2, 0)
        s = (1 - alpha) * s + alpha * dc_daily[lag]
        dc_fc[d] = s
    dc_fcst = np.repeat(dc_fc[:, None, :], H, axis=1)
    load_bus = load_bus + dc_act
    load_bus_fcst = load_bus_fcst + dc_fcst

    # --- wind -----------------------------------------------------------------------
    T = D * H
    season_cf = np.select([np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8]), np.isin(month, [9, 10, 11])],
                          [0.45, 0.30, 0.38], 0.40)
    wind_zones = {"WEST": 2.0, "PANHANDLE": 2.0, "SOUTH": 18.0}       # diurnal peak hour
    corr = np.array([[1, .7, .3], [.7, 1, .25], [.3, .25, 1]])
    Lc = np.linalg.cholesky(corr)
    zt = np.zeros((T, 3))
    for t in range(1, T):
        zt[t] = 0.93 * zt[t - 1] + 0.37 * (Lc @ rng.normal(size=3))
    zt = zt.reshape(D, H, 3)
    wind_cf = np.zeros((D, H, Z))
    for j, (z, peak) in enumerate(wind_zones.items()):
        zi = grid.zone_names.index(z)
        di = 0.08 * np.cos(2 * np.pi * (hours - peak) / 24.0)
        wind_cf[:, :, zi] = np.clip(season_cf[:, None] + 0.17 * zt[:, :, j] + di[None, :], 0.02, 0.95)
    werr = np.zeros((D, H, Z))
    for h in range(H):
        prev = werr[:, h - 1] if h else 0
        werr[:, h] = 0.85 * prev + rng.normal(0, 0.032, (D, Z))
    night = ((hours <= 6) | (hours >= 21))[None, :, None]
    wbias = 0.05 * ((wind_cf > 0.45) & night)
    wind_fcst_cf = np.clip(wind_cf + werr + wbias, 0.0, 0.97)
    shift = rng.random(D) < 0.25
    rolled = np.roll(wind_fcst_cf, 2, axis=1)
    wind_fcst_cf[shift] = rolled[shift]

    # --- solar ----------------------------------------------------------------------
    daylen = 12 + 2.0 * np.cos(2 * np.pi * (doy - 172) / 365.0)
    noon = 13.3
    rise = noon - daylen / 2
    phase = (hours[None, :] + 0.5 - rise[:, None]) / daylen[:, None]
    clear = np.where((phase > 0) & (phase < 1), np.sin(np.pi * np.clip(phase, 0, 1)), 0.0)
    cloud = np.clip(rng.beta(5, 2, (D, Z)) + 0.1 * wsum, 0.05, 1.0)
    cloud_f = np.clip(cloud + rng.normal(0, 0.06, (D, Z)) + 0.35 * np.clip(0.6 - cloud, 0, None), 0.05, 1.0)
    solar_cf = 0.85 * clear[:, :, None] * cloud[:, None, :]
    solar_fcst_cf = 0.85 * clear[:, :, None] * cloud_f[:, None, :]

    # --- gas price --------------------------------------------------------------------
    x = np.log(gas_start)
    gas = np.zeros(D)
    for d in range(D):
        x = x + 0.03 * (np.log(3.0) - x) + rng.normal(0, 0.035)
        gas[d] = np.exp(x)
    gas = gas * (1 + 5.0 * storm) * (1 + 0.15 * heat)

    # --- outages ----------------------------------------------------------------------
    G = len(grid.gens["cap"])
    thermal = np.isin(grid.gens["fuel"], THERMAL)
    shoulder = np.isin(month, [3, 4, 5, 10, 11])
    planned_line = np.zeros((D, grid.m), bool)
    for d in range(D):
        if rng.random() < (0.30 if shoulder[d] else 0.05):
            for _ in range(10):
                l = int(rng.integers(grid.m))
                dur = int(rng.integers(3, 9))
                window = planned_line[d:d + dur].copy()
                window[:, l] = True
                if all(grid.connected_without(tuple(np.where(w)[0])) for w in window):
                    planned_line[d:d + dur, l] = True
                    break
    planned_gen = np.zeros((D, G), bool)
    for d in range(D):
        if rng.random() < (0.20 if shoulder[d] else 0.03):
            g = int(rng.choice(np.where(thermal)[0]))
            planned_gen[d:d + int(rng.integers(5, 21)), g] = True
    forced_gen = np.zeros((D, H, G), bool)
    p_fail = 0.012 + 0.16 * storm
    for d in range(D):
        fail = (rng.random(G) < p_fail[d]) & thermal & ~planned_gen[d]
        for g in np.where(fail)[0]:
            forced_gen[d, int(rng.integers(H)):, g] = True
    forced_line = np.zeros((D, H, grid.m), bool)
    for d in range(D):
        if rng.random() < 0.10 + 0.3 * storm[d]:
            for _ in range(10):
                l = int(rng.integers(grid.m))
                out = tuple(np.where(planned_line[d])[0]) + (l,)
                if grid.connected_without(out):
                    h0 = int(rng.integers(H))
                    forced_line[d, h0:h0 + int(rng.integers(3, 11)), l] = True
                    break

    return dict(
        dates=dates, temp=temp, temp_fcst=temp_fcst, load_bus=load_bus, load_bus_fcst=load_bus_fcst,
        dc_mw=dc_mw, dc_fcst=dc_fcst, wind_cf=wind_cf, wind_fcst_cf=wind_fcst_cf, solar_cf=solar_cf,
        solar_fcst_cf=solar_fcst_cf, gas=gas, planned_line=planned_line, planned_gen=planned_gen,
        forced_gen=forced_gen, forced_line=forced_line, storm=storm, heat=heat,
    )


# ---------------------------------------------------------------------------
# DC optimal power flow
# ---------------------------------------------------------------------------

@dataclass
class OPFResult:
    lmp: np.ndarray
    energy: float
    shadow: np.ndarray   # [L] >= 0, signed by direction in `direction`
    flow: np.ndarray     # [L]
    inj: np.ndarray      # [N]
    shed: float


def solve_opf(ptdf: np.ndarray, Mg: np.ndarray, avail: np.ndarray, cost: np.ndarray, load: np.ndarray,
              rating: np.ndarray, active: np.ndarray, voll: float) -> OPFResult:
    N = load.shape[0]
    G = avail.shape[0]
    P = ptdf[active]
    F = rating[active]
    PM = P @ Mg
    base = P @ load
    A_ub = np.vstack([np.hstack([PM, P]), np.hstack([-PM, -P])])
    b_ub = np.concatenate([F + base, F - base])
    A_eq = np.ones((1, G + N))
    b_eq = np.array([load.sum()])
    c = np.concatenate([cost, np.full(N, voll)])
    bounds = np.column_stack([np.zeros(G + N), np.concatenate([avail, load])])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"OPF failed: {res.message}")
    lam = res.eqlin.marginals[0]
    mu = res.ineqlin.marginals
    k = active.sum()
    mu_pos, mu_neg = mu[:k], mu[k:]
    # d(cost)/d(load_i) = lam + sum_l mu+_l P_li - sum_l mu-_l P_li
    lmp = lam + P.T @ (mu_pos - mu_neg)
    x = res.x
    inj = Mg @ x[:G] + x[G:] - load
    flow_a = P @ inj
    shadow = np.zeros(ptdf.shape[0])
    flow = np.zeros(ptdf.shape[0])
    shadow[active] = -(mu_pos - mu_neg)     # positive when the forward limit binds
    flow[active] = flow_a
    return OPFResult(lmp=lmp, energy=lam, shadow=shadow, flow=flow, inj=inj, shed=float(x[G:].sum()))


def _costs(grid: Grid, gas: float, hr_scale: np.ndarray | None = None) -> np.ndarray:
    g = grid.gens
    hr = g["hr"] * (hr_scale if hr_scale is not None else 1.0)
    fuel_price = np.where(g["fuel"] == "coal", COAL_PRICE, gas)
    return np.where(np.isnan(g["fixed"]), hr * fuel_price + g["vom"], g["fixed"])


def _avail(grid: Grid, wind_cf_zone: np.ndarray, solar_cf_zone: np.ndarray, out: np.ndarray) -> np.ndarray:
    g = grid.gens
    z = grid.zone_of[g["bus"]]
    cf = np.where(g["fuel"] == "wind", wind_cf_zone[z], np.where(g["fuel"] == "solar", solar_cf_zone[z], 1.0))
    return np.where(out, 0.0, g["cap"] * cf)


def scarcity_adder(avail_total: float, load_total: float, voll: float, weight: float = 1.0) -> float:
    margin = (avail_total - load_total) / max(load_total, 1.0)
    return float(min(voll, weight * 2000.0 * np.exp(-max(margin, 0.0) / 0.05)))


def _analyst_grid(grid: Grid, seed: int) -> tuple[Grid, np.ndarray, np.ndarray]:
    """The trader's own (imperfect) network model: perturbed ratings, some lines unknown."""
    rng = np.random.default_rng(seed + 99)
    rating = grid.rating * np.exp(rng.normal(0, 0.15, grid.m))
    known = np.ones(grid.m, bool)
    internal = np.where(~grid.is_tie)[0]
    drop = rng.choice(internal, size=max(1, int(0.08 * len(internal))), replace=False)
    for l in drop:
        known[l] = False
        if not grid.connected_without(tuple(np.where(~known)[0])):
            known[l] = True
    hr_scale = np.exp(rng.normal(0, 0.04, len(grid.gens["cap"])))
    return rating, known, hr_scale


def clear_days(args) -> dict:
    """Worker: clear DA, RT and the analyst's OPF for a slice of days."""
    seed, day_idx, scen = args
    grid = build_grid(seed)
    voll = scen["voll"]
    N, L, H = grid.n, grid.m, 24
    G = len(grid.gens["cap"])
    Mg = np.zeros((N, G))
    Mg[grid.gens["bus"], np.arange(G)] = 1.0
    a_rating, a_known, a_hr = _analyst_grid(grid, seed)
    out = {k: [] for k in ["da_lmp", "da_energy", "da_shadow", "da_flow", "rt_lmp", "rt_energy", "rt_shadow",
                           "rt_flow", "rt_inj", "op_lmp", "op_energy", "op_shadow", "da_margin", "rt_margin",
                           "rt_shed", "da_adder", "rt_adder"]}
    for j, d in enumerate(day_idx):
        rows = {k: [] for k in out}
        gas = scen["gas"][j]
        gas_prev = scen["gas_prev"][j]
        cost = _costs(grid, gas)
        cost_an = _costs(grid, gas_prev, a_hr)
        for h in range(H):
            pl = scen["planned_line"][j]
            fl = scen["forced_line"][j, h]
            da_out = tuple(np.where(pl)[0])
            rt_out = tuple(np.where(pl | fl)[0])
            # ---- day-ahead on forecasts
            av = _avail(grid, scen["wind_fcst_cf"][j, h], scen["solar_fcst_cf"][j, h], scen["planned_gen"][j])
            load_f = scen["load_bus_fcst"][j, h]
            r = solve_opf(grid.ptdf(da_out), Mg, av, cost, load_f, grid.rating, ~pl, voll)
            add = scarcity_adder(av.sum(), load_f.sum(), voll, weight=0.5)
            rows["da_lmp"].append(r.lmp + add)
            rows["da_energy"].append(r.energy + add)
            rows["da_shadow"].append(r.shadow)
            rows["da_flow"].append(r.flow)
            rows["da_margin"].append((av.sum() - load_f.sum()) / load_f.sum())
            rows["da_adder"].append(add)
            # ---- analyst's own OPF: same forecasts, imperfect network, yesterday's gas
            act = a_known & ~pl
            ra = solve_opf(grid.ptdf(tuple(np.where(~a_known | pl)[0])), Mg, av, cost_an, load_f, a_rating, act, voll)
            rows["op_lmp"].append(ra.lmp)
            rows["op_energy"].append(ra.energy)
            rows["op_shadow"].append(ra.shadow)
            # ---- real time on actuals + forced outages
            out_g = scen["planned_gen"][j] | scen["forced_gen"][j, h]
            av = _avail(grid, scen["wind_cf"][j, h], scen["solar_cf"][j, h], out_g)
            load_a = scen["load_bus"][j, h]
            rr = solve_opf(grid.ptdf(rt_out), Mg, av, cost, load_a, grid.rating, ~(pl | fl), voll)
            add = scarcity_adder(av.sum(), load_a.sum(), voll, weight=1.0)
            rows["rt_lmp"].append(np.minimum(rr.lmp + add, voll))
            rows["rt_energy"].append(rr.energy + add)
            rows["rt_shadow"].append(rr.shadow)
            rows["rt_flow"].append(rr.flow)
            rows["rt_inj"].append(rr.inj)
            rows["rt_margin"].append((av.sum() - load_a.sum()) / load_a.sum())
            rows["rt_shed"].append(rr.shed)
            rows["rt_adder"].append(add)
        for k in out:
            out[k].append(np.array(rows[k]))
    return {k: np.array(v) for k, v in out.items()} | {"day_idx": np.array(day_idx)}


def simulate(cfg: dict, out_dir: Path, log=print) -> Path:
    sc = cfg["sim"]
    seed = cfg["project"]["seed"]
    grid = build_grid(seed)
    scen = generate_scenarios(grid, sc["start"], sc["end"], seed, sc["gas_start"], sc["data_center_growth"])
    D = len(scen["dates"])
    gas_prev = np.concatenate([[scen["gas"][0]], scen["gas"][:-1]])
    workers = max(1, int(sc.get("workers", 1)))
    chunks = np.array_split(np.arange(D), max(workers * 8, 1))
    keys = ["load_bus", "load_bus_fcst", "wind_cf", "wind_fcst_cf", "solar_cf", "solar_fcst_cf",
            "planned_line", "planned_gen", "forced_gen", "forced_line"]
    jobs = []
    for ch in chunks:
        if len(ch) == 0:
            continue
        sub = {k: scen[k][ch] for k in keys}
        sub["gas"] = scen["gas"][ch]
        sub["gas_prev"] = gas_prev[ch]
        sub["voll"] = sc["voll"]
        jobs.append((seed, ch.tolist(), sub))
    t0 = time.time()
    results = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(clear_days, jobs)):
                results.append(r)
                log(f"  cleared {sum(len(x['day_idx']) for x in results)}/{D} days ({time.time() - t0:.0f}s)")
    else:
        for r in map(clear_days, jobs):
            results.append(r)
    merged = {k: np.concatenate([r[k] for r in results]) for k in results[0]}
    order = np.argsort(merged.pop("day_idx"))
    merged = {k: v[order] for k, v in merged.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in scen.items() if k != "dates"} | merged
    np.savez_compressed(out_dir / "market.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    meta = {
        "dates": [d.strftime("%Y-%m-%d") for d in scen["dates"]],
        "grid": grid.to_json(),
        "zones": grid.zone_names,
        "zone_of": grid.zone_of.tolist(),
        "dc_bus": grid.dc_bus.tolist(),
        "gens": {k: (v.tolist() if v.dtype != object else list(v)) for k, v in grid.gens.items()},
        "reactance": grid.reactance.tolist(),
        "source": "sim",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    log(f"  market cleared: {D} days x 24 h x 3 OPFs in {time.time() - t0:.0f}s")
    return out_dir
