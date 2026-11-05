"""Real ERCOT data adapter -> the canonical raw schema (see ``schema.py``).

Runs where the public internet is reachable (GitHub Actions, Modal, a laptop). Sources:
  * ERCOT Public API through ``gridstatus.ErcotAPI``: DA hourly SPPs, RT 15-minute SPPs,
    DAM and SCED shadow prices, load forecast by model, wind/solar forecasts by region.
    Needs a free ERCOT API account: set ERCOT_API_USERNAME, ERCOT_API_PASSWORD and
    ERCOT_PUBLIC_API_SUBSCRIPTION_KEY.
  * Open-Meteo historical-forecast API (no key) for archived temperature forecasts.
  * FRED series DHHNGSP (no key) for Henry Hub spot gas.

Topology: ERCOT's bus-branch model is CEII-restricted, so the "lines" for real data are
the most frequently binding constraints. Each constraint is attached to the two nodes
whose DA congestion co-moves most strongly (one on each side) in the first training
year only, so the graph is point-in-time. The adaptive (learned) adjacency in the GNN
does the rest. The physics loss needs a bus-branch model and is switched off here.

Every network call is cached per day under ``cache_dir`` so reruns are cheap.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

CT = "US/Central"
ZONE_CENTROIDS = {          # forecast-zone centroids for Open-Meteo
    "NORTH": (32.8, -97.0), "HOUSTON": (29.8, -95.4), "SOUTH": (29.4, -98.5), "WEST": (31.9, -102.1),
}
LOAD_ZONES = {"LZ_NORTH": "NORTH", "LZ_HOUSTON": "HOUSTON", "LZ_SOUTH": "SOUTH", "LZ_WEST": "WEST"}
WIND_REGIONS = ["PANHANDLE", "COASTAL", "SOUTH", "WEST", "NORTH"]
SOLAR_REGIONS = ["CENTEREAST", "CENTERWEST", "FARWEST", "FAREAST", "NORTHWEST", "SOUTHEAST"]
HUB_AVG = "HB_HUBAVG"


def _api():
    from gridstatus.ercot_api.ercot_api import ErcotAPI
    return ErcotAPI(
        username=os.environ.get("ERCOT_API_USERNAME"),
        password=os.environ.get("ERCOT_API_PASSWORD"),
        public_subscription_key=os.environ.get("ERCOT_PUBLIC_API_SUBSCRIPTION_KEY"),
    )


def _cached(cache: Path, name: str, day: pd.Timestamp, fn):
    path = cache / name / f"{day:%Y-%m-%d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    for attempt in range(3):
        try:
            df = fn()
            break
        except Exception:                      # network hiccup: back off and retry
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _col(df: pd.DataFrame, *cands: str) -> str:
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    for c in df.columns:
        if any(k.lower() in c.lower() for k in cands):
            return c
    raise KeyError(f"none of {cands} in {list(df.columns)}")


def _hourly(df: pd.DataFrame, value: str, key: str = "Location") -> pd.DataFrame:
    t = _col(df, "Interval Start")
    out = df.assign(_t=pd.to_datetime(df[t]).dt.tz_convert(CT).dt.floor("h"))
    return out.groupby(["_t", key])[value].mean().unstack(key)


def _vintage(df: pd.DataFrame, deadline: pd.Timestamp) -> pd.DataFrame:
    """Keep the latest forecast published before the DA bid deadline."""
    if "Publish Time" not in df.columns:
        return df
    pub = pd.to_datetime(df["Publish Time"]).dt.tz_convert(CT)
    ok = df[pub < deadline]
    if ok.empty:
        return ok
    last = pd.to_datetime(ok["Publish Time"]).max()
    return ok[pd.to_datetime(ok["Publish Time"]) == last]


def _open_meteo(day: pd.Timestamp) -> pd.DataFrame:
    import urllib.request
    rows = []
    for zone, (lat, lon) in ZONE_CENTROIDS.items():
        url = ("https://historical-forecast-api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}&hourly=temperature_2m&timezone=America%2FChicago"
               f"&start_date={day:%Y-%m-%d}&end_date={day:%Y-%m-%d}")
        with urllib.request.urlopen(url, timeout=60) as r:
            js = json.loads(r.read())
        rows.append(pd.Series(js["hourly"]["temperature_2m"], name=zone))
    return pd.concat(rows, axis=1)


def _gas() -> pd.Series:
    import urllib.request
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP"
    with urllib.request.urlopen(url, timeout=60) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    df.columns = ["date", "gas"]
    s = pd.to_numeric(df["gas"], errors="coerce")
    return pd.Series(s.values, index=pd.to_datetime(df["date"])).ffill()


def fetch(cfg: dict, out_dir: Path, end: str | None = None, log=print) -> Path:
    ec = cfg["ercot"]
    cache = out_dir / "cache"
    api = _api()
    end = pd.Timestamp(end or pd.Timestamp.now(tz=CT).normalize() + pd.Timedelta(days=1)).tz_localize(None)
    days = pd.date_range(ec["start"], end, freq="D")
    da, rt, dsh, rsh, lf, wf, sf, tf = {}, {}, {}, {}, {}, {}, {}, {}
    for i, day in enumerate(days):
        deadline = (day - pd.Timedelta(days=1)).tz_localize(CT) + pd.Timedelta(hours=10)
        da[day] = _hourly(_cached(cache, "da_spp", day, lambda: api.get_spp_day_ahead_hourly(day)), "SPP")
        if day < end:   # tomorrow's RT has not happened yet
            rt[day] = _hourly(_cached(cache, "rt_spp", day, lambda: api.get_spp_real_time_15_min(day)), "SPP")
            s = _cached(cache, "sced_shadow", day, lambda: api.get_shadow_prices_sced(day))
            rsh[day] = _hourly(s, _col(s, "Shadow Price"), _col(s, "Constraint Name"))
        s = _cached(cache, "dam_shadow", day, lambda: api.get_shadow_prices_dam(day))
        dsh[day] = _hourly(s, _col(s, "Shadow Price"), _col(s, "Constraint Name"))
        lf[day] = _vintage(_cached(cache, "load_fcst", day, lambda: api.get_load_forecast_by_model(day)), deadline)
        wf[day] = _vintage(_cached(cache, "wind_fcst", day,
                                   lambda: api.get_wind_actual_and_forecast_by_geographical_region_hourly(day)), deadline)
        sf[day] = _vintage(_cached(cache, "solar_fcst", day,
                                   lambda: api.get_solar_actual_and_forecast_by_geographical_region_hourly(day)), deadline)
        tf[day] = _cached(cache, "temp_fcst", day, lambda: _open_meteo(day))
        if i % 30 == 0:
            log(f"  ercot: {day:%Y-%m-%d} ({i + 1}/{len(days)})")
    return assemble(days, end, da, rt, dsh, rsh, lf, wf, sf, tf, _gas(), cfg, out_dir, log)


def assemble(days, end, da, rt, dsh, rsh, lf, wf, sf, tf, gas, cfg, out_dir: Path, log=print) -> Path:
    ec = cfg["ercot"]
    D, H = len(days), 24
    # --- node universe: most liquid points in the first year + hubs and load zones
    first_year = [d for d in days[: min(365, D)] if d in da]
    pts = pd.concat([da[d] for d in first_year], axis=0)
    keep = list(pts.columns[pts.notna().mean() > 0.98])
    must = [c for c in keep if c.startswith(("HB_", "LZ_"))]
    rest = [c for c in keep if c not in must][: max(ec["max_nodes"] - len(must), 0)]
    nodes = must + rest
    N = len(nodes)

    def cube(store: dict, cols: list[str]) -> np.ndarray:
        out = np.full((D, H, len(cols)), np.nan)
        for i, d in enumerate(days):
            if d in store and len(store[d]):
                frame = store[d].reindex(columns=cols)
                out[i, : min(H, len(frame))] = frame.values[:H]
        return out

    da_lmp = cube(da, nodes)
    rt_lmp = cube(rt, nodes)
    hub = nodes.index(HUB_AVG) if HUB_AVG in nodes else 0
    da_energy, rt_energy = da_lmp[:, :, hub], rt_lmp[:, :, hub]
    # forward-fill small gaps; the realized day D (tomorrow) stays NaN in RT
    da_lmp = pd.DataFrame(da_lmp.reshape(D * H, N)).ffill().values.reshape(D, H, N)

    # --- constraints as "lines" (point-in-time: first year only)
    cnt = pd.concat([dsh[d].notna().sum() for d in first_year if d in dsh]).groupby(level=0).sum()
    cons = list(cnt.sort_values(ascending=False).index[:150])
    L = len(cons)
    da_sh = np.nan_to_num(cube(dsh, cons))
    rt_sh = np.nan_to_num(cube(rsh, cons))
    n_fy = len(first_year)
    da_cong_fy = (da_lmp[:n_fy] - da_energy[:n_fy, :, None]).reshape(-1, N)
    lines = []
    for l in range(L):
        s = da_sh[:n_fy, :, l].reshape(-1)
        if s.std() == 0:
            lines.append((0, 1))
            continue
        c = np.array([np.corrcoef(s, np.nan_to_num(da_cong_fy[:, j]))[0, 1] if da_cong_fy[:, j].std() > 0 else 0
                      for j in range(N)])
        lines.append((int(np.nanargmin(c)), int(np.nanargmax(c))))

    # --- node -> load zone by DA price co-movement (first year)
    lz_idx = {z: nodes.index(lz) for lz, z in LOAD_ZONES.items() if lz in nodes}
    zones = list(ZONE_CENTROIDS)
    zone_of = np.zeros(N, int)
    for j in range(N):
        best = max(lz_idx.items(), key=lambda kv: np.corrcoef(np.nan_to_num(da_cong_fy[:, j]),
                                                               np.nan_to_num(da_cong_fy[:, kv[1]]))[0, 1])
        zone_of[j] = zones.index(best[0])

    # --- forecasts
    def zone_cube(store: dict, prefix_map: dict) -> np.ndarray:
        out = np.zeros((D, H, len(prefix_map)))
        for i, d in enumerate(days):
            f = store.get(d)
            if f is None or f.empty:
                continue
            for k, cols in enumerate(prefix_map.values()):
                hit = [c for c in f.columns if any(p.lower() in c.lower() for p in cols)]
                if hit:
                    out[i, :, k] = pd.to_numeric(f[hit[0]], errors="coerce").values[:H]
        return out

    load_z = zone_cube(lf, {z: [z] for z in zones})
    share = np.ones(N) / np.maximum(np.bincount(zone_of, minlength=len(zones))[zone_of], 1)
    load_bus_fcst = load_z[:, :, zone_of] * share[None, None, :]
    wind = zone_cube(wf, {r: [f"STWPF {r}"] for r in WIND_REGIONS})
    solar = zone_cube(sf, {r: [f"STPPF {r}"] for r in SOLAR_REGIONS})
    wind_cf = wind / np.maximum(wind.max(axis=(0, 1), keepdims=True), 1)
    solar_cf = solar / np.maximum(solar.max(axis=(0, 1), keepdims=True), 1)
    # project regional renewables onto the four forecast zones (coarse, documented)
    rz = {"WEST": [0, 3], "SOUTH": [1, 2], "NORTH": [4], "HOUSTON": [1]}
    wind_z = np.stack([wind_cf[:, :, rz[z]].mean(-1) for z in zones], -1)
    sz = {"WEST": [2, 4], "SOUTH": [5], "NORTH": [0, 1], "HOUSTON": [3]}
    solar_z = np.stack([solar_cf[:, :, sz[z]].mean(-1) for z in zones], -1)
    temp = np.zeros((D, H, len(zones)))
    for i, d in enumerate(days):
        if d in tf:
            temp[i] = tf[d][zones].values[:H]
    gas_d = gas.reindex(days, method="ffill").shift(1).bfill().values     # yesterday's spot

    raw = dict(
        da_lmp=da_lmp, rt_lmp=rt_lmp, da_energy=da_energy, rt_energy=rt_energy,
        load_bus_fcst=load_bus_fcst, wind_fcst_cf=wind_z, solar_fcst_cf=solar_z, temp_fcst=temp,
        gas=gas_d, planned_line=np.zeros((D, L), bool), rt_shadow=rt_sh, da_shadow=da_sh,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "market.npz", **{k: np.asarray(v, dtype=float if v.dtype != bool else bool)
                                                   for k, v in raw.items()})
    meta = {
        "dates": [f"{d:%Y-%m-%d}" for d in days],
        "zones": zones, "zone_of": zone_of.tolist(), "dc_bus": [],
        "gens": {"bus": [], "fuel": [], "cap": []},
        "reactance": [1.0] * L,
        "grid": {
            "nodes": [{"id": j, "name": nodes[j], "zone": zones[zone_of[j]], "lon": None, "lat": None, "dc": False}
                      for j in range(N)],
            "lines": [{"id": l, "from": a, "to": b, "rating": 1000.0, "tie": False, "name": cons[l]}
                      for l, (a, b) in enumerate(lines)],
        },
        "source": "ercot",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    log(f"  ercot raw dataset: {D} days, {N} nodes, {L} constraints")
    return out_dir
