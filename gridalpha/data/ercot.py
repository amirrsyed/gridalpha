"""Real ERCOT data adapter -> the canonical raw schema (see ``schema.py``).

Runs where the public internet is reachable (GitHub Actions, Modal, a laptop). Sources:
  * ERCOT Public API through ``gridstatus.ErcotAPI``: DA hourly SPPs, RT 15-minute SPPs,
    DAM and SCED shadow prices, the seven-day load forecast by model and weather zone, and
    wind/solar forecasts by geographical region. Needs a free ERCOT API account:
    ERCOT_API_USERNAME, ERCOT_API_PASSWORD, ERCOT_PUBLIC_API_SUBSCRIPTION_KEY.
  * Open-Meteo historical-forecast API (no key) for temperature forecasts.
  * FRED series DHHNGSP (no key) for Henry Hub spot gas.

Point-in-time rules for operating day D (bids due D-1 10:00 CT):
  * forecasts: the latest publication posted before D-1 10:00 CT, rows for day D only,
  * prices and shadow prices: realized values, used by the features only from D-2 back.

Topology: ERCOT's bus-branch model is CEII-restricted, so the "lines" for real data are
the most frequently binding constraints, each attached to the two nodes whose DA
congestion co-moves most strongly with it (one on each side), measured on the first
year only so the graph is point-in-time. The GNN's learned adjacency does the rest.
The physics loss needs a bus-branch model and is switched off for real data.

Every source is fetched per day and cached in reduced form (24 hourly rows) under
``raw/cache``. Days in the last three days are never cached, because ERCOT and the
forecast vendors keep filling them in. A dataset that fails for a day is logged once
per run and left empty for that day; the run continues.
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
H = 24
ZONES = ["NORTH", "HOUSTON", "SOUTH", "WEST"]                  # ERCOT forecast zones
ZONE_CENTROIDS = {"NORTH": (32.8, -97.0), "HOUSTON": (29.8, -95.4), "SOUTH": (29.4, -98.5), "WEST": (31.9, -102.1)}
WEATHER_TO_ZONE = {"Coast": "HOUSTON", "East": "NORTH", "North": "NORTH", "North Central": "NORTH",
                   "Far West": "WEST", "West": "WEST", "South Central": "SOUTH", "Southern": "SOUTH"}
LOAD_ZONES = {"LZ_NORTH": "NORTH", "LZ_HOUSTON": "HOUSTON", "LZ_SOUTH": "SOUTH", "LZ_WEST": "WEST"}
WIND_REGIONS = ["PANHANDLE", "COASTAL", "SOUTH", "WEST", "NORTH"]
SOLAR_REGIONS = ["CenterEast", "CenterWest", "FarWest", "FarEast", "NorthWest", "SouthEast"]
# coarse projection of vendor regions onto forecast zones
WIND_TO_ZONE = {"NORTH": ["NORTH", "PANHANDLE"], "HOUSTON": ["COASTAL"], "SOUTH": ["SOUTH", "COASTAL"],
                "WEST": ["WEST", "PANHANDLE"]}
SOLAR_TO_ZONE = {"NORTH": ["CenterEast", "NorthWest"], "HOUSTON": ["SouthEast", "FarEast"],
                 "SOUTH": ["SouthEast", "CenterWest"], "WEST": ["FarWest", "CenterWest"]}
HUB_AVG = "HB_HUBAVG"


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def _api():
    from gridstatus.ercot_api.ercot_api import ErcotAPI
    return ErcotAPI(
        username=os.environ.get("ERCOT_API_USERNAME"),
        password=os.environ.get("ERCOT_API_PASSWORD"),
        public_subscription_key=os.environ.get("ERCOT_PUBLIC_API_SUBSCRIPTION_KEY"),
    )


def _today_ct() -> pd.Timestamp:
    return pd.Timestamp.now(tz=CT).normalize().tz_localize(None)


def _ct(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if s.dt.tz is None:
        return s.dt.tz_localize(CT, ambiguous="NaT", nonexistent="shift_forward")
    return s.dt.tz_convert(CT)


class SchemaError(Exception):
    """A column we rely on is missing: our parsing is wrong, so never cache the empty result."""


CACHE_VERSION = "v2"     # bump when a reducer changes so old cached days are rebuilt


def _col(df: pd.DataFrame, *cands: str) -> str:
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    for c in df.columns:
        if any(k.lower() in c.lower() for k in cands):
            return c
    raise SchemaError(f"none of {cands} in columns {list(df.columns)}")


def _hours_of(df: pd.DataFrame, day: pd.Timestamp, time_cols=("Interval Start", "SCED Timestamp")) -> pd.DataFrame:
    """Rows whose start time falls on `day` (Central time), with an `_h` hour column."""
    ts = _ct(df[_col(df, *time_cols)])
    keep = ts.dt.date == day.date()
    return df[keep.values].assign(_h=ts[keep].dt.hour.values)


def _pivot_hourly(df: pd.DataFrame, day: pd.Timestamp, value: tuple, key: tuple) -> pd.DataFrame:
    """Long frame -> [24 hours x key] means for one operating day."""
    if df is None or df.empty:
        return pd.DataFrame(index=range(H))
    d = _hours_of(df, day)
    v, k = _col(d, *value), _col(d, *key)
    d[v] = pd.to_numeric(d[v], errors="coerce")
    return d.groupby(["_h", k])[v].mean().unstack(k).reindex(range(H))


def _vintage(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Latest forecast published before the DA deadline (D-1 10:00 CT), rows for day D only."""
    if df is None or df.empty:
        return pd.DataFrame()
    deadline = (day - pd.Timedelta(days=1)).tz_localize(CT) + pd.Timedelta(hours=10)
    pub = _ct(df[_col(df, "Publish Time", "Posted Datetime")])
    ok = (pub < deadline).values
    if not ok.any():
        return pd.DataFrame()
    last = pub[ok].max()
    return _hours_of(df[(pub == last).values], day)


class _Fetcher:
    def __init__(self, cache: Path, log=print):
        self.cache = cache
        self.log = log
        self.errors: dict[str, str] = {}
        self.cutoff = _today_ct() - pd.Timedelta(days=3)

    def get(self, name: str, day: pd.Timestamp, fn) -> pd.DataFrame:
        path = self.cache / CACHE_VERSION / name / f"{day:%Y-%m-%d}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        df, final = None, False
        for attempt in range(2):
            try:
                df = fn()
                final = True
                break
            except SchemaError as e:                         # our parsing is wrong: do not cache
                self._note(name, day, e)
                break
            except KeyError as e:                            # ERCOT has no report for this day
                self._note(name, day, e)
                final = True
                break
            except Exception as e:  # noqa: BLE001 - network or rate limit: retry once, then skip
                if type(e).__name__ == "NoDataFoundException":
                    self._note(name, day, e)
                    final = True
                    break
                if attempt == 1:
                    self._note(name, day, e)
                time.sleep(5)
        if df is None:
            df = pd.DataFrame(index=range(H))
        if final and day <= self.cutoff:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.columns = [str(c) for c in df.columns]
            df.to_parquet(path)
        return df

    def _note(self, name: str, day: pd.Timestamp, e: Exception):
        if name not in self.errors:
            msg = f"{type(e).__name__}: {str(e)[:400]}"
            self.errors[name] = f"{day:%Y-%m-%d} {msg}"
            self.log(f"  ercot: {name} failed for {day:%Y-%m-%d} ({msg}); continuing without it")


# ---------------------------------------------------------------------------------------
# per-dataset reducers: each returns a 24-row frame for one operating day
# ---------------------------------------------------------------------------------------

def _load_fcst(api, day):
    raw = api.get_load_forecast_by_model(day - pd.Timedelta(days=1))
    if "In Use Flag" in raw.columns:
        raw = raw[raw["In Use Flag"].astype(bool)]
    v = _vintage(raw, day)
    if v.empty:
        return pd.DataFrame(index=range(H))
    out = pd.DataFrame(index=range(H))
    for z in ZONES:
        cols = [c for c, zz in WEATHER_TO_ZONE.items() if zz == z and c in v.columns]
        if cols:
            vals = v[cols].apply(pd.to_numeric, errors="coerce")
            out[z] = vals.groupby(v["_h"]).mean().sum(axis=1, min_count=1).reindex(range(H))
    return out


def _region_fcst(fetch_fn, prefix, regions, day):
    raw = fetch_fn(day - pd.Timedelta(days=1))
    v = _vintage(raw, day)
    out = pd.DataFrame(index=range(H))
    if v.empty:
        return out
    for r in regions:
        c = f"{prefix} {r}"
        if c in v.columns:
            out[r] = pd.to_numeric(v[c], errors="coerce").groupby(v["_h"]).mean().reindex(range(H))
    return out


def _open_meteo(day: pd.Timestamp) -> pd.DataFrame:
    import urllib.request
    out = pd.DataFrame(index=range(H))
    for zone, (lat, lon) in ZONE_CENTROIDS.items():
        url = ("https://historical-forecast-api.open-meteo.com/v1/forecast?"
               f"latitude={lat}&longitude={lon}&hourly=temperature_2m&timezone=America%2FChicago"
               f"&start_date={day:%Y-%m-%d}&end_date={day:%Y-%m-%d}")
        with urllib.request.urlopen(url, timeout=60) as r:
            js = json.loads(r.read())
        out[zone] = pd.Series(js["hourly"]["temperature_2m"][:H], dtype=float).reindex(range(H)).values
    return out


def _gas(log=print) -> pd.Series:
    import urllib.request
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP"
        with urllib.request.urlopen(url, timeout=60) as r:
            df = pd.read_csv(io.BytesIO(r.read()))
        df.columns = ["date", "gas"]
        return pd.Series(pd.to_numeric(df["gas"], errors="coerce").values, index=pd.to_datetime(df["date"])).ffill()
    except Exception as e:  # noqa: BLE001
        log(f"  ercot: FRED gas download failed ({e}); using a flat $3.00/MMBtu")
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------------------
# fetch + assemble
# ---------------------------------------------------------------------------------------

def fetch(cfg: dict, out_dir: Path, end: str | None = None, log=print, checkpoint=None, every: int = 40) -> Path:
    """Download (or read from cache) every day from ercot.start to tomorrow and build the raw dataset.

    checkpoint: optional callable run every `every` days, used to save the cache to cloud storage so a
    run that hits a time limit loses at most a few dozen days of downloads."""
    ec = cfg["ercot"]
    fx = _Fetcher(out_dir / "cache", log)
    api = _api()
    today = _today_ct()
    end = pd.Timestamp(end) if end else today + pd.Timedelta(days=1)       # tomorrow = the day we bid for
    days = pd.date_range(ec["start"], end, freq="D")
    S = {k: {} for k in ["da", "rt", "dsh", "rsh", "load", "wind", "solar", "temp"]}
    t0 = time.time()
    for i, d in enumerate(days):
        S["da"][d] = fx.get("da_spp", d, lambda: _pivot_hourly(api.get_spp_day_ahead_hourly(d), d, ("SPP",), ("Location",)))
        S["dsh"][d] = fx.get("dam_shadow", d, lambda: _pivot_hourly(api.get_shadow_prices_dam(d), d, ("Shadow Price",),
                                                                  ("Constraint Name",)))
        if d < today:            # real time for today is incomplete and tomorrow has not happened
            S["rt"][d] = fx.get("rt_spp", d, lambda: _pivot_hourly(api.get_spp_real_time_15_min(d), d, ("SPP",),
                                                                  ("Location",)))
            S["rsh"][d] = fx.get("sced_shadow", d, lambda: _pivot_hourly(api.get_shadow_prices_sced(d), d,
                                                                       ("Shadow Price",), ("Constraint Name",)))
        S["load"][d] = fx.get("load_fcst", d, lambda: _load_fcst(api, d))
        S["wind"][d] = fx.get("wind_fcst", d, lambda: _region_fcst(
            api.get_wind_actual_and_forecast_by_geographical_region_hourly, "STWPF", WIND_REGIONS, d))
        S["solar"][d] = fx.get("solar_fcst", d, lambda: _region_fcst(
            api.get_solar_actual_and_forecast_by_geographical_region_hourly, "STPPF", SOLAR_REGIONS, d))
        S["temp"][d] = fx.get("temp_fcst", d, lambda: _open_meteo(d))
        if i % 10 == 0 or i == len(days) - 1:
            log(f"  ercot: {d:%Y-%m-%d} ({i + 1}/{len(days)} days, {time.time() - t0:.0f}s)")
        if checkpoint is not None and (i + 1) % every == 0:
            try:
                checkpoint()
                log(f"  ercot: cache saved through {d:%Y-%m-%d}")
            except Exception as ex:  # noqa: BLE001 - a failed checkpoint must not stop the download
                log(f"  ercot: cache checkpoint failed ({ex}); continuing")
    if fx.errors:
        (out_dir / "fetch_errors.json").write_text(json.dumps(fx.errors, indent=1))
    return assemble(days, S, _gas(log), cfg, out_dir, log)


def _frame_cube(store: dict, days, cols: list[str]) -> np.ndarray:
    out = np.full((len(days), H, len(cols)), np.nan)
    for i, d in enumerate(days):
        f = store.get(d)
        if f is not None and len(f.columns):
            out[i] = f.reindex(index=range(H), columns=cols).values.astype(float)
    return out


def _coverage(store: dict, days) -> float:
    return float(np.mean([store.get(d) is not None and len(store[d].columns) > 0 and store[d].notna().any().any()
                          for d in days]))


def assemble(days, S: dict, gas: pd.Series, cfg: dict, out_dir: Path, log=print) -> Path:
    ec = cfg["ercot"]
    D = len(days)
    first = list(range(min(365, D)))
    # --- node universe: points present in >98% of first-year hours, hubs and load zones first
    da_fy = pd.concat([S["da"][days[i]] for i in first if len(S["da"][days[i]].columns)], axis=0)
    if da_fy.empty:
        raise RuntimeError("no day-ahead prices were downloaded; see raw/fetch_errors.json")
    keep = list(da_fy.columns[da_fy.notna().mean() > 0.98])
    must = [c for c in keep if c.startswith(("HB_", "LZ_"))]
    rest = [c for c in keep if c not in must][: max(ec["max_nodes"] - len(must), 0)]
    nodes = must + rest
    N = len(nodes)

    da_lmp = _frame_cube(S["da"], days, nodes)
    rt_lmp = _frame_cube(S["rt"], days, nodes)
    hub = nodes.index(HUB_AVG) if HUB_AVG in nodes else 0
    # small gaps in DA: carry the previous hour; tomorrow's DA stays NaN until it is published
    da2 = pd.DataFrame(da_lmp[:-1].reshape(-1, N)).ffill(limit=3).values.reshape(D - 1, H, N)
    da_lmp[:-1] = da2
    da_energy, rt_energy = da_lmp[:, :, hub], rt_lmp[:, :, hub]

    # --- constraints as "lines" (point-in-time: first year only)
    cnt = pd.concat([S["dsh"][days[i]].notna().sum() for i in first]).groupby(level=0).sum()
    cons = [c for c in cnt.sort_values(ascending=False).index[:150] if cnt[c] > 0]
    L = len(cons)
    da_sh = np.nan_to_num(_frame_cube(S["dsh"], days, cons))
    rt_sh = np.nan_to_num(_frame_cube(S["rsh"], days, cons))
    n_fy = len(first)
    cong = np.nan_to_num((da_lmp[:n_fy] - da_energy[:n_fy, :, None]).reshape(-1, N))
    cz = (cong - cong.mean(0)) / (cong.std(0) + 1e-9)
    lines = []
    for l in range(L):
        s = da_sh[:n_fy, :, l].reshape(-1)
        if s.std() == 0:
            lines.append((0, min(1, N - 1)))
            continue
        c = ((s - s.mean()) / s.std()) @ cz / len(s)
        a, b = int(np.argmin(c)), int(np.argmax(c))
        lines.append((a, b if b != a else (a + 1) % N))

    # --- node -> forecast zone by first-year DA congestion co-movement with the load zones
    lz = {z: nodes.index(c) for c, z in LOAD_ZONES.items() if c in nodes}
    zone_of = np.zeros(N, int)
    if lz:
        corr = {z: cz.T @ cz[:, j] / len(cz) for z, j in lz.items()}
        best = np.argmax(np.stack([corr[z] for z in ZONES if z in corr]), axis=0)
        present = [z for z in ZONES if z in corr]
        zone_of = np.array([ZONES.index(present[k]) for k in best])

    # --- forecasts
    load_z = np.nan_to_num(_frame_cube(S["load"], days, ZONES))
    share = 1.0 / np.maximum(np.bincount(zone_of, minlength=len(ZONES))[zone_of], 1)
    load_bus_fcst = load_z[:, :, zone_of] * share[None, None, :]
    wind = _frame_cube(S["wind"], days, WIND_REGIONS)
    solar = _frame_cube(S["solar"], days, SOLAR_REGIONS)
    wind_cf = np.nan_to_num(wind / np.maximum(np.nanmax(wind[:n_fy], axis=(0, 1), initial=1.0), 1.0))
    solar_cf = np.nan_to_num(solar / np.maximum(np.nanmax(solar[:n_fy], axis=(0, 1), initial=1.0), 1.0))
    wind_z = np.stack([wind_cf[:, :, [WIND_REGIONS.index(r) for r in WIND_TO_ZONE[z]]].mean(-1) for z in ZONES], -1)
    solar_z = np.stack([solar_cf[:, :, [SOLAR_REGIONS.index(r) for r in SOLAR_TO_ZONE[z]]].mean(-1) for z in ZONES], -1)
    temp = _frame_cube(S["temp"], days, ZONES)
    temp = np.nan_to_num(temp, nan=float(np.nanmean(temp)) if np.isfinite(temp).any() else 20.0)
    if len(gas):
        gas_d = gas.reindex(pd.DatetimeIndex(days) - pd.Timedelta(days=1), method="ffill").bfill().fillna(3.0).values
    else:
        gas_d = np.full(D, 3.0)

    raw = dict(
        da_lmp=da_lmp, rt_lmp=rt_lmp, da_energy=da_energy, rt_energy=rt_energy,
        load_bus_fcst=load_bus_fcst, wind_fcst_cf=wind_z, solar_fcst_cf=solar_z, temp_fcst=temp,
        gas=gas_d, planned_line=np.zeros((D, L), bool), rt_shadow=rt_sh, da_shadow=da_sh,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "market.npz", **{k: np.asarray(v) for k, v in raw.items()})
    coverage = {k: round(_coverage(S[k], days), 3) for k in S}
    meta = {
        "dates": [f"{d:%Y-%m-%d}" for d in days],
        "zones": ZONES, "zone_of": zone_of.tolist(), "dc_bus": [],
        "gens": {"bus": [], "fuel": [], "cap": []},
        "reactance": [1.0] * L,
        "grid": {
            "nodes": [{"id": j, "name": nodes[j], "zone": ZONES[zone_of[j]], "lon": None, "lat": None, "dc": False}
                      for j in range(N)],
            "lines": [{"id": l, "from": a, "to": b, "rating": 1000.0, "tie": False, "name": cons[l]}
                      for l, (a, b) in enumerate(lines)],
        },
        "source": "ercot",
        "coverage": coverage,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    log(f"  ercot raw dataset: {D} days, {N} nodes, {L} constraints; share of days with data: {coverage}")
    return out_dir
