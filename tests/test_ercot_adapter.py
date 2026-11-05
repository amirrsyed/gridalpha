"""Offline test of the ERCOT adapter with a fake API that returns gridstatus-shaped frames."""
import numpy as np
import pandas as pd
import pytest

from gridalpha.config import load_config
from gridalpha.data import ercot
from gridalpha.data.build import build_features, leak_test, load_raw

CT = "US/Central"
LOCS = ["HB_HUBAVG", "LZ_NORTH", "LZ_HOUSTON", "LZ_SOUTH", "LZ_WEST"] + [f"RN_{i}" for i in range(8)]
CONS = ["WEST_TO_NORTH", "PANHANDLE_EXPORT", "HOUSTON_IMPORT"]


def _starts(day, freq):
    return pd.date_range(pd.Timestamp(day).tz_localize(CT), periods=24 * (60 // freq), freq=f"{freq}min")


class FakeAPI:
    def __init__(self, rng):
        self.rng = rng
        self.calls = 0

    def _prices(self, day, freq, noise):
        self.calls += 1
        rows = [(t, loc, 30 + 5 * (i % 3) + self.rng.normal(0, noise)) for t in _starts(day, freq)
                for i, loc in enumerate(LOCS)]
        return pd.DataFrame(rows, columns=["Interval Start", "Location", "SPP"])

    def get_spp_day_ahead_hourly(self, day):
        if pd.Timestamp(day) >= ercot._today_ct() + pd.Timedelta(days=1):
            raise RuntimeError("not published yet")          # tomorrow's DA before 13:30 CT
        return self._prices(day, 60, 2)

    def get_spp_real_time_15_min(self, day):
        return self._prices(day, 15, 6)

    def _shadow(self, day, col):
        rows = [(t, c, abs(self.rng.normal(0, 20))) for t in _starts(day, 60) for c in CONS if self.rng.random() < 0.3]
        return pd.DataFrame(rows, columns=[col, "Constraint Name", "Shadow Price"])

    def get_shadow_prices_dam(self, day):
        return self._shadow(day, "Interval Start")

    def get_shadow_prices_sced(self, day):
        return self._shadow(day, "SCED Timestamp")

    def get_load_forecast_by_model(self, pub_day):
        pubs = pd.date_range(pd.Timestamp(pub_day).tz_localize(CT), periods=24, freq="h")
        starts = pd.date_range(pd.Timestamp(pub_day).tz_localize(CT), periods=72, freq="h")
        rows = []
        for p in pubs:
            for m, flag in [("E", True), ("A3", False)]:
                for t in starts:
                    rows.append({"Publish Time": p, "Interval Start": t, "Model": m, "In Use Flag": flag,
                                 **{z: 5000.0 for z in ercot.WEATHER_TO_ZONE}})
        return pd.DataFrame(rows)

    def _regions(self, pub_day, prefix, regions):
        pubs = pd.date_range(pd.Timestamp(pub_day).tz_localize(CT), periods=24, freq="h")
        starts = pd.date_range(pd.Timestamp(pub_day).tz_localize(CT), periods=72, freq="h")
        return pd.DataFrame([{"Publish Time": p, "Interval Start": t, **{f"{prefix} {r}": 1000.0 for r in regions}}
                             for p in pubs for t in starts])

    def get_wind_actual_and_forecast_by_geographical_region_hourly(self, pub_day):
        return self._regions(pub_day, "STWPF", ercot.WIND_REGIONS)

    def get_solar_actual_and_forecast_by_geographical_region_hourly(self, pub_day):
        raise KeyError("archives")                            # an endpoint with no report for these days


@pytest.fixture
def fake(monkeypatch, tmp_path):
    api = FakeAPI(np.random.default_rng(0))
    monkeypatch.setattr(ercot, "_api", lambda: api)
    monkeypatch.setattr(ercot, "_open_meteo", lambda d: pd.DataFrame({z: np.full(24, 20.0) for z in ercot.ZONES}))
    monkeypatch.setattr(ercot, "_gas", lambda log=print: pd.Series(dtype=float))
    monkeypatch.setattr(ercot.time, "sleep", lambda s: None)
    return api, tmp_path


def test_fetch_assembles_a_valid_point_in_time_dataset(fake):
    api, tmp = fake
    start = (ercot._today_ct() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    cfg = load_config(overrides=[f"ercot.start={start}", "ercot.max_nodes=50", "features.bind_freq_window=10"])
    out = ercot.fetch(cfg, tmp / "raw", log=lambda *a: None)
    raw, meta = load_raw(out)
    D = len(meta["dates"])
    assert raw["da_lmp"].shape == (D, 24, len(LOCS))
    assert np.isnan(raw["da_lmp"][-1]).all()                  # tomorrow's DA not published yet
    assert np.isnan(raw["rt_lmp"][-2:]).all()                 # today and tomorrow have no full RT
    assert np.isfinite(raw["rt_lmp"][:-2]).mean() > 0.99      # a DST-change day can lack an hour
    assert raw["load_bus_fcst"][5].sum() == pytest.approx(8 * 5000.0 * 24, rel=1e-6)   # in-use model only
    assert meta["coverage"]["solar"] == 0.0 and meta["coverage"]["wind"] > 0.9
    F = build_features(raw, meta, cfg)
    for k in ["node", "glob", "hist", "edge"]:
        assert np.isfinite(F[k]).all()
    assert leak_test(raw, meta, cfg, n_checks=2)["passed"]

    # second run: old days come from the cache, only the last few days are fetched again
    calls = api.calls
    ercot.fetch(cfg, tmp / "raw", log=lambda *a: None)
    assert api.calls - calls <= 4 * 2
