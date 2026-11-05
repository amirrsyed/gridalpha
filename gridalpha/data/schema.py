"""Canonical raw-data schema shared by the simulator and real-ISO adapters.

A raw dataset is a directory with ``market.npz`` and ``meta.json``. Shapes use
D = operating days, H = 24 hours, N = pricing nodes, L = modeled lines, Z = zones,
G = generators. Arrays marked optional may be absent for real data; features and
losses that need them are switched off automatically.

Realized quantities (prices, shadow prices, injections) for day d are treated as
known only from the bid deadline of day d+2 onward (the latest complete operating
day at a D-1 10:00 deadline is D-2). Forecasts for day d are those published before
the deadline on d-1.
"""

REQUIRED = {
    "da_lmp": "[D,H,N] day-ahead LMP $/MWh",
    "rt_lmp": "[D,H,N] real-time LMP (hourly average) $/MWh",
    "da_energy": "[D,H] system energy component of DA LMP",
    "rt_energy": "[D,H] system energy component of RT LMP",
    "load_bus_fcst": "[D,H,N] load forecast allocated to nodes, MW",
    "wind_fcst_cf": "[D,H,Z] wind capacity-factor forecast by zone",
    "solar_fcst_cf": "[D,H,Z] solar capacity-factor forecast by zone",
    "temp_fcst": "[D,H,Z] temperature forecast by zone, C",
    "gas": "[D] gas price $/MMBtu (day-ahead index)",
    "planned_line": "[D,L] planned transmission outage flags",
    "rt_shadow": "[D,H,L] RT shadow prices (labels)",
    "da_shadow": "[D,H,L] DA shadow prices",
}

OPTIONAL = {
    "planned_gen": "[D,G] planned generator outages",
    "dc_mw": "[D,N] energized data-center capacity, MW",
    "da_margin": "[D,H] forecast reserve margin share",
    "op_lmp": "[D,H,N] trader's own DC-OPF LMP on forecasts",
    "op_energy": "[D,H] energy component of op_lmp",
    "op_shadow": "[D,H,L] shadow prices from the trader's own OPF",
    "rt_inj": "[D,H,N] RT nodal injections, MW (physics labels)",
}

META_KEYS = ["dates", "grid", "zones", "zone_of", "gens", "reactance", "source"]
