"""A Texas-like synthetic transmission network.

Seven zones on rough Texas geography: wind in the West and Panhandle, solar in the
Far West, load in North/Houston/Central/South, AI data-center sinks in North and
Central, and thin export corridors so congestion and negative prices arise from the
physics rather than being painted on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ZONES = {
    #            lon range          lat range        buses  load share
    "FAR_WEST":  ((-104.0, -101.2), (30.2, 32.4), 5, 0.06),
    "WEST":      ((-101.0, -98.8), (31.2, 33.4), 7, 0.03),
    "PANHANDLE": ((-102.8, -100.2), (34.2, 36.3), 5, 0.02),
    "NORTH":     ((-98.4, -95.6), (32.1, 33.9), 10, 0.33),
    "CENTRAL":   ((-98.6, -96.9), (29.9, 31.7), 6, 0.12),
    "HOUSTON":   ((-96.0, -94.6), (29.1, 30.4), 8, 0.27),
    "SOUTH":     ((-99.8, -97.1), (26.3, 29.4), 7, 0.17),
}

# inter-zone ties: (zone_a, zone_b, number of lines, rating MW per line)
TIES = [
    ("PANHANDLE", "WEST", 2, 650.0),
    ("PANHANDLE", "NORTH", 2, 600.0),
    ("FAR_WEST", "WEST", 2, 700.0),
    ("FAR_WEST", "SOUTH", 1, 450.0),
    ("WEST", "NORTH", 2, 750.0),
    ("WEST", "CENTRAL", 2, 600.0),
    ("NORTH", "CENTRAL", 2, 900.0),
    ("NORTH", "HOUSTON", 2, 1000.0),
    ("CENTRAL", "HOUSTON", 1, 700.0),
    ("CENTRAL", "SOUTH", 2, 800.0),
    ("SOUTH", "HOUSTON", 2, 750.0),
]

# generator fleet by zone: (fuel, total MW, units, heat rate low-high or fixed $/MWh offer)
FLEET = {
    "FAR_WEST":  [("solar", 2000, 4), ("gas_ct", 450, 2)],
    "WEST":      [("wind", 3600, 6), ("solar", 1400, 3), ("gas_ct", 300, 1)],
    "PANHANDLE": [("wind", 2900, 5)],
    "NORTH":     [("nuclear", 1200, 1), ("coal", 800, 1), ("gas_cc", 2600, 4), ("gas_ct", 1300, 4)],
    "CENTRAL":   [("coal", 1500, 2), ("gas_cc", 900, 2), ("gas_ct", 600, 2), ("solar", 600, 2)],
    "HOUSTON":   [("nuclear", 1300, 1), ("gas_cc", 3600, 6), ("gas_ct", 1300, 4)],
    "SOUTH":     [("wind", 1300, 3), ("solar", 1200, 3), ("gas_cc", 1500, 3), ("gas_ct", 600, 2)],
}

HEAT_RATE = {"gas_cc": (6.6, 7.6), "gas_ct": (9.5, 11.5), "coal": (9.8, 10.6)}
FIXED_OFFER = {"wind": -15.0, "solar": -5.0, "nuclear": 9.0}
VOM = {"gas_cc": 3.0, "gas_ct": 5.0, "coal": 4.0}
COAL_PRICE = 2.1  # $/MMBtu equivalent

# AI data centers: (zone, MW at first energization, MW at full build, energize date, full date)
DATA_CENTERS = [
    ("NORTH", 220.0, 650.0, "2023-06-01", "2026-06-01"),
    ("NORTH", 0.0, 420.0, "2025-02-01", "2026-09-01"),
    ("CENTRAL", 150.0, 500.0, "2023-09-01", "2025-12-01"),
    ("CENTRAL", 0.0, 380.0, "2025-08-01", "2026-10-15"),
    ("WEST", 0.0, 250.0, "2024-10-01", "2025-10-01"),
]


@dataclass
class Grid:
    names: list[str]
    zone_of: np.ndarray            # [N] zone index
    zone_names: list[str]
    xy: np.ndarray                 # [N, 2] lon, lat
    load_share: np.ndarray         # [N] share of system (non-DC) load
    lines: np.ndarray              # [L, 2] from, to
    reactance: np.ndarray          # [L]
    rating: np.ndarray             # [L] MW
    is_tie: np.ndarray             # [L] bool
    gens: dict                     # arrays: bus, fuel, cap, hr, vom, fixed
    dc_bus: np.ndarray             # [K] bus of each data center
    dc_spec: list
    slack: int = 0
    _ptdf_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def m(self) -> int:
        return len(self.lines)

    def susceptance(self) -> np.ndarray:
        return 1.0 / self.reactance

    def ptdf(self, out: tuple[int, ...] = ()) -> np.ndarray:
        """Power transfer distribution factors [L, N] with lines in ``out`` removed.

        Removed lines get a zero row. Withdrawals are at the slack bus.
        """
        key = tuple(sorted(out))
        if key in self._ptdf_cache:
            return self._ptdf_cache[key]
        b = self.susceptance().copy()
        b[list(key)] = 0.0
        n = self.n
        A = np.zeros((self.m, n))
        A[np.arange(self.m), self.lines[:, 0]] = 1.0
        A[np.arange(self.m), self.lines[:, 1]] = -1.0
        Bbus = A.T @ (b[:, None] * A)
        keep = [i for i in range(n) if i != self.slack]
        Binv = np.zeros((n, n))
        Binv[np.ix_(keep, keep)] = np.linalg.pinv(Bbus[np.ix_(keep, keep)])
        P = (b[:, None] * A) @ Binv
        if len(self._ptdf_cache) < 4096:
            self._ptdf_cache[key] = P
        return P

    def connected_without(self, out: tuple[int, ...]) -> bool:
        keep = np.ones(self.m, bool)
        keep[list(out)] = False
        adj = [[] for _ in range(self.n)]
        for (a, b) in self.lines[keep]:
            adj[a].append(b)
            adj[b].append(a)
        seen = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return len(seen) == self.n

    def bus_matrix(self) -> np.ndarray:
        """Nodal susceptance matrix B [N, N] (for the physics loss)."""
        b = self.susceptance()
        n = self.n
        A = np.zeros((self.m, n))
        A[np.arange(self.m), self.lines[:, 0]] = 1.0
        A[np.arange(self.m), self.lines[:, 1]] = -1.0
        return A.T @ (b[:, None] * A)

    def to_json(self) -> dict:
        return {
            "nodes": [
                {"id": i, "name": self.names[i], "zone": self.zone_names[self.zone_of[i]],
                 "lon": round(float(self.xy[i, 0]), 3), "lat": round(float(self.xy[i, 1]), 3),
                 "dc": bool(i in set(self.dc_bus.tolist()))}
                for i in range(self.n)
            ],
            "lines": [
                {"id": l, "from": int(a), "to": int(b), "rating": float(self.rating[l]),
                 "tie": bool(self.is_tie[l])}
                for l, (a, b) in enumerate(self.lines)
            ],
        }


def _dist_km(a: np.ndarray, b: np.ndarray) -> float:
    dx = (a[0] - b[0]) * 95.0
    dy = (a[1] - b[1]) * 111.0
    return float(np.hypot(dx, dy))


def build_grid(seed: int = 7) -> Grid:
    rng = np.random.default_rng(seed)
    names, zone_of, xy, share = [], [], [], []
    zone_names = list(ZONES)
    for zi, (z, ((x0, x1), (y0, y1), nb, sh)) in enumerate(ZONES.items()):
        w = rng.dirichlet(np.ones(nb) * 2.0)
        for k in range(nb):
            names.append(f"{z[:2]}{k + 1:02d}" if z != "PANHANDLE" else f"PH{k + 1:02d}")
            zone_of.append(zi)
            xy.append([rng.uniform(x0, x1), rng.uniform(y0, y1)])
            share.append(sh * w[k])
    zone_of = np.array(zone_of)
    xy = np.array(xy)
    share = np.array(share)
    share /= share.sum()

    lines, ratings, ties = [], [], []
    seen = set()

    def add(a, b, rating, tie):
        key = (min(a, b), max(a, b))
        if a == b or key in seen:
            return False
        seen.add(key)
        lines.append(key)
        ratings.append(rating)
        ties.append(tie)
        return True

    # intra-zone: minimum spanning tree + one extra nearest neighbor each
    for zi in range(len(zone_names)):
        idx = np.where(zone_of == zi)[0]
        in_tree = [idx[0]]
        rest = list(idx[1:])
        while rest:
            best = min(((a, b) for a in in_tree for b in rest), key=lambda p: _dist_km(xy[p[0]], xy[p[1]]))
            add(best[0], best[1], rng.uniform(800, 1200), False)
            in_tree.append(best[1])
            rest.remove(best[1])
        for a in idx:
            others = sorted((b for b in idx if b != a), key=lambda b: _dist_km(xy[a], xy[b]))
            for b in others[:2]:
                if add(a, b, rng.uniform(700, 1100), False):
                    break
    # inter-zone ties: closest distinct bus pairs
    for za, zb, k, rating in TIES:
        ia = np.where(zone_of == zone_names.index(za))[0]
        ib = np.where(zone_of == zone_names.index(zb))[0]
        pairs = sorted(((a, b) for a in ia for b in ib), key=lambda p: _dist_km(xy[p[0]], xy[p[1]]))
        used_a, used_b, made = set(), set(), 0
        for a, b in pairs:
            if a in used_a or b in used_b:
                continue
            if add(a, b, rating * rng.uniform(0.9, 1.1), True):
                used_a.add(a)
                used_b.add(b)
                made += 1
            if made == k:
                break

    lines = np.array(lines)
    reactance = np.array([_dist_km(xy[a], xy[b]) for a, b in lines]) / 100.0 + 0.05

    # generators
    g_bus, g_fuel, g_cap, g_hr, g_vom, g_fixed = [], [], [], [], [], []
    for z, units in FLEET.items():
        idx = np.where(zone_of == zone_names.index(z))[0]
        for fuel, total, count in units:
            caps = rng.dirichlet(np.ones(count) * 3) * total
            for c in caps:
                g_bus.append(int(rng.choice(idx)))
                g_fuel.append(fuel)
                g_cap.append(float(c))
                if fuel in HEAT_RATE:
                    g_hr.append(float(rng.uniform(*HEAT_RATE[fuel])))
                    g_vom.append(VOM[fuel])
                    g_fixed.append(np.nan)
                else:
                    g_hr.append(np.nan)
                    g_vom.append(0.0)
                    g_fixed.append(FIXED_OFFER[fuel] + rng.uniform(-2, 2))
    gens = {
        "bus": np.array(g_bus), "fuel": np.array(g_fuel), "cap": np.array(g_cap),
        "hr": np.array(g_hr), "vom": np.array(g_vom), "fixed": np.array(g_fixed),
    }

    # data centers: put each on the highest-load bus of its zone not already used
    dc_bus = []
    for zone, *_ in DATA_CENTERS:
        idx = [i for i in np.where(zone_of == zone_names.index(zone))[0] if i not in dc_bus]
        dc_bus.append(int(max(idx, key=lambda i: share[i])))

    slack = int(np.where(zone_of == zone_names.index("NORTH"))[0][0])
    return Grid(names=names, zone_of=zone_of, zone_names=zone_names, xy=xy, load_share=share,
                lines=lines, reactance=reactance, rating=np.array(ratings), is_tie=np.array(ties),
                gens=gens, dc_bus=np.array(dc_bus), dc_spec=DATA_CENTERS, slack=slack)
