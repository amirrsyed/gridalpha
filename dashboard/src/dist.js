// Predictive-distribution math shared by the views. Mirrors gridalpha/trade/kelly.py:
// piecewise-linear quantile function with exponential tails, quasi-Monte-Carlo sampling,
// and expected log growth E[log(1 + f X / C)].

export function quantileFn(q, taus) {
  const n = taus.length;
  const bLo = Math.max((q[1] - q[0]) / Math.log(taus[1] / taus[0]), 1e-6);
  const bHi = Math.max((q[n - 1] - q[n - 2]) / Math.log((1 - taus[n - 2]) / (1 - taus[n - 1])), 1e-6);
  return (u) => {
    if (u < taus[0]) return q[0] - bLo * Math.log(taus[0] / Math.max(u, 1e-9));
    if (u > taus[n - 1]) return q[n - 1] + bHi * Math.log((1 - taus[n - 1]) / Math.max(1 - u, 1e-9));
    let k = 0;
    while (k < n - 2 && u > taus[k + 1]) k++;
    const w = (u - taus[k]) / (taus[k + 1] - taus[k]);
    return q[k] * (1 - w) + q[k + 1] * w;
  };
}

// payoff samples per MWh for a side (+1 DEC, -1 INC), net of cost
export function payoffSamples(q, taus, side, cost, S = 800) {
  const Q = quantileFn(q, taus);
  const out = new Float64Array(S);
  for (let i = 0; i < S; i++) out[i] = side * Q((i + 0.5) / S) - cost;
  return out;
}

export function growth(f, X, C) {
  let s = 0;
  for (let i = 0; i < X.length; i++) {
    const r = 1 + (f * X[i]) / C;
    if (r <= 0) return -Infinity;
    s += Math.log(r);
  }
  return s / X.length;
}

export function kellyStar(X, C) {
  let mean = 0;
  let minX = Infinity;
  for (const x of X) { mean += x; if (x < minX) minX = x; }
  mean /= X.length;
  if (mean <= 0) return 0;
  let hi = minX < 0 ? (0.999 * C) / -minX : 10;
  hi = Math.min(hi, 10);
  let lo = 0;
  for (let it = 0; it < 80; it++) {            // golden-section search
    const a = lo + 0.382 * (hi - lo);
    const b = lo + 0.618 * (hi - lo);
    if (growth(a, X, C) < growth(b, X, C)) lo = a; else hi = b;
  }
  return (lo + hi) / 2;
}

export function growthCurve(X, C, fStar, points = 40) {
  const top = Math.max(2 * fStar, 1e-3);
  const out = [];
  for (let i = 0; i <= points; i++) {
    const f = (top * i) / points;
    const g = growth(f, X, C);
    if (Number.isFinite(g)) out.push([f, g]);
  }
  return out;
}

// density between adjacent quantiles, for a stepped distribution plot
export function densitySteps(q, taus) {
  const pts = [];
  for (let k = 0; k < q.length - 1; k++) {
    const w = Math.max(q[k + 1] - q[k], 1e-6);
    const d = (taus[k + 1] - taus[k]) / w;
    pts.push([q[k], d], [q[k + 1], d]);
  }
  return pts;
}

export const fmt = {
  usd: (x, d = 0) => (x < 0 ? "-$" : "$") + Math.abs(x).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }),
  num: (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : x.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d })),
  pct: (x, d = 1) => (x * 100).toFixed(d) + "%",
  signed: (x, d = 2) => (x > 0 ? "+" : "") + x.toFixed(d),
};

// Python writes Kelly-fraction keys as "1.0", "0.5", "0.25"
export const kkey = (x) => (Number.isInteger(Number(x)) ? Number(x).toFixed(1) : String(Number(x)));
export const kusd = (v) => (Math.abs(v) >= 1000 ? `${v < 0 ? "-" : ""}$${(Math.abs(v) / 1000).toFixed(0)}k` : `${v < 0 ? "-" : ""}$${Math.abs(v).toFixed(0)}`);
