import { useMemo, useState } from "react";
import { Callout, Drawer, HTMLSelect, SegmentedControl, Slider, Tag } from "@blueprintjs/core";
import { Cell, Column, SelectionModes, Table2 } from "@blueprintjs/table";
import Chart from "../Chart.jsx";
import { baseAxis, baseOption } from "../palette.js";
import { densitySteps, fmt, growthCurve, kellyStar, payoffSamples } from "../dist.js";

// Normalize a live ticket or a backtest trade row into one shape.
function fromTrade(row, fields, nodes) {
  const t = Object.fromEntries(fields.map((f, i) => [f, row[i]]));
  const node = nodes[t.n];
  return {
    id: `${t.date}-HE${String(t.he).padStart(2, "0")}-${node.name}`, date: t.date, he: t.he, node: node.name,
    node_id: t.n, zone: node.zone, dc_node: node.dc, side: t.side, mw: t.mw, price_limit: t.limit, exp_da: t.exp_da,
    exp_rt: t.exp_rt, q: t.q, mean_edge: t.edge, p_profit: t.pp, kelly_applied: t.kf, kelly_single: t.ks,
    cleared: t.cleared, da: t.da, rt: t.rt, pnl: t.pnl, theme: t.theme, realized: true,
  };
}

const SORTS = {
  edge: (a, b) => b.mean_edge * b.mw - a.mean_edge * a.mw,
  mw: (a, b) => b.mw - a.mw,
  he: (a, b) => a.he - b.he || b.mw - a.mw,
  pnl: (a, b) => (b.pnl ?? 0) - (a.pnl ?? 0),
};

export default function Tickets({ data, p }) {
  const m = data.meta;
  const nodes = data.grid.nodes;
  const byDate = useMemo(() => {
    const out = {};
    const bt = data.backtest;
    for (const r of bt?.trades ?? []) {
      const t = fromTrade(r, bt.trade_fields, nodes);
      (out[t.date] ||= []).push(t);
    }
    return out;
  }, [data]);
  const liveDay = data.live?.operating_day;
  const days = useMemo(() => [...(liveDay ? [liveDay] : []), ...Object.keys(byDate).sort().reverse()], [byDate, liveDay]);
  const [day, setDay] = useState(days[0]);
  const [side, setSide] = useState("ALL");
  const [minEdge, setMinEdge] = useState(0);
  const [kappa, setKappa] = useState(m.kelly_fraction);
  const [sortBy, setSortBy] = useState("edge");
  const [open, setOpen] = useState(null);

  const isLive = day === liveDay;
  const base = isLive ? data.live.tickets.map((t) => ({ ...t, realized: false })) : byDate[day] ?? [];
  const scale = kappa / m.kelly_fraction;
  const rows = base
    .filter((t) => side === "ALL" || t.side === side)
    .filter((t) => t.mean_edge >= minEdge)
    .map((t) => ({ ...t, mw_whatif: Math.min(Math.round(t.mw * scale * 10) / 10, m.max_mw_node_hour) }))
    .sort(SORTS[sortBy]);

  const totMW = rows.reduce((s, t) => s + t.mw_whatif, 0);
  const expPnl = rows.reduce((s, t) => s + t.mean_edge * t.mw_whatif, 0);
  const realPnl = rows.reduce((s, t) => s + (t.pnl ?? 0) * (t.mw ? t.mw_whatif / t.mw : 0), 0);
  const cleared = rows.filter((t) => t.cleared).length;

  const cols = [
    ["Side", (t) => <span className={`side-${t.side}`}>{t.side}</span>, 60],
    ["Node", (t) => <>{t.node}{t.dc_node ? " ◆" : ""}</>, 84],
    ["Zone", (t) => t.zone, 96],
    ["HE", (t) => t.he, 46],
    ["MW", (t) => fmt.num(t.mw_whatif, 1), 60],
    ["Price limit", (t) => fmt.num(t.price_limit, 2), 88],
    ["Exp DA", (t) => fmt.num(t.exp_da, 2), 74],
    ["Exp RT", (t) => fmt.num(t.exp_rt, 2), 74],
    ["Edge $/MWh", (t) => fmt.num(t.mean_edge, 2), 90],
    ["P(profit)", (t) => fmt.pct(t.p_profit, 0), 74],
    ["Kelly f", (t) => fmt.num(t.kelly_applied * scale, 3), 70],
    ...(isLive
      ? [["VaR95", (t) => fmt.usd(t.var95_usd * scale), 74], ["Constraint", (t) => t.constraint, 140]]
      : [["Cleared", (t) => (t.cleared ? "yes" : "no"), 64], ["DA→RT", (t) => `${fmt.num(t.da, 1)}→${fmt.num(t.rt, 1)}`, 120],
         ["P&L", (t) => <span className={t.pnl >= 0 ? "pos" : "neg"}>{fmt.usd(t.pnl * (t.mw ? t.mw_whatif / t.mw : 0))}</span>, 82],
         ["Theme", (t) => t.theme, 130]]),
  ];

  return (
    <div className="grid12">
      <div className="panel span12">
        <div className="controls">
          <label className="ctl" htmlFor="tk-day">Operating day
            <HTMLSelect id="tk-day" value={day} onChange={(e) => setDay(e.target.value)}
              options={days.map((d) => ({ value: d, label: d === liveDay ? `${d} (next day, live)` : d }))} />
          </label>
          <div className="ctl">Side
            <SegmentedControl size="small" value={side} onValueChange={setSide}
              options={[{ label: "All", value: "ALL" }, { label: "DEC", value: "DEC" }, { label: "INC", value: "INC" }]} />
          </div>
          <label className="ctl" htmlFor="tk-sort">Sort
            <HTMLSelect id="tk-sort" value={sortBy} onChange={(e) => setSortBy(e.target.value)}
              options={[{ value: "edge", label: "Expected $ (edge × MW)" }, { value: "mw", label: "Size" }, { value: "he", label: "Hour" },
                ...(isLive ? [] : [{ value: "pnl", label: "Realized P&L" }])]} />
          </label>
          <div className="ctl">Min edge
            <div className="slider"><Slider min={0} max={20} stepSize={0.5} labelValues={[0, 10, 20]} value={minEdge} onChange={setMinEdge}
              labelRenderer={(v) => `$${v}`} /></div>
          </div>
          <div className="ctl">Kelly fraction κ (what-if)
            <div className="slider"><Slider min={0.05} max={1} stepSize={0.05} labelValues={[0.25, 0.5, 0.75, 1]} value={kappa} onChange={setKappa}
              labelRenderer={(v) => v.toFixed(2)} /></div>
          </div>
        </div>
        <Callout compact intent={isLive ? "primary" : "none"} icon={isLive ? "time" : "history"} style={{ marginBottom: 10 }}>
          {isLive
            ? <>{rows.length} tickets, {fmt.num(totMW, 1)} MW, expected P&amp;L <b>{fmt.usd(expPnl)}</b> at κ = {kappa.toFixed(2)}. Each virtual clears only inside its price limit. Click a row for its distribution and Kelly curve.</>
            : <>{rows.length} tickets, {cleared} cleared, realized P&amp;L <b className={realPnl >= 0 ? "pos" : "neg"}>{fmt.usd(realPnl)}</b> at κ = {kappa.toFixed(2)} (model expected {fmt.usd(expPnl)}).</>}
          {Math.abs(scale - 1) > 1e-6 && " What-if sizes scale the published sizes linearly and keep the per-node cap; the zone and VaR caps are not re-applied."}
        </Callout>
        <div className="table-wrap">
          <Table2 numRows={rows.length} enableRowHeader={false} columnWidths={cols.map((c) => c[2])} defaultRowHeight={26}
            cellRendererDependencies={[rows, kappa]} selectionModes={SelectionModes.ROWS_AND_CELLS}
            onSelection={(r) => { const row = r?.[0]?.rows?.[0]; if (row != null && rows[row]) setOpen(rows[row]); }}>
            {cols.map(([name, render]) => (
              <Column key={name} name={name} cellRenderer={(i) => <Cell>{rows[i] ? render(rows[i]) : ""}</Cell>} />
            ))}
          </Table2>
        </div>
        <p className="legend-note">◆ data-center node. DEC = buy day-ahead, sell real-time (profits when RT &gt; DA). INC = the reverse. Edge is the mean payoff per MWh after ${m.cost_per_mwh}/MWh fees and uplift.</p>
      </div>
      <TicketDrawer t={open} m={m} p={p} kappa={kappa} onClose={() => setOpen(null)} />
    </div>
  );
}

function TicketDrawer({ t, m, p, kappa, onClose }) {
  const narrow = typeof window !== "undefined" && window.innerWidth < 640;
  const calc = useMemo(() => {
    if (!t) return null;
    const s = t.side === "DEC" ? 1 : -1;
    const X = payoffSamples(t.q, m.quantiles, s, m.cost_per_mwh, 1200);
    const fs = t.kelly_single ?? kellyStar(X, m.risk_per_mwh);
    return { X, fs, curve: t.growth_curve ?? growthCurve(X, m.risk_per_mwh, fs) };
  }, [t, m]);
  if (!t) return <Drawer isOpen={false} onClose={onClose} />;
  const s = t.side === "DEC" ? 1 : -1;
  const mid = m.quantiles.indexOf(0.5);
  const applied = t.kelly_applied * (kappa / m.kelly_fraction);
  const dens = densitySteps(t.q, m.quantiles);
  const lines = [
    { xAxis: 0, name: "0", lineStyle: { color: p.axis } },
    { xAxis: s * m.cost_per_mwh, name: "break-even", lineStyle: { color: p.muted, type: "dashed" } },
    { xAxis: t.q[mid], name: "median", lineStyle: { color: p.gnn } },
  ];
  if (t.realized) lines.push({ xAxis: t.rt - t.da, name: "realized", lineStyle: { color: t.pnl >= 0 ? p.good : p.bad, width: 2 } });
  const distOpt = {
    ...baseOption(p),
    tooltip: { trigger: "axis", backgroundColor: p.panel, borderColor: p.grid, textStyle: { color: p.fg }, valueFormatter: (v) => v.toFixed(4) },
    grid: { left: 52, right: 16, top: 24, bottom: 40 },
    xAxis: { type: "value", name: "RT − DA, $/MWh", nameLocation: "middle", nameGap: 26, ...baseAxis(p), splitLine: { show: false } },
    yAxis: { type: "value", name: "density", ...baseAxis(p) },
    series: [{ type: "line", data: dens, step: false, symbol: "none", areaStyle: { color: p.gnn, opacity: 0.18 },
      lineStyle: { color: p.gnn, width: 1.5 },
      markLine: { symbol: "none", label: { show: false }, data: lines } }],
  };
  const gOpt = {
    ...baseOption(p),
    grid: { left: 70, right: 16, top: 24, bottom: 40 },
    tooltip: { trigger: "axis", backgroundColor: p.panel, borderColor: p.grid, textStyle: { color: p.fg }, valueFormatter: (v) => v.toExponential(2) },
    xAxis: { type: "value", name: "fraction of bankroll f", nameLocation: "middle", nameGap: 26, ...baseAxis(p), splitLine: { show: false } },
    yAxis: { type: "value", scale: true, ...baseAxis(p) },
    series: [{ type: "line", data: calc.curve, symbol: "none", lineStyle: { color: p.gnn, width: 2 },
      markLine: { symbol: "none", label: { color: p.muted, fontSize: 10, formatter: "{b}" },
        data: [{ xAxis: calc.fs, name: "full Kelly f*", lineStyle: { color: p.gnn, type: "dashed" } },
               { xAxis: Math.min(applied, calc.curve.at(-1)?.[0] ?? applied), name: "applied", lineStyle: { color: p.good } }] } }],
  };
  const drivers = t.drivers ?? [];
  const dmax = Math.max(1e-6, ...drivers.map((d) => Math.abs(d.usd)));
  return (
    <Drawer isOpen onClose={onClose} size={narrow ? "100%" : 600} title={`${t.side} ${t.mw} MW · ${t.node} · HE${t.he} · ${t.date ?? ""}`}
      icon={t.side === "DEC" ? "trending-up" : "trending-down"}>
      <div className="drawer-body">
        <p className="plain">
          {t.side === "DEC" ? "Bid to buy" : "Offer to sell"} {t.mw} MW day-ahead at <b>{t.node}</b> ({t.zone}) for hour ending {t.he},
          {" "}{t.side === "DEC" ? "and sell it back" : "and buy it back"} in real time. The order {t.side === "DEC" ? "clears only if the DA price is at or below" : "clears only if the DA price is at or above"}{" "}
          <b className="mono">{fmt.usd(t.price_limit, 2)}</b>; beyond that the expected edge is gone.
        </p>
        <dl className="kv">
          <dt>Expected DA / RT</dt><dd>{fmt.usd(t.exp_da, 2)} / {fmt.usd(t.exp_rt, 2)}</dd>
          <dt>Mean edge after costs</dt><dd>${fmt.num(t.mean_edge)} per MWh</dd>
          <dt>P(profit)</dt><dd>{fmt.pct(t.p_profit)}</dd>
          <dt>Spread quantiles</dt><dd>q10 {fmt.signed(t.q[2])} · q50 {fmt.signed(t.q[mid])} · q90 {fmt.signed(t.q[6])}</dd>
          <dt>Kelly (single / applied)</dt><dd>{fmt.num(calc.fs, 3)} / {fmt.num(applied, 3)}</dd>
          {t.var95_usd != null && <><dt>95% VaR / q01 loss</dt><dd>{fmt.usd(t.var95_usd)} / {fmt.usd(t.loss_q01_usd)}</dd></>}
          {t.constraint && <><dt>Top constraint</dt><dd>{t.constraint} (exposure {fmt.num(t.constraint_exposure, 3)})</dd></>}
          {t.ens_disagreement != null && <><dt>Ensemble disagreement</dt><dd>±${fmt.num(t.ens_disagreement)} on the median</dd></>}
          {t.realized && <><dt>Realized DA → RT</dt><dd>{fmt.usd(t.da, 2)} → {fmt.usd(t.rt, 2)} ({t.cleared ? "cleared" : "did not clear"})</dd>
            <dt>Realized P&amp;L</dt><dd className={t.pnl >= 0 ? "pos" : "neg"}>{fmt.usd(t.pnl)}</dd>
            <dt>Theme</dt><dd>{t.theme}</dd></>}
          {t.valid_until && <><dt>Valid until</dt><dd>{t.valid_until}</dd></>}
          {t.model_version && <><dt>Model</dt><dd>{t.model_version}</dd></>}
        </dl>
        <section>
          <h4>Predictive distribution of RT − DA</h4>
          <Chart option={distOpt} height={210} />
          <p className="legend-note">Solid purple: forecast median. Dashed: break-even after costs.{t.realized ? " Green or red: what actually happened." : ""}</p>
        </section>
        <section>
          <h4>Kelly growth curve</h4>
          <p className="legend-note">Expected log growth of the bankroll (vertical axis) versus the fraction staked on this trade alone. Past f* the curve falls: overbetting loses money even with a real edge.</p>
          <Chart option={gOpt} height={200} />
        </section>
        {drivers.length > 0 && (
          <section>
            <h4>Drivers</h4>
            <p className="legend-note">Change in the forecast median when each input group is reset to its average. Positive supports this side.</p>
            {drivers.map((d) => (
              <div className="bar-cell" key={d.group} style={{ fontSize: 13 }}>
                <span style={{ width: 170 }}>{d.group}</span>
                <span className="bar" style={{ width: `${(Math.abs(d.usd) / dmax) * 140}px`, background: d.usd >= 0 ? p.good : p.bad }} />
                <span className="mono">{fmt.signed(d.usd)} $/MWh</span>
              </div>
            ))}
          </section>
        )}
        <div><Tag minimal>{t.side === "DEC" ? "Virtual demand bid" : "Virtual supply offer"}</Tag></div>
      </div>
    </Drawer>
  );
}
