import { useMemo, useState } from "react";
import { HTMLTable, SegmentedControl } from "@blueprintjs/core";
import Chart from "../Chart.jsx";
import { baseAxis, baseOption } from "../palette.js";
import { fmt, kkey, kusd } from "../dist.js";

const NAMES = { gnn: "ST-GNN (ensemble)", lgbm: "LightGBM", naive: "Naive (30-day quantiles)" };

export default function Backtest({ data, p }) {
  const bt = data.backtest;
  const m = data.meta;
  const [k, setK] = useState(kkey(m.kelly_fraction));
  const models = Object.keys(bt.models);
  const dates = bt.dates ?? [];
  const cap0 = m.capital0;

  const equity = useMemo(() => {
    const opt = baseOption(p);
    return {
      ...opt,
      tooltip: { ...opt.tooltip, valueFormatter: (v) => fmt.usd(v) },
      xAxis: { type: "category", data: dates, ...baseAxis(p), axisLabel: { ...baseAxis(p).axisLabel, hideOverlap: true }, splitLine: { show: false } },
      yAxis: { type: "value", scale: true, ...baseAxis(p), axisLabel: { ...baseAxis(p).axisLabel, formatter: kusd }, name: "cumulative P&L" },
      series: models.map((mm) => {
        let c = 0;
        return { name: NAMES[mm], type: "line", symbol: "none", lineStyle: { width: mm === "gnn" ? 2.2 : 1.4, color: p[mm] },
          itemStyle: { color: p[mm] }, data: bt.models[mm].daily[k].map((x) => (c += x)),
          endLabel: { show: true, color: p[mm], fontFamily: "IBM Plex Mono, monospace", fontSize: 11, formatter: (x) => fmt.usd(x.value) } };
      }),
      grid: { left: 60, right: 90, top: 30, bottom: 34 },
    };
  }, [bt, k, p]);

  const dd = useMemo(() => {
    let c = cap0, peak = cap0;
    const v = bt.models.gnn.daily[k].map((x) => { c += x; peak = Math.max(peak, c); return -(peak - c) / peak; });
    return {
      ...baseOption(p),
      tooltip: { ...baseOption(p).tooltip, valueFormatter: (x) => fmt.pct(x) },
      grid: { left: 60, right: 18, top: 14, bottom: 30 },
      xAxis: { type: "category", data: dates, ...baseAxis(p), axisLabel: { ...baseAxis(p).axisLabel, hideOverlap: true }, splitLine: { show: false } },
      yAxis: { type: "value", ...baseAxis(p), axisLabel: { ...baseAxis(p).axisLabel, formatter: (x) => `${(x * 100).toFixed(1)}%` } },
      legend: { show: false },
      series: [{ name: "Drawdown", type: "line", symbol: "none", data: v, lineStyle: { color: p.bad, width: 1 }, areaStyle: { color: p.bad, opacity: 0.2 } }],
    };
  }, [bt, k, p]);

  const monthly = useMemo(() => {
    const agg = {};
    bt.models.gnn.daily[k].forEach((x, i) => { const mo = dates[i].slice(0, 7); agg[mo] = (agg[mo] ?? 0) + x; });
    const keys = Object.keys(agg);
    return {
      ...baseOption(p),
      tooltip: { ...baseOption(p).tooltip, valueFormatter: (v) => fmt.usd(v) },
      grid: { left: 60, right: 18, top: 14, bottom: 30 },
      xAxis: { type: "category", data: keys, ...baseAxis(p), splitLine: { show: false } },
      yAxis: { type: "value", ...baseAxis(p), axisLabel: { ...baseAxis(p).axisLabel, formatter: kusd } },
      legend: { show: false },
      series: [{ type: "bar", name: "GNN monthly P&L", data: keys.map((x) => ({ value: agg[x], itemStyle: { color: agg[x] >= 0 ? p.good : p.bad } })) }],
    };
  }, [bt, k, p]);

  const themes = useMemo(() => {
    const th = bt.models.gnn.metrics[k].by_theme;
    const names = Object.keys(th);
    return {
      ...baseOption(p),
      tooltip: { ...baseOption(p).tooltip, valueFormatter: (v) => fmt.usd(v) },
      grid: { left: 130, right: 30, top: 10, bottom: 30 },
      yAxis: { type: "category", data: names, ...baseAxis(p), splitLine: { show: false } },
      xAxis: { type: "value", ...baseAxis(p), axisLabel: { ...baseAxis(p).axisLabel, formatter: kusd } },
      series: [{ type: "bar", data: names.map((n) => ({ value: th[n], itemStyle: { color: th[n] >= 0 ? p.gnn : p.bad } })) }],
    };
  }, [bt, k, p]);

  const kellyCmp = data.meta.kappas.map((kk) => ({ kk, ...bt.models.gnn.metrics[kkey(kk)] }));

  return (
    <div className="grid12">
      <div className="panel span12">
        <div className="controls">
          <h3 style={{ margin: 0 }}>Out-of-sample cumulative P&amp;L, walk-forward</h3>
          <div className="ctl">Kelly fraction κ
            <SegmentedControl size="small" value={k} onValueChange={setK}
              options={data.meta.kappas.map((x) => ({ label: x === 1 ? "Full" : String(x), value: kkey(x) }))} />
          </div>
        </div>
        <p className="caption">Every model is sized with the same Kelly rules and caps on a fixed {fmt.usd(cap0)} bankroll. Models retrain at each fold boundary using only data available at the time.</p>
        <Chart option={equity} height={300} />
      </div>
      <div className="panel span6"><h3>GNN drawdown</h3><Chart option={dd} height={180} /></div>
      <div className="panel span6"><h3>GNN monthly P&amp;L</h3><Chart option={monthly} height={180} /></div>
      <div className="panel span7">
        <h3>Results by model</h3>
        <div className="scroll-x">
          <HTMLTable compact striped style={{ width: "100%" }}>
            <thead><tr><th>Model</th><th className="num">P&amp;L</th><th className="num">Return</th><th className="num">Sharpe</th><th className="num">Max DD</th><th className="num">Hit rate</th><th className="num">Tickets</th><th className="num">Cleared MWh</th><th className="num">Worst day</th></tr></thead>
            <tbody>{models.map((mm) => { const r = bt.models[mm].metrics[k]; return (
              <tr key={mm}><td style={{ color: p[mm], fontWeight: 600 }}>{NAMES[mm]}</td>
                <td className={`num ${r.pnl >= 0 ? "pos" : "neg"}`}>{fmt.usd(r.pnl)}</td><td className="num">{fmt.pct(r.return)}</td>
                <td className="num">{fmt.num(r.sharpe)}</td><td className="num">{fmt.pct(r.max_drawdown)}</td><td className="num">{fmt.pct(r.hit_rate)}</td>
                <td className="num">{r.n_tickets.toLocaleString()}</td><td className="num">{fmt.num(r.mwh, 0)}</td><td className="num neg">{fmt.usd(r.worst_day)}</td></tr>); })}
            </tbody>
          </HTMLTable>
        </div>
        <h3 style={{ marginTop: 16 }}>Kelly fraction trade-off (GNN)</h3>
        <div className="scroll-x">
          <HTMLTable compact striped style={{ width: "100%" }}>
            <thead><tr><th>κ</th><th className="num">P&amp;L</th><th className="num">Sharpe</th><th className="num">Max DD</th><th className="num">Worst day</th><th className="num">At size cap</th></tr></thead>
            <tbody>{kellyCmp.map((r) => (
              <tr key={r.kk}><td>{r.kk === 1 ? "Full Kelly" : `${r.kk} × Kelly`}</td><td className={`num ${r.pnl >= 0 ? "pos" : "neg"}`}>{fmt.usd(r.pnl)}</td>
                <td className="num">{fmt.num(r.sharpe)}</td><td className="num">{fmt.pct(r.max_drawdown)}</td><td className="num neg">{fmt.usd(r.worst_day)}</td><td className="num">{r.share_at_cap != null ? fmt.pct(r.share_at_cap, 0) : "–"}</td></tr>))}
            </tbody>
          </HTMLTable>
        </div>
      </div>
      <div className="panel span5">
        <h3>GNN P&amp;L by <i>Power 2026</i> theme</h3>
        <p className="caption">Each trade is tagged by what was happening at its node and hour: scarcity pricing, negative prices, a data-center load pocket, or the evening ramp.</p>
        <Chart option={themes} height={220} />
      </div>
    </div>
  );
}
