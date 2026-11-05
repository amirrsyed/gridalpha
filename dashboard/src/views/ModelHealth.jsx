import { useMemo } from "react";
import { HTMLTable, Tag } from "@blueprintjs/core";
import Chart from "../Chart.jsx";
import { baseAxis, baseOption } from "../palette.js";
import { fmt } from "../dist.js";

const INTENT = { passed: "success", running: "primary", gate_failed: "warning", failed: "danger", pending: "none" };

export default function ModelHealth({ data, p }) {
  const md = data.model;
  const cal = data.calibration;
  const taus = cal.taus;

  const calOpt = useMemo(() => ({
    ...baseOption(p),
    tooltip: { ...baseOption(p).tooltip, valueFormatter: (v) => fmt.pct(v) },
    xAxis: { type: "value", min: 0, max: 1, name: "nominal quantile", nameLocation: "middle", nameGap: 24, ...baseAxis(p) },
    yAxis: { type: "value", min: 0, max: 1, ...baseAxis(p) },
    series: [
      { name: "perfect", type: "line", symbol: "none", data: [[0, 0], [1, 1]], lineStyle: { color: p.axis, type: "dashed" } },
      ...(cal.gnn_raw ? [{ name: "GNN before conformal", type: "line", data: taus.map((t, i) => [t, cal.gnn_raw[i]]), lineStyle: { color: p.gnn, type: "dotted" }, itemStyle: { color: p.gnn }, symbolSize: 4 }] : []),
      ...["gnn", "lgbm", "naive"].filter((m) => cal[m]).map((m) => ({
        name: m === "gnn" ? "GNN calibrated" : m === "lgbm" ? "LightGBM calibrated" : "Naive calibrated",
        type: "line", data: taus.map((t, i) => [t, cal[m][i]]), lineStyle: { color: p[m], width: m === "gnn" ? 2 : 1.2 }, itemStyle: { color: p[m] }, symbolSize: 5 })),
    ],
    legend: { ...baseOption(p).legend, type: "scroll" },
    grid: { left: 44, right: 16, top: 56, bottom: 40 },
  }), [cal, p]);

  const pitOpt = useMemo(() => {
    const h = cal.pit_hist ?? [];
    const tot = h.reduce((a, b) => a + b, 0) || 1;
    return {
      ...baseOption(p),
      grid: { left: 46, right: 16, top: 16, bottom: 30 },
      xAxis: { type: "category", data: h.map((_, i) => (i / h.length).toFixed(2)), ...baseAxis(p), splitLine: { show: false } },
      yAxis: { type: "value", ...baseAxis(p), axisLabel: { ...baseAxis(p).axisLabel, formatter: (v) => fmt.pct(v, 0) } },
      series: [{ type: "bar", data: h.map((x) => x / tot), itemStyle: { color: p.gnn, opacity: 0.8 },
        markLine: { symbol: "none", data: [{ yAxis: 1 / (h.length || 1) }], lineStyle: { color: p.muted, type: "dashed" }, label: { show: false } } }],
    };
  }, [cal, p]);

  const curveOpt = useMemo(() => {
    const ser = [];
    (md.curves ?? []).forEach((fold, fi) => fold.forEach((c, si) => ser.push({
      name: `fold ${fi} · seed ${si}`, type: "line", symbol: "none", data: c.map((e) => [e.epoch, e.val_pinball]),
      lineStyle: { width: 1.2, color: p.gnn, opacity: 0.35 + 0.65 * ((fi + 1) / (md.curves.length || 1)) } })));
    return {
      ...baseOption(p),
      legend: { show: false },
      tooltip: { trigger: "item", backgroundColor: p.panel, borderColor: p.grid, textStyle: { color: p.fg, fontSize: 12 }, formatter: (x) => `${x.seriesName}<br/>epoch ${x.value[0]}: $${x.value[1].toFixed(3)}` },
      xAxis: { type: "value", name: "epoch", nameLocation: "middle", nameGap: 24, minInterval: 1, min: 1, ...baseAxis(p) },
      yAxis: { type: "value", name: "val pinball $/MWh", scale: true, ...baseAxis(p) },
      grid: { left: 56, right: 16, top: 30, bottom: 40 },
      series: ser,
    };
  }, [md, p]);

  const stages = Object.entries(md.pipeline ?? {});
  const folds = md.folds?.gnn ?? [];
  return (
    <div className="grid12">
      <div className="panel span5">
        <h3>Autopilot pipeline</h3>
        <p className="caption">Each stage must pass its gate before the next runs. A failure opens a GitHub issue for the AI engineer.</p>
        <div className="stage-list">
          {stages.map(([s, st]) => (
            <div className="stage" key={s}>
              <span style={{ fontWeight: 600 }}>{s}</span>
              <span><Tag intent={INTENT[st.status] ?? "none"} minimal>{(st.status ?? "pending").replace("_", " ")}</Tag></span>
              <span className="checks">
                {st.gate ? Object.entries(st.gate.checks).map(([c, ok]) => <Tag key={c} minimal intent={ok ? "success" : "danger"} icon={ok ? "tick" : "cross"}>{c.replaceAll("_", " ")}</Tag>)
                  : <span className="legend-note">{st.secs != null ? `${fmt.num(st.secs, 0)} s` : ""}</span>}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="panel span7">
        <h3>Out-of-sample skill by fold</h3>
        <p className="caption">Pinball loss over all quantiles, $/MWh (lower is better), and how often the realized spread fell inside the calibrated 10–90% band (target 80%).</p>
        <div className="scroll-x">
          <HTMLTable compact striped style={{ width: "100%" }}>
            <thead><tr><th>Fold</th><th>Test window</th><th className="num">GNN</th><th className="num">LightGBM</th><th className="num">Naive</th><th className="num">GNN vs LGBM</th><th className="num">80% band</th></tr></thead>
            <tbody>{folds.map((f, i) => {
              const l = md.folds.lgbm?.[i], n = md.folds.naive?.[i];
              return (<tr key={i}><td>{f.fold}</td><td className="mono">{f.test_start} → {f.test_end}</td>
                <td className="num">{fmt.num(f.pinball, 3)}</td><td className="num">{l ? fmt.num(l.pinball, 3) : "–"}</td><td className="num">{n ? fmt.num(n.pinball, 3) : "–"}</td>
                <td className={`num ${l && f.pinball < l.pinball ? "pos" : "neg"}`}>{l ? fmt.pct(1 - f.pinball / l.pinball) : "–"}</td>
                <td className="num">{fmt.pct(f.coverage80)}</td></tr>);
            })}</tbody>
          </HTMLTable>
        </div>
      </div>
      <div className="panel span6"><h3>Quantile calibration</h3><p className="caption">Share of outcomes below each forecast quantile. Points on the dashed line mean the stated quantiles are honest, which Kelly sizing depends on.</p><Chart option={calOpt} height={280} /></div>
      <div className="panel span6"><h3>PIT histogram (GNN, calibrated)</h3><p className="caption">Where outcomes fell inside the forecast distribution. Flat at the dashed line is ideal; a U shape means bands are too narrow.</p><Chart option={pitOpt} height={280} /></div>
      <div className="panel span12"><h3>Training curves</h3><p className="caption">Validation pinball by epoch for every fold and ensemble seed (later folds darker). Training stops early when validation stops improving.</p><Chart option={curveOpt} height={240} /></div>
    </div>
  );
}
