import { useEffect, useMemo, useState } from "react";
import { Button, HTMLSelect, HTMLTable, SegmentedControl, Slider } from "@blueprintjs/core";
import Chart from "../Chart.jsx";
import { fmt } from "../dist.js";

export default function GridMap({ data, p }) {
  const { nodes, lines } = data.grid;
  const maps = data.maps ?? { dates: [] };
  const liveDay = data.live?.operating_day;
  const days = [...(liveDay ? [liveDay] : []), ...[...maps.dates].reverse()];
  const [day, setDay] = useState(days[0]);
  const [layer, setLayer] = useState("pred");
  const [links, setLinks] = useState("physical");
  const [hour, setHour] = useState(18);
  const [playing, setPlaying] = useState(false);
  const isLive = day === liveDay;

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setHour((h) => (h % 24) + 1), 700);
    return () => clearInterval(id);
  }, [playing]);

  const view = useMemo(() => {
    if (isLive) {
      const g = data.live.grid_forecast;
      return { vals: g.median_spread[hour - 1], bind: g.bind[hour - 1] };
    }
    const i = maps.dates.indexOf(day);
    const src = layer === "real" ? maps.real : maps.pred;
    return { vals: src[i][hour - 1], bind: maps.bind[i][hour - 1] };
  }, [data, day, layer, hour, isLive]);

  const hasGeo = nodes[0]?.lon != null;
  const option = useMemo(() => {
    const vmax = 25;
    const gnodes = nodes.map((n, i) => {
      const v = view.vals[i];
      return {
        id: String(i), name: n.name, value: v,
        x: hasGeo ? n.lon : Math.cos((2 * Math.PI * i) / nodes.length),
        y: hasGeo ? -n.lat : Math.sin((2 * Math.PI * i) / nodes.length),
        symbol: n.dc ? "diamond" : "circle",
        symbolSize: (n.dc ? 16 : 9) + Math.min(Math.abs(v), 40) * 0.45,
        itemStyle: { borderColor: n.dc ? p.fg : p.panel, borderWidth: n.dc ? 1.5 : 1 },
        zone: n.zone,
      };
    });
    const phys = links === "learned" ? [] : lines.map((l, k) => {
      const b = view.bind[k] ?? 0;
      return { source: String(l.from), target: String(l.to), value: b,
        lineStyle: { width: 0.8 + 5 * b, color: b > 0.6 ? p.bad : b > 0.3 ? p.inc : p.axis, opacity: 0.35 + 0.6 * b, curveness: 0 } };
    });
    const learned = links === "physical" ? [] : (data.model?.learned_edges ?? []).map(([a, b, w]) => ({
      source: String(a), target: String(b), value: w,
      lineStyle: { width: 0.6 + 3 * w, color: p.learned, type: "dashed", opacity: 0.55, curveness: 0.25 } }));
    return {
      backgroundColor: "transparent", animation: false,
      tooltip: {
        backgroundColor: p.panel, borderColor: p.grid, textStyle: { color: p.fg, fontSize: 12 },
        formatter: (x) => x.dataType === "edge"
          ? `${nodes[+x.data.source].name} – ${nodes[+x.data.target].name}<br/>${x.data.lineStyle.type === "dashed" ? "learned link weight" : "P(binding)"}: ${fmt.num(x.data.value, 2)}`
          : `<b>${x.data.name}</b> · ${x.data.zone}<br/>${layer === "real" && !isLive ? "realized" : "forecast median"} RT − DA: ${fmt.signed(x.data.value)} $/MWh`,
      },
      visualMap: { type: "continuous", min: -vmax, max: vmax, dimension: 0, seriesIndex: 0, calculable: false, orient: "horizontal",
        left: "center", bottom: 0, itemWidth: 12, itemHeight: 200, text: [`+$${vmax}`, `−$${vmax}`], textStyle: { color: p.muted, fontSize: 11 },
        inRange: { color: [p.cold, p.neutral, p.hot] } },
      series: [{ type: "graph", layout: "none", roam: true, data: gnodes, links: [...phys, ...learned],
        top: 10, bottom: 50, left: 10, right: 10, label: { show: false },
        emphasis: { focus: "adjacency", label: { show: true, color: p.fg, fontSize: 11 } } }],
    };
  }, [view, links, p, layer, isLive, data]);

  const topNodes = nodes.map((n, i) => ({ n, v: view.vals[i] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 6);
  const topLines = lines.map((l, k) => ({ l, b: view.bind[k] ?? 0 })).sort((a, b) => b.b - a.b).slice(0, 5);

  return (
    <div className="grid12">
      <div className="panel span8">
        <h3>Where congestion is expected, hour by hour</h3>
        <p className="caption">Node color is the {layer === "real" && !isLive ? "realized" : "forecast median"} real-time minus day-ahead spread. Lines thicken and turn red with the model's probability that they bind. Diamonds are data-center loads. Drag to pan, scroll to zoom.</p>
        <div className="controls">
          <label className="ctl" htmlFor="map-day">Day
            <HTMLSelect id="map-day" value={day} onChange={(e) => { setDay(e.target.value); if (e.target.value === liveDay) setLayer("pred"); }}
              options={days.map((d) => ({ value: d, label: d === liveDay ? `${d} (live)` : d }))} />
          </label>
          <div className="ctl">Layer
            <SegmentedControl size="small" value={isLive ? "pred" : layer} onValueChange={setLayer}
              options={[{ label: "Forecast", value: "pred" }, { label: "Realized", value: "real", disabled: isLive }]} />
          </div>
          <div className="ctl">Links
            <SegmentedControl size="small" value={links} onValueChange={setLinks}
              options={[{ label: "Physical", value: "physical" }, { label: "Learned", value: "learned" }, { label: "Both", value: "both" }]} />
          </div>
          <div className="ctl">
            <Button size="small" icon={playing ? "pause" : "play"} onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause" : "Play the day"} />
            <span className="mono">HE{String(hour).padStart(2, "0")}</span>
            <div className="slider" style={{ width: 220 }}><Slider min={1} max={24} stepSize={1} labelValues={[1, 6, 12, 18, 24]} value={hour} onChange={(h) => { setPlaying(false); setHour(h); }} /></div>
          </div>
        </div>
        <Chart option={option} height={520} />
      </div>
      <div className="panel span4">
        <h3>HE{hour} at a glance</h3>
        <p className="caption">Largest spreads and the lines most likely to bind.</p>
        <div className="scroll-x">
          <HTMLTable compact striped style={{ width: "100%" }}>
            <thead><tr><th>Node</th><th>Zone</th><th style={{ textAlign: "right" }}>RT − DA</th></tr></thead>
            <tbody>{topNodes.map(({ n, v }) => (
              <tr key={n.id}><td>{n.name}{n.dc ? " ◆" : ""}</td><td>{n.zone}</td><td className={`num ${v >= 0 ? "pos" : "neg"}`} style={{ textAlign: "right" }}>{fmt.signed(v)}</td></tr>))}
            </tbody>
          </HTMLTable>
          <HTMLTable compact striped style={{ width: "100%", marginTop: 12 }}>
            <thead><tr><th>Line</th><th style={{ textAlign: "right" }}>P(bind)</th></tr></thead>
            <tbody>{topLines.map(({ l, b }) => (
              <tr key={l.id}><td>{l.name ?? `${nodes[l.from].name}–${nodes[l.to].name}`}{l.tie ? " (tie)" : ""}</td><td className="num" style={{ textAlign: "right" }}>{fmt.pct(b, 0)}</td></tr>))}
            </tbody>
          </HTMLTable>
        </div>
        <p className="legend-note" style={{ marginTop: 10 }}>Learned links are the strongest entries of the adjacency the GNN learned on its own from congestion co-movement. They often connect nodes that are far apart on the wires but behind the same constraint.</p>
      </div>
    </div>
  );
}
