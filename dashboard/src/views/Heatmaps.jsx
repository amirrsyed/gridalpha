import { useMemo, useState } from "react";
import { HTMLSelect } from "@blueprintjs/core";
import Chart from "../Chart.jsx";
import { baseAxis } from "../palette.js";

export default function Heatmaps({ data, p }) {
  const maps = data.maps ?? { dates: [] };
  const nodes = data.grid.nodes;
  const days = [...maps.dates].reverse();
  const [day, setDay] = useState(days[0]);
  const order = useMemo(() => nodes.map((n, i) => i).sort((a, b) => nodes[a].zone.localeCompare(nodes[b].zone) || a - b), [nodes]);
  const i = maps.dates.indexOf(day);
  const hours = Array.from({ length: 24 }, (_, h) => `HE${h + 1}`);

  const heat = (mat, title) => {
    const pts = [];
    order.forEach((n, y) => mat.forEach((row, h) => pts.push([h, y, row[n]])));
    return {
      backgroundColor: "transparent", animation: false,
      title: { text: title, left: 0, top: 0, textStyle: { color: p.fg, fontSize: 13, fontWeight: 600 } },
      tooltip: { backgroundColor: p.panel, borderColor: p.grid, textStyle: { color: p.fg, fontSize: 12 },
        formatter: (x) => `${nodes[order[x.value[1]]].name} · HE${x.value[0] + 1}<br/>RT − DA ${x.value[2] > 0 ? "+" : ""}${x.value[2].toFixed(1)} $/MWh` },
      grid: { left: 64, right: 10, top: 28, bottom: 30 },
      xAxis: { type: "category", data: hours, ...baseAxis(p), axisLabel: { ...baseAxis(p).axisLabel, interval: 3 }, splitLine: { show: false } },
      yAxis: { type: "category", data: order.map((n) => nodes[n].name), ...baseAxis(p), axisLabel: { ...baseAxis(p).axisLabel, fontSize: 9 }, splitLine: { show: false } },
      visualMap: { show: false, min: -25, max: 25, inRange: { color: [p.cold, p.heatNeutral, p.hot] } },
      series: [{ type: "heatmap", data: pts, progressive: 0 }],
    };
  };

  if (i < 0) return <p>No forecast history available.</p>;
  return (
    <div className="grid12">
      <div className="panel span12">
        <h3>Forecast median vs realized spread, by node and hour</h3>
        <p className="caption">Out-of-sample forecasts for the last 30 days of the backtest. Blue: real-time came in below day-ahead; red: above. Nodes are grouped by zone. Color saturates at ±$25/MWh.</p>
        <div className="controls">
          <label className="ctl" htmlFor="hm-day">Day
            <HTMLSelect id="hm-day" value={day} onChange={(e) => setDay(e.target.value)} options={days} />
          </label>
        </div>
        <div className="grid12">
          <div className="span6"><Chart option={heat(maps.pred[i], "Forecast (median)")} height={640} /></div>
          <div className="span6"><Chart option={heat(maps.real[i], "Realized")} height={640} /></div>
        </div>
      </div>
    </div>
  );
}
