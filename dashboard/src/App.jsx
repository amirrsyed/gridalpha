import { useEffect, useMemo, useState } from "react";
import { Button, Callout, Classes, Tab, Tabs, Tag, Tooltip } from "@blueprintjs/core";
import { palette } from "./palette.js";
import { fmt, kkey } from "./dist.js";
import Tickets from "./views/Tickets.jsx";
import GridMap from "./views/GridMap.jsx";
import Heatmaps from "./views/Heatmaps.jsx";
import Backtest from "./views/Backtest.jsx";
import ModelHealth from "./views/ModelHealth.jsx";

const store = {
  get: (k, d) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

function systemDark() {
  const attr = document.documentElement.getAttribute("data-theme");
  if (attr) return attr === "dark";
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

export default function App({ data }) {
  const [override, setOverride] = useState(null);
  const [sys, setSys] = useState(systemDark());
  const dark = override ?? sys;
  const [tab, setTab] = useState(() => {
    const h = location.hash.replace("#", "");
    return ["tickets", "map", "heatmaps", "backtest", "model"].includes(h) ? h : store.get("ga-tab", "tickets");
  });

  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const on = () => setSys(systemDark());
    mq?.addEventListener?.("change", on);
    const mo = new MutationObserver(on);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => { mq?.removeEventListener?.("change", on); mo.disconnect(); };
  }, []);
  useEffect(() => {
    document.body.classList.toggle(Classes.DARK, dark);
    if (override !== null) document.documentElement.setAttribute("data-theme", override ? "dark" : "light");
  }, [dark, override]);
  useEffect(() => {
    const on = () => {
      const h = location.hash.replace("#", "");
      if (h) setTab(h);
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  const p = useMemo(() => palette(dark), [dark]);
  const m = data.meta;
  const k = kkey(m.kelly_fraction);
  const gnnBt = data.backtest?.models?.gnn?.metrics?.[k];
  const lgbBt = data.backtest?.models?.lgbm?.metrics?.[k];
  const live = data.live;
  const gOv = data.model?.gnn_overall;
  const lOv = data.model?.lgbm_overall;
  const sim = m.source === "sim";

  const onTab = (t) => {
    setTab(t);
    store.set("ga-tab", t);
    history.replaceState(null, "", `#${t}`);
  };

  return (
    <>
      <header className="topbar">
        <div className="brand">GridAlpha <small>Signal Desk</small></div>
        <Tag minimal intent={sim ? "warning" : "success"}>{sim ? "Simulated ERCOT-style market" : "ERCOT"}</Tag>
        {live && <span className="meta-chip">Operating day {live.operating_day} · bids due D−1 10:00 CT</span>}
        <div className="spacer" />
        <span className="meta-chip">{m.model_version}</span>
        <Tooltip content={dark ? "Switch to light theme" : "Switch to dark theme"}>
          <Button variant="minimal" icon={dark ? "flash" : "moon"} aria-label="Toggle theme" onClick={() => setOverride(!dark)} />
        </Tooltip>
      </header>

      <section className="strip" aria-label="Summary">
        <div className="stat">
          <span className="label">Tickets for {live?.operating_day ?? "next day"}</span>
          <span className="value">{live ? live.tickets.length : "–"}</span>
          <span className="sub">{live ? `${fmt.num(live.tickets.reduce((s, t) => s + t.mw, 0), 1)} MW · κ = ${m.kelly_fraction}` : "no live signal"}</span>
        </div>
        <div className="stat">
          <span className="label">Expected P&amp;L, next day</span>
          <span className="value">{live ? fmt.usd(live.tickets.reduce((s, t) => s + t.exp_pnl_usd, 0)) : "–"}</span>
          <span className="sub">bankroll {live ? fmt.usd(live.capital) : "–"}</span>
        </div>
        <div className="stat">
          <span className="label">Backtest P&amp;L (GNN, κ = {m.kelly_fraction})</span>
          <span className={`value ${gnnBt && gnnBt.pnl >= 0 ? "pos" : "neg"}`}>{gnnBt ? fmt.usd(gnnBt.pnl) : "–"}</span>
          <span className="sub">{gnnBt ? `Sharpe ${fmt.num(gnnBt.sharpe, 2)} · max DD ${fmt.pct(gnnBt.max_drawdown)}` : ""}{lgbBt ? ` · LightGBM ${fmt.usd(lgbBt.pnl)}` : ""}</span>
        </div>
        <div className="stat">
          <span className="label">Forecast skill (pinball, $/MWh)</span>
          <span className="value">{gOv ? fmt.num(gOv.pinball, 3) : "–"}</span>
          <span className="sub">{gOv && lOv ? `${fmt.pct(1 - gOv.pinball / lOv.pinball)} better than LightGBM · 80% band covers ${fmt.pct(gOv.coverage80)}` : ""}</span>
        </div>
      </section>

      {m.diagnostic?.length > 0 && (
        <Callout intent="danger" icon="warning-sign" title="Diagnostic run: not for trading" style={{ marginBottom: 14 }}>
          The autopilot's gates failed at {m.diagnostic.map((d) => `${d.stage} (${d.failed.join(", ").replaceAll("_", " ")})`).join(" and ")}.
          In this run the GNN did not beat LightGBM by the required margin, so its tickets are shown for inspection only and the
          production pipeline withholds them. {sim && "Simulated P&L is also far larger than a real market would allow: the simulator's built-in forecast biases are easy to exploit, so read the figures below as a test of the pipeline, not as evidence of edge."}
        </Callout>
      )}

      <Tabs id="views" selectedTabId={tab} onChange={onTab} renderActiveTabPanelOnly size="large">
        <Tab id="tickets" title="Trade tickets" panel={<Tickets data={data} p={p} dark={dark} />} />
        <Tab id="map" title="Grid map" panel={<GridMap data={data} p={p} />} />
        <Tab id="heatmaps" title="Forecast vs realized" panel={<Heatmaps data={data} p={p} />} />
        <Tab id="backtest" title="Backtest" panel={<Backtest data={data} p={p} />} />
        <Tab id="model" title="Model health" panel={<ModelHealth data={data} p={p} />} />
      </Tabs>

      <p className="footer">
        {sim
          ? "Prices, outages and weather on this page come from GridAlpha's physics-based market simulator (DC optimal power flow on a 48-bus Texas-like grid), used to develop and test the pipeline. Switch the data source to ERCOT to run on real market data. "
          : ""}
        Tickets are signals, not orders: the system never submits bids. Kelly sizes assume the model's predictive distribution is correct, which is why a fraction of Kelly is applied and capped.
      </p>
    </>
  );
}
