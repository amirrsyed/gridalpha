import React from "react";
import { createRoot } from "react-dom/client";
import { Icons } from "@blueprintjs/icons";
import "@blueprintjs/core/lib/css/blueprint.css";
import "@blueprintjs/table/lib/css/table.css";
import "./styles.css";
import App from "./App.jsx";

async function loadData() {
  if (window.__GRIDALPHA__) return window.__GRIDALPHA__;
  const r = await fetch("./signals.json", { cache: "no-store" });
  if (!r.ok) throw new Error(`signals.json: HTTP ${r.status}`);
  return r.json();
}

Icons.setLoaderOptions({ loader: "all" });
const root = createRoot(document.getElementById("root"));
Promise.all([loadData(), Icons.loadAll().catch(() => null)])
  .then(([data]) => root.render(<App data={data} />))
  .catch((e) => {
    root.render(<p style={{ padding: 24 }}>Could not load the signal data ({String(e.message)}). Run the publish stage to create signals.json.</p>);
  });
