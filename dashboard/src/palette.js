// Chart colors per theme. Semantic colors (good/bad) stay separate from the model accent.
export function palette(dark) {
  return dark
    ? { fg: "#f6f7f9", muted: "#abb3bf", grid: "#383e47", axis: "#5f6b7c", panel: "#252a31",
        gnn: "#9881f3", lgbm: "#4c90f0", naive: "#8f99a8", dec: "#4c90f0", inc: "#ec9a3c",
        good: "#32a467", bad: "#e76a6e", cold: "#4c90f0", hot: "#e76a6e", neutral: "#738091", heatNeutral: "#2f343c", learned: "#9881f3" }
    : { fg: "#1c2127", muted: "#5f6b7c", grid: "#e5e8eb", axis: "#abb3bf", panel: "#ffffff",
        gnn: "#634dbf", lgbm: "#2d72d2", naive: "#8f99a8", dec: "#2d72d2", inc: "#c87619",
        good: "#238551", bad: "#cd4246", cold: "#2d72d2", hot: "#cd4246", neutral: "#8f99a8", heatNeutral: "#edeff2", learned: "#634dbf" };
}

export function baseAxis(p) {
  return {
    axisLine: { lineStyle: { color: p.axis } },
    axisTick: { lineStyle: { color: p.axis } },
    axisLabel: { color: p.muted, fontFamily: "IBM Plex Mono, ui-monospace, monospace", fontSize: 11 },
    splitLine: { lineStyle: { color: p.grid } },
    nameTextStyle: { color: p.muted, fontSize: 11 },
  };
}

export function baseOption(p) {
  return {
    backgroundColor: "transparent",
    textStyle: { color: p.fg, fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif" },
    tooltip: { trigger: "axis", backgroundColor: p.panel, borderColor: p.grid, textStyle: { color: p.fg, fontSize: 12 } },
    legend: { textStyle: { color: p.muted }, top: 0, itemWidth: 14, itemHeight: 8 },
    grid: { left: 56, right: 18, top: 30, bottom: 34 },
    animation: false,
  };
}
