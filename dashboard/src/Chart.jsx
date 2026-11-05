import { useEffect, useRef } from "react";
import * as echarts from "echarts";

// Thin ECharts wrapper: re-renders when `option` changes, resizes with its box.
export default function Chart({ option, height = 280, onClick }) {
  const el = useRef(null);
  const inst = useRef(null);
  useEffect(() => {
    inst.current = echarts.init(el.current, null, { renderer: "canvas" });
    const ro = new ResizeObserver(() => inst.current && inst.current.resize());
    ro.observe(el.current);
    return () => { ro.disconnect(); inst.current.dispose(); };
  }, []);
  useEffect(() => {
    inst.current.setOption(option, true);
  }, [option]);
  useEffect(() => {
    if (!onClick) return;
    inst.current.on("click", onClick);
    return () => inst.current && inst.current.off("click", onClick);
  }, [onClick]);
  return <div ref={el} style={{ width: "100%", height }} />;
}
