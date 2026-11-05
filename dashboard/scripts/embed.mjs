// Embed signals.json into the single-file artifact build: dist-artifact/index.html -> dist-artifact/gridalpha.html
import { readFileSync, writeFileSync } from "node:fs";

const [, , dataPath = "public/signals.json", htmlPath = "dist-artifact/index.html", out = "dist-artifact/gridalpha.html"] = process.argv;
const data = readFileSync(dataPath, "utf8").replace(/</g, "\\u003c");
let html = readFileSync(htmlPath, "utf8");
html = html.replace(/<!doctype html>/i, "").replace(/<\/?html[^>]*>/gi, "").replace(/<\/?head>/gi, "").replace(/<\/?body>/gi, "");
html = html.replace("<div id=\"root\"></div>", `<div id="root"></div>\n<script>window.__GRIDALPHA__ = ${data};</script>`);
// the title and the viewport-sides meta must sit in the first 8 KB, ahead of the inlined CSS
const head = [];
html = html.replace(/<title>[\s\S]*?<\/title>/i, (m) => { head.push(m); return ""; });
html = html.replace(/<meta name="claude:viewport-fit-sides"[^>]*>/i, (m) => { head.push(m); return ""; });
html = head.join("\n") + "\n" + html.trim() + "\n";
writeFileSync(out, html);
console.log(`wrote ${out} (${(Buffer.byteLength(html) / 1e6).toFixed(2)} MB)`);
