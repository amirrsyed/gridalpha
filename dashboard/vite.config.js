import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// `npm run build`            -> static site that fetches public/signals.json (GitHub Pages)
// `npm run build:artifact`   -> one self-contained HTML file (data embedded by scripts/embed.mjs)
export default defineConfig(({ mode }) => ({
  plugins: [react(), ...(mode === "artifact" ? [viteSingleFile()] : [])],
  base: "./",
  build: { outDir: mode === "artifact" ? "dist-artifact" : "dist", chunkSizeWarningLimit: 4000 },
}));
