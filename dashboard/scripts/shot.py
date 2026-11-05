"""Screenshot each dashboard tab (desktop light + dark, phone dark) and print console errors.

    python3 scripts/shot.py dist-artifact/gridalpha.html OUT_DIR
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
html = "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1,viewport-fit=cover'></head><body>" + src.read_text() + "</body></html>"
tmp = out / "page.html"
tmp.write_text(html)
tabs = ["tickets", "map", "heatmaps", "backtest", "model"]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    for name, vp, scheme in [("desk-dark", (1440, 1000), "dark"), ("desk-light", (1440, 1000), "light"), ("phone", (400, 860), "dark")]:
        ctx = b.new_context(viewport={"width": vp[0], "height": vp[1]}, color_scheme=scheme)
        pg = ctx.new_page()
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        for t in tabs if name != "phone" else ["tickets", "backtest"]:
            pg.goto(f"file://{tmp}#{t}")
            pg.wait_for_timeout(1500)
            pg.screenshot(path=str(out / f"{name}-{t}.png"), full_page=True)
        if name == "desk-dark":
            pg.goto(f"file://{tmp}#tickets")
            pg.wait_for_timeout(1200)
            cell = pg.locator(".bp6-table-cell").nth(3)
            if cell.count():
                cell.click()
                pg.wait_for_timeout(1000)
                pg.screenshot(path=str(out / f"{name}-drawer.png"))
        sw = pg.evaluate("document.documentElement.scrollWidth")
        print(name, "scrollWidth", sw, "errors:", errs[:5])
        ctx.close()
    b.close()
