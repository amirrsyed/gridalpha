"""Command line: ``python -m gridalpha.cli <command>``.

  run        run the autopilot (resumes; --from STAGE to redo from a stage, --only STAGE)
  status     print the pipeline state
  daily      refresh data and produce tomorrow's tickets with the current models
"""
from __future__ import annotations

import argparse
import json
import sys

from .autopilot import STAGES, Autopilot
from .config import load_config


def main(argv=None):
    p = argparse.ArgumentParser(prog="gridalpha")
    p.add_argument("command", choices=["run", "status", "daily"])
    p.add_argument("--profile", default=None, help="config/profiles/<name>.yaml")
    p.add_argument("--set", action="append", default=[], help="override, e.g. train.epochs=5")
    p.add_argument("--from", dest="from_stage", choices=STAGES)
    p.add_argument("--until", choices=STAGES)
    p.add_argument("--only", choices=STAGES)
    p.add_argument("--diagnostic", action="store_true",
                   help="continue past failed gates for inspection; outputs are marked not-for-trading")
    a = p.parse_args(argv)
    cfg = load_config(overrides=a.set, profile=a.profile)
    ap = Autopilot(cfg, log=lambda *x: print(*x, flush=True))
    if a.command == "status":
        for s in STAGES:
            st = ap.state["stages"][s]
            print(f"{s:10s} {st.get('status', 'pending'):12s} {st.get('ended', '')}  {st.get('error') or ''}")
        return 0
    if a.command == "daily":
        for s in ["ingest", "validate", "features", "signals", "publish"]:
            ap.state["stages"][s] = {"status": "pending"}
        ok = all(ap.run(only=s) for s in ["ingest", "validate", "features", "signals", "publish"])
        return 0 if ok else 1
    if a.from_stage:
        ap.reset(a.from_stage)
    ok = ap.run(until=a.until, only=a.only, diagnostic=a.diagnostic)
    print(json.dumps({s: ap.state["stages"][s].get("status") for s in STAGES}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
