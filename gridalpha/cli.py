"""Command line: ``python -m gridalpha.cli <command>``.

  run        run the autopilot (resumes; --from STAGE to redo from a stage, --only STAGE)
  status     print the pipeline state
  daily      refresh data and produce tomorrow's tickets with the current models
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .autopilot import STAGES, Autopilot
from .config import load_config


def _gh_output(key: str, value: str):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as fh:
            fh.write(f"{key}={value}\n")


def models_ready(ap) -> tuple[bool, str]:
    """Daily tickets need a retrain whose train, calibrate and backtest gates all passed."""
    st = ap.state["stages"]
    for s in ["train", "calibrate", "backtest"]:
        if st.get(s, {}).get("status") != "passed":
            return False, f"the last retrain's {s} stage is '{st.get(s, {}).get('status', 'pending')}'"
    k = ap.cfg["walk_forward"]["folds"] - 1
    missing = [s for s in ap.cfg["train"]["ensemble_seeds"]
               if not (ap.A / "models" / f"gnn_fold{k}_seed{s}.pt").exists()]
    if missing:
        return False, f"model checkpoints for seeds {missing} are missing"
    return True, ""


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
        ready, why = models_ready(ap)
        _gh_output("ready", "true" if ready else "false")
        if not ready:
            print(f"::notice::No tickets today: {why}. Run the retrain workflow first.")
            return 0
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
