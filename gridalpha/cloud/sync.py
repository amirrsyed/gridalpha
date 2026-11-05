"""Keep artifacts and pipeline state on the Modal Volume between GitHub Actions runs.

    python -m gridalpha.cloud.sync pull     # before a run
    python -m gridalpha.cloud.sync push     # after it

The pipeline state (stage results, gate outcomes, monthly cloud spend) is saved too, so
the budget cap counts across runs and the dashboard shows the last retrain's gates.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..config import artifacts, load_config
from .modal_launcher import pull, push

SUBDIRS = ("raw", "features", "models", "preds", "backtest", "signals")
STATE_REMOTE = "/artifacts/state/pipeline.json"


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pull"
    cfg = load_config()
    A = artifacts(cfg)
    state = Path(cfg["_root"]) / cfg["project"]["state_file"]
    if cmd == "pull":
        pull(A, subdirs=SUBDIRS, extra_files={STATE_REMOTE: state})
    elif cmd == "push":
        push(A, subdirs=SUBDIRS, extra_files={state: STATE_REMOTE})
    else:
        raise SystemExit("usage: sync pull|push")
    print(f"sync {cmd}: done")


if __name__ == "__main__":
    main()
