"""Keep artifacts (models, predictions, raw cache, pipeline state) on the Modal Volume between runs.

    python -m gridalpha.cloud.sync pull     # before a GitHub Actions run
    python -m gridalpha.cloud.sync push     # after it
"""
from __future__ import annotations

import sys

from ..config import artifacts, load_config
from .modal_launcher import pull, push

SUBDIRS = ("raw", "features", "models", "preds", "backtest", "signals")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pull"
    cfg = load_config()
    A = artifacts(cfg)
    if cmd == "pull":
        pull(A, subdirs=SUBDIRS)
    elif cmd == "push":
        push(A, subdirs=SUBDIRS)
    else:
        raise SystemExit("usage: sync pull|push")


if __name__ == "__main__":
    main()
