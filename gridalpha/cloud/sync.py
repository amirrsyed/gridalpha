"""Keep artifacts and pipeline state on the Modal Volume between GitHub Actions runs.

    python -m gridalpha.cloud.sync pull     # before a run
    python -m gridalpha.cloud.sync push     # after it

The pipeline state (stage results, gate outcomes, monthly cloud spend) is saved too, so
the budget cap counts across runs and the dashboard shows the last retrain's gates.

The ERCOT download cache is thousands of small files, and fetching them one by one took
about half an hour per run. On push it is also packed into one tar file; on pull that
tar is unpacked first, and only cache files newer than it are fetched individually.
"""
from __future__ import annotations

import sys
import tarfile
import time
from pathlib import Path

from ..config import artifacts, load_config
from .modal_launcher import pull, push

SUBDIRS = ("raw", "features", "models", "preds", "backtest", "signals")
STATE_REMOTE = "/artifacts/state/pipeline.json"
PACK_REMOTE = "/artifacts/packs/raw_cache.tar"


def pack_cache(A: Path) -> Path | None:
    cache = A / "raw" / "cache"
    if not cache.is_dir():
        return None
    out = A / "packs" / "raw_cache.tar"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tar.part")
    with tarfile.open(tmp, "w") as tar:          # parquet is already compressed
        tar.add(cache, arcname="cache")
    tmp.replace(out)
    return out


def unpack_cache(A: Path) -> int:
    pack = A / "packs" / "raw_cache.tar"
    if not pack.is_file():
        return 0
    dest = A / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(pack) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("cache") and ".." not in m.name]
        tar.extractall(dest, members=members, filter="data")
    return len(members)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pull"
    cfg = load_config()
    A = artifacts(cfg)
    state = Path(cfg["_root"]) / cfg["project"]["state_file"]
    t0 = time.time()
    if cmd == "pull":
        pull(A, subdirs=(), extra_files={PACK_REMOTE: A / "packs" / "raw_cache.tar"})
        n = unpack_cache(A)
        pull(A, subdirs=SUBDIRS, extra_files={STATE_REMOTE: state})     # skips files already present
        print(f"sync pull: done in {time.time() - t0:.0f}s ({n} cache files from the pack)")
    elif cmd == "push":
        pack = pack_cache(A)
        extra = {state: STATE_REMOTE}
        if pack:
            extra[pack] = PACK_REMOTE
        push(A, subdirs=SUBDIRS, extra_files=extra)
        print(f"sync push: done in {time.time() - t0:.0f}s")
    else:
        raise SystemExit("usage: sync pull|push")


if __name__ == "__main__":
    main()
