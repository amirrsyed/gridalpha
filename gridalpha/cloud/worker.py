"""Run a list of training jobs against an artifacts directory (used by every backend).

    python -m gridalpha.cloud.worker --kind member --jobs '[{"fold":0,"seed":11}]' [--profile local]
"""
from __future__ import annotations

import argparse
import json

from ..config import load_config


def run_jobs(kind: str, jobs: list[dict], overrides: list[str] | None = None, profile: str | None = None,
             log=print) -> list[str]:
    from ..data.build import load_features
    from ..config import artifacts
    from ..train import walk_forward as wf
    cfg = load_config(overrides=overrides or [], profile=profile)
    F = load_features(artifacts(cfg) / "features" / "features.npz")
    folds = {f["k"]: f for f in wf.plan_folds(F, cfg)}
    src = cfg["project"]["data_source"]
    done = []
    for j in jobs:
        f = folds[j["fold"]]
        if kind == "pretrain":
            p = wf.run_pretrain(F, cfg, f, log=log)
        elif kind == "member":
            p = wf.run_member(F, cfg, f, j["seed"], src, log=log)
        elif kind == "lgbm":
            p = wf.run_lgbm(F, cfg, f, log=log)
        else:
            raise ValueError(kind)
        done.append(str(p))
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--profile")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    print(json.dumps(run_jobs(a.kind, json.loads(a.jobs), a.set, a.profile)))
