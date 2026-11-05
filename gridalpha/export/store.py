"""Save / load assembled predictions without pickle (npz arrays + json side-car)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARRAY_KEYS = ["days", "q_raw", "q_cal", "da", "bind", "members_med", "fold", "attrib"]


def save_assembled(res: dict, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: res[k] for k in ARRAY_KEYS if k in res}
    if res.get("adaptive_adj") is not None:
        arrays["adaptive_adj"] = res["adaptive_adj"]
    for i, p in enumerate(res.get("pit_val", [])):
        arrays[f"pit_val_{i}"] = p
    np.savez_compressed(base.with_suffix(".npz"), **arrays)
    side = {k: res.get(k) for k in ["stats", "overall", "curves", "attrib_names"]}
    base.with_suffix(".json").write_text(json.dumps(side, default=float))


def load_assembled(base: Path) -> dict:
    z = np.load(base.with_suffix(".npz"))
    res = {k: z[k] for k in z.files if not k.startswith("pit_val_")}
    res["pit_val"] = [z[f"pit_val_{i}"] for i in range(sum(k.startswith("pit_val_") for k in z.files))]
    res.update(json.loads(base.with_suffix(".json").read_text()))
    return res
