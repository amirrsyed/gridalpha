"""Pack out-of-sample predictions and realized prices into one small file for offline analysis.

    python -m gridalpha.tools.export_diag OUT.npz

Reads artifacts/preds/{gnn,lgbm,naive}_assembled.* and artifacts/raw/market.npz.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from ..config import artifacts, load_config
from ..export.store import load_assembled


def main():
    out = sys.argv[1]
    cfg = load_config()
    A = artifacts(cfg)
    raw = np.load(A / "raw" / "market.npz")
    meta = json.loads((A / "raw" / "meta.json").read_text())
    pack = {"da_lmp": raw["da_lmp"].astype(np.float32), "rt_lmp": raw["rt_lmp"].astype(np.float32),
            "da_energy": raw["da_energy"].astype(np.float32), "rt_energy": raw["rt_energy"].astype(np.float32)}
    for m in ["gnn", "lgbm", "naive"]:
        r = load_assembled(A / "preds" / f"{m}_assembled")
        pack[f"{m}_days"] = r["days"]
        pack[f"{m}_fold"] = r["fold"]
        pack[f"{m}_q"] = r["q_cal"].astype(np.float16)
        pack[f"{m}_qraw"] = r["q_raw"].astype(np.float16)
        if m == "gnn":
            pack["gnn_da"] = r["da"].astype(np.float16)
            pack["gnn_bind"] = r["bind"].astype(np.float16)
            pack["gnn_members_med"] = r["members_med"].astype(np.float16)
        for i, p in enumerate(r["pit_val"]):
            pack[f"{m}_pit{i}"] = p.astype(np.float16)
    pack["meta"] = np.array(json.dumps({k: meta[k] for k in ["dates", "zones", "zone_of", "grid", "coverage"]
                                        if k in meta}))
    np.savez_compressed(out, **pack)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
