"""Strict JSON for files a browser reads: NaN and +-Infinity become null.

Python's json writes NaN/Infinity/-Infinity tokens that JSON.parse rejects, which blanks the dashboard.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return x


def dumps(obj: Any, **kw) -> str:
    kw.setdefault("default", float)
    return json.dumps(clean(obj), allow_nan=False, **kw)
