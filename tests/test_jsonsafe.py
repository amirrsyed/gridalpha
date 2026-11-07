import json

import numpy as np

from gridalpha import jsonsafe
from gridalpha.trade.kelly import growth


def test_dumps_is_strict_json():
    obj = {"a": float("-inf"), "b": [np.nan, np.float32(1.5), np.int64(3)], "c": np.array([np.inf, 2.0]), "d": True}
    s = jsonsafe.dumps(obj, separators=(",", ":"))
    assert "Infinity" not in s and "NaN" not in s
    assert json.loads(s) == {"a": None, "b": [None, 1.5, 3], "c": [None, 2.0], "d": True}


def test_growth_curve_can_hit_minus_inf():
    # the source of the dashboard bug: overbetting past ruin gives -inf growth
    X = np.array([-100.0, 50.0, 80.0])
    g = growth(np.array([0.0, 0.5, 2.0]), X, 75.0)
    assert np.isneginf(g[-1])
    assert all(np.isfinite(v) for f, v in zip([0.0, 0.5, 2.0], g) if np.isfinite(v))
