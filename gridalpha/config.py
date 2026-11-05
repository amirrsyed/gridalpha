"""Config loading with dotted overrides (``--set train.epochs=5``)."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None,
                profile: str | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "default.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if profile:
        with open(ROOT / "config" / "profiles" / f"{profile}.yaml") as fh:
            _merge(cfg, yaml.safe_load(fh) or {})
        cfg["_profile"] = profile
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    cfg["_root"] = str(ROOT)
    return cfg


def load_gates(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("GRIDALPHA_GATES") or ROOT / "config" / "gates.yaml")
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as fh:
        return yaml.safe_load(fh)


def artifacts(cfg: dict) -> Path:
    p = Path(cfg["_root"]) / cfg["project"]["artifacts_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def deep_copy(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
