"""Modal app: cloud GPUs for pretraining, ensemble members and hyperparameter trials.

Deploy once (CI does this on every retrain):
    modal deploy -m gridalpha.cloud.modal_app
The autopilot then calls ``run_job`` through ``modal_launcher.run_on_modal`` and fans the
fold x seed jobs out in parallel with ``starmap``. Artifacts live on a Modal Volume.

Cost control: per-second billing, a timeout on every call, and the autopilot's monthly
budget check before anything is launched.
"""
from __future__ import annotations

from pathlib import Path

import modal
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CFG = yaml.safe_load((_ROOT / "config" / "default.yaml").read_text())["compute"]["modal"]

APP_NAME = "gridalpha"
VOLUME_NAME = "gridalpha-artifacts"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.4", "numpy", "scipy", "pandas", "pyyaml", "lightgbm", "pyarrow")
    .add_local_dir(str(_ROOT / "config"), "/root/config")
    .add_local_python_source("gridalpha")
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(gpu=_CFG["gpu"], volumes={"/vol": volume}, timeout=int(_CFG["timeout_min"]) * 60, retries=1)
def run_job(kind: str, job: dict, overrides: list[str], profile: str | None) -> list[str]:
    from gridalpha.cloud.worker import run_jobs
    volume.reload()
    out = run_jobs(kind, [job], list(overrides) + ["project.artifacts_dir=/vol/artifacts", "train.device=cuda"],
                   profile)
    volume.commit()
    return out


@app.function(cpu=4, memory=16384, volumes={"/vol": volume}, timeout=3 * 3600)
def ingest_ercot(overrides: list[str]) -> str:
    """Heavy ERCOT history download, run once in the cloud and cached on the volume."""
    from gridalpha.config import load_config, artifacts
    from gridalpha.data.ercot import fetch
    cfg = load_config(overrides=list(overrides) + ["project.artifacts_dir=/vol/artifacts",
                                                   "project.data_source=ercot"])
    p = fetch(cfg, artifacts(cfg) / "raw")
    volume.commit()
    return str(p)
