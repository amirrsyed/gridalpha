"""Client side of the Modal backend: sync artifacts up, fan jobs out, sync results down."""
from __future__ import annotations

from pathlib import Path

VOLUME_NAME = "gridalpha-artifacts"
APP_NAME = "gridalpha"


def _function(name: str):
    import modal
    try:
        return modal.Function.from_name(APP_NAME, name)
    except AttributeError:                        # older modal clients
        return modal.Function.lookup(APP_NAME, name)


def push(local: Path, remote_prefix: str = "/artifacts", subdirs=("features", "models", "preds", "raw")):
    import modal
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        for sub in subdirs:
            d = local / sub
            if d.exists():
                batch.put_directory(str(d), f"{remote_prefix}/{sub}")


def pull(local: Path, remote_prefix: str = "/artifacts", subdirs=("models", "preds")):
    import modal
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    for sub in subdirs:
        try:
            entries = vol.listdir(f"{remote_prefix}/{sub}", recursive=True)
        except Exception:  # noqa: BLE001
            continue
        for e in entries:
            if getattr(e, "type", None) and str(e.type).endswith("DIRECTORY"):
                continue
            dest = local / Path(e.path).relative_to(remote_prefix.lstrip("/"))
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in vol.read_file(e.path):
                    fh.write(chunk)


def run_on_modal(ap, kind: str, jobs: list[dict]):
    """Upload what the jobs need, run them in parallel on GPUs, download checkpoints + predictions."""
    push(ap.A, subdirs=("features", "models", "preds"))
    fn = _function("run_job")
    profile = ap.cfg.get("_profile")
    overrides = [f"project.data_source={ap.cfg['project']['data_source']}"]
    ap.log(f"  modal: launching {len(jobs)} {kind} job(s) in parallel")
    for res in fn.starmap([(kind, j, overrides, profile) for j in jobs], return_exceptions=False):
        ap.log(f"  modal: done {res}")
    pull(ap.A)
