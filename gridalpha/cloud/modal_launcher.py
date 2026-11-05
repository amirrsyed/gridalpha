"""Client side of the Modal backend: sync artifacts up, fan jobs out, sync results down."""
from __future__ import annotations

import os
from pathlib import Path

VOLUME_NAME = "gridalpha-artifacts"
APP_NAME = "gridalpha"


def _function(name: str):
    import modal
    try:
        return modal.Function.from_name(APP_NAME, name)
    except AttributeError:                        # older modal clients
        return modal.Function.lookup(APP_NAME, name)


def _volume():
    import modal
    return modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _is_file(entry) -> bool:
    from modal.volume import FileEntryType
    return entry.type == FileEntryType.FILE


def push(local: Path, remote_prefix: str = "/artifacts", subdirs=("features", "models", "preds", "raw"),
         extra_files: dict | None = None):
    """Upload local subdirectories (and single files) to the volume, overwriting files that exist."""
    vol = _volume()
    with vol.batch_upload(force=True) as batch:
        for sub in subdirs:
            d = local / sub
            if not d.is_dir():
                continue
            for f in d.rglob("*"):
                if f.is_file():
                    batch.put_file(str(f), f"{remote_prefix}/{f.relative_to(local).as_posix()}")
        for src, dest in (extra_files or {}).items():
            if Path(src).is_file():
                batch.put_file(str(src), dest)


def _download(vol, remote: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with open(tmp, "wb") as fh:
        for chunk in vol.read_file(remote):
            fh.write(chunk)
    os.replace(tmp, dest)


def pull(local: Path, remote_prefix: str = "/artifacts", subdirs=("models", "preds"), extra_files: dict | None = None):
    """Download files under the given volume subdirectories; skip files already present with the same size."""
    vol = _volume()
    prefix = remote_prefix.strip("/")
    for sub in subdirs:
        try:
            entries = vol.listdir(f"{remote_prefix}/{sub}", recursive=True)
        except Exception:  # noqa: BLE001 - nothing stored yet
            continue
        for e in entries:
            if not _is_file(e):
                continue
            rel = Path(e.path.strip("/")).relative_to(prefix)
            dest = local / rel
            if dest.is_file() and getattr(e, "size", None) == dest.stat().st_size:
                continue
            _download(vol, e.path, dest)
    for remote, dest in (extra_files or {}).items():
        try:
            _download(vol, remote, Path(dest))
        except Exception:  # noqa: BLE001 - first run: nothing saved yet
            pass


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
