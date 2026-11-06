"""Sync round trip against an in-memory stand-in for the Modal volume."""
import io
from pathlib import Path
from types import SimpleNamespace

from gridalpha.cloud import modal_launcher, sync

FILE, DIR = 1, 2


class FakeVolume:
    def __init__(self):
        self.files = {}
        self.reads = 0

    def batch_upload(self, force=False):
        vol = self

        class B:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def put_file(self, src, dest):
                vol.files[dest] = Path(src).read_bytes()
        return B()

    def listdir(self, path, recursive=False):
        hits = [k for k in self.files if k.startswith(path.rstrip("/") + "/")]
        if not hits:
            raise FileNotFoundError(path)
        return [SimpleNamespace(path=k.lstrip("/"), type=FILE, size=len(self.files[k])) for k in hits]

    def read_file(self, path):
        self.reads += 1
        key = path if path.startswith("/") else "/" + path
        if key not in self.files:
            raise FileNotFoundError(path)
        yield self.files[key]


def test_push_then_pull_uses_the_pack(tmp_path, monkeypatch):
    vol = FakeVolume()
    monkeypatch.setattr(modal_launcher, "_volume", lambda: vol)
    monkeypatch.setattr(modal_launcher, "_is_file", lambda e: e.type == FILE)
    src = tmp_path / "src"
    for i in range(60):
        f = src / "raw" / "cache" / "v2" / "da_spp" / f"2025-01-{i:02d}.parquet"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * (i + 1))
    (src / "models").mkdir()
    (src / "models" / "m.pt").write_bytes(b"model")
    state = tmp_path / "state.json"
    state.write_text("{}")

    pack = sync.pack_cache(src)
    modal_launcher.push(src, subdirs=sync.SUBDIRS, extra_files={state: sync.STATE_REMOTE, pack: sync.PACK_REMOTE})

    dst = tmp_path / "dst"
    modal_launcher.pull(dst, subdirs=(), extra_files={sync.PACK_REMOTE: dst / "packs" / "raw_cache.tar"})
    assert sync.unpack_cache(dst) > 60
    vol.reads = 0
    modal_launcher.pull(dst, subdirs=sync.SUBDIRS, extra_files={sync.STATE_REMOTE: tmp_path / "state2.json"})
    assert vol.reads == 2                      # only the model and the state file; the cache came from the pack
    assert (dst / "raw" / "cache" / "v2" / "da_spp" / "2025-01-59.parquet").read_bytes() == b"x" * 60
    assert (dst / "models" / "m.pt").read_bytes() == b"model"
