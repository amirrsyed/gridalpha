"""Alternative backend: a short-lived DigitalOcean GPU Droplet.

Creates a GPU Droplet, copies the repo and features over SSH, runs the jobs, copies the
results back and always destroys the Droplet (also on failure), so billing stops.

Needs: DIGITALOCEAN_TOKEN, DO_SSH_KEY_ID (key registered in DigitalOcean) and the private
key at DO_SSH_KEY_PATH (default ~/.ssh/id_ed25519). Modal is the primary backend; use this
if you have a paid DigitalOcean account (Student Pack credits expired Aug 1, 2026).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

API = "https://api.digitalocean.com/v2"
IMAGE = "gpu-h100x1-base"   # DigitalOcean's AI/ML-ready GPU image slug; override with DO_IMAGE


def _req(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {os.environ['DIGITALOCEAN_TOKEN']}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
        return json.loads(raw) if raw else {}


def _ssh(ip: str, cmd: str, key: str):
    subprocess.run(["ssh", "-i", key, "-o", "StrictHostKeyChecking=accept-new", f"root@{ip}", cmd], check=True)


def _scp(src: str, dst: str, key: str):
    subprocess.run(["scp", "-r", "-i", key, "-o", "StrictHostKeyChecking=accept-new", src, dst], check=True)


def run_on_droplet(ap, kind: str, jobs: list[dict]):
    dc = ap.cfg["compute"]["digitalocean"]
    key = os.path.expanduser(os.environ.get("DO_SSH_KEY_PATH", "~/.ssh/id_ed25519"))
    body = {"name": f"gridalpha-{kind}-{int(time.time())}", "region": dc["region"], "size": dc["size"],
            "image": os.environ.get("DO_IMAGE", IMAGE), "ssh_keys": [os.environ["DO_SSH_KEY_ID"]],
            "tags": ["gridalpha", "ephemeral"]}
    drop = _req("POST", "/droplets", body)["droplet"]
    did = drop["id"]
    ap.log(f"  digitalocean: created droplet {did}")
    try:
        ip = None
        for _ in range(90):
            d = _req("GET", f"/droplets/{did}")["droplet"]
            nets = [n["ip_address"] for n in d["networks"]["v4"] if n["type"] == "public"]
            if d["status"] == "active" and nets:
                ip = nets[0]
                break
            time.sleep(10)
        if not ip:
            raise RuntimeError("droplet never became active")
        time.sleep(30)
        root = Path(ap.cfg["_root"])
        _ssh(ip, "mkdir -p /opt/gridalpha/artifacts", key)
        for sub in ["gridalpha", "config", "requirements.txt"]:
            _scp(str(root / sub), f"root@{ip}:/opt/gridalpha/", key)
        for sub in ["features", "models", "preds"]:
            if (ap.A / sub).exists():
                _scp(str(ap.A / sub), f"root@{ip}:/opt/gridalpha/artifacts/", key)
        prof = f"--profile {ap.cfg['_profile']}" if ap.cfg.get("_profile") else ""
        _ssh(ip, "cd /opt/gridalpha && pip install -q -r requirements.txt && "
                 f"python -m gridalpha.cloud.worker --kind {kind} --jobs {shlex.quote(json.dumps(jobs))} {prof} "
                 "--set train.device=auto", key)
        for sub in ["models", "preds"]:
            _scp(f"root@{ip}:/opt/gridalpha/artifacts/{sub}", str(ap.A), key)
    finally:
        _req("DELETE", f"/droplets/{did}")
        ap.log(f"  digitalocean: destroyed droplet {did}")
