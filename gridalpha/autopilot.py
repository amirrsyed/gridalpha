"""Autopilot: a resumable, gated state machine that runs the whole project.

    ingest -> validate* -> features -> pretrain -> baselines* -> train* -> calibrate* -> backtest* -> signals -> publish
    (* = stage with a pass/fail gate from config/gates.yaml)

Each stage is idempotent and records metrics in ``state/pipeline.json``. A failure or a
missed gate stops the run and opens a GitHub issue labeled ``claude`` (when running in
GitHub Actions), which the Claude Code action picks up to propose a fix. Compute-heavy
stages go through the launcher, which runs them locally, on Modal or on a DigitalOcean
GPU Droplet, and refuses jobs that would exceed the monthly budget.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from .config import artifacts, load_gates

STAGES = ["ingest", "validate", "features", "pretrain", "baselines", "train", "calibrate", "backtest", "signals",
          "publish"]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class GateFailed(Exception):
    pass


class Autopilot:
    def __init__(self, cfg: dict, log=print):
        self.cfg = cfg
        self.gates = load_gates()
        self.log = log
        self.root = Path(cfg["_root"])
        self.state_path = self.root / cfg["project"]["state_file"]
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()
        self.A = artifacts(cfg)
        self._F = None

    # -- state ------------------------------------------------------------------
    def _load(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"created": _now(), "stages": {s: {"status": "pending"} for s in STAGES}, "spend_usd": {},
                "runs": []}

    def save(self):
        self.state["updated"] = _now()
        self.state["profile"] = self.cfg.get("_profile", "default")
        self.state["data_source"] = self.cfg["project"]["data_source"]
        self.state_path.write_text(json.dumps(self.state, indent=2, default=float))

    def reset(self, from_stage: str):
        hit = False
        for s in STAGES:
            hit = hit or s == from_stage
            if hit:
                self.state["stages"][s] = {"status": "pending"}
        self.save()

    # -- shared data --------------------------------------------------------------
    @property
    def F(self) -> dict:
        if self._F is None:
            from .data.build import load_features
            self._F = load_features(self.A / "features" / "features.npz")
        return self._F

    def folds(self):
        from .train.walk_forward import plan_folds
        return plan_folds(self.F, self.cfg)

    def meta(self) -> dict:
        return json.loads((self.A / "raw" / "meta.json").read_text())

    # -- runner -------------------------------------------------------------------------
    def run(self, until: str | None = None, only: str | None = None, diagnostic: bool = False) -> bool:
        """diagnostic=True keeps going past failed gates so results can be inspected. The gate stays
        failed in the state, an alert is still raised, and every output is marked not-for-trading."""
        run = {"started": _now(), "stages": [], "diagnostic": diagnostic}
        self.state["runs"].append(run)
        for name in STAGES:
            st = self.state["stages"][name]
            if only and name != only:
                continue
            if not only and (st.get("status") == "passed" or (diagnostic and st.get("diagnostic_override"))):
                self.log(f"[skip] {name} (passed {st.get('ended')})")
                if until == name:
                    break
                continue
            self.log(f"[run ] {name}")
            st.update({"status": "running", "started": _now(), "error": None})
            self.save()
            t0 = time.time()
            try:
                metrics = getattr(self, f"stage_{name}")() or {}
                st["metrics"] = metrics
                gate = self.check_gate(name, metrics)
                st["gate"] = gate
                if gate and not gate["passed"]:
                    raise GateFailed(f"gate failed: {gate['failed']}")
                st["status"] = "passed"
            except GateFailed as e:
                st["status"] = "gate_failed"
                st["error"] = str(e)
            except Exception as e:  # noqa: BLE001
                st["status"] = "failed"
                st["error"] = f"{type(e).__name__}: {e}"
                st["traceback"] = traceback.format_exc()[-4000:]
            st["ended"] = _now()
            st["secs"] = round(time.time() - t0, 1)
            run["stages"].append({"stage": name, "status": st["status"], "secs": st["secs"]})
            self.save()
            self.log(f"[{st['status']}] {name} in {st['secs']}s" + (f" -> {st['error']}" if st.get("error") else ""))
            if st["status"] == "gate_failed" and diagnostic:
                st["diagnostic_override"] = True
                self.notify_failure(name, st)
                self.save()
                continue
            if st["status"] != "passed":
                self.notify_failure(name, st)
                run["ended"] = _now()
                self.save()
                return False
            if until == name or only:
                break
        run["ended"] = _now()
        self.save()
        return True

    # -- gates -----------------------------------------------------------------------------
    def check_gate(self, name: str, m: dict) -> dict | None:
        g = self.gates.get(name)
        if not g:
            return None
        checks = {}
        if name == "validate":
            checks["missing_frac"] = m["missing_frac"] <= g["max_missing_frac"]
            if g.get("leak_test_must_pass"):
                checks["leak_test"] = bool(m["leak_test"]["passed"])
        elif name == "baselines":
            if g.get("lgbm_beats_naive_pinball"):
                checks["lgbm_beats_naive_pinball"] = m["lgbm"]["pinball"] < m["naive"]["pinball"]
        elif name == "train":
            if g.get("gnn_beats_lgbm_pinball"):
                imp = 1 - m["gnn"]["pinball"] / m["lgbm"]["pinball"]
                checks["gnn_beats_lgbm_pinball"] = imp >= g["min_pinball_improvement"]
        elif name == "calibrate":
            checks["coverage80"] = abs(m["coverage80"] - 0.8) <= g["coverage_tolerance"]
        elif name == "backtest":
            k = str(self.cfg["trade"]["kelly_fraction"])
            gm = m["gnn"][k]
            checks["sharpe"] = gm["sharpe"] >= g["min_sharpe"]
            checks["max_drawdown"] = gm["max_drawdown"] <= g["max_drawdown"]
            if g.get("gnn_pnl_beats_lgbm"):
                checks["gnn_pnl_beats_lgbm"] = gm["pnl"] > m["lgbm"][k]["pnl"]
        failed = [k for k, v in checks.items() if not v]
        return {"passed": not failed, "checks": {k: bool(v) for k, v in checks.items()}, "failed": failed}

    # -- notifications ------------------------------------------------------------------------
    def notify_failure(self, name: str, st: dict):
        body = (f"Autopilot stage **{name}** ended with status `{st['status']}`.\n\n"
                f"Error: `{st.get('error')}`\n\nGate: ```json\n{json.dumps(st.get('gate'), indent=2)}\n```\n\n"
                f"Metrics: ```json\n{json.dumps(st.get('metrics'), indent=2, default=float)[:6000]}\n```\n\n"
                f"Traceback tail:\n```\n{(st.get('traceback') or '')[-2500:]}\n```\n\n"
                "@claude Diagnose this failure and open a PR that fixes it. Do not edit config/gates.yaml "
                "or trading limits. Run `python -m gridalpha.cli run --profile ci` to verify before opening the PR.")
        alert = self.root / "state" / "ALERT.md"
        alert.write_text(f"# Autopilot alert ({_now()})\n\n{body}\n")
        if os.environ.get("GITHUB_ACTIONS") == "true" and shutil.which("gh"):
            title = f"[autopilot] {name}: {st['status']}"
            try:
                subprocess.run(["gh", "issue", "create", "--title", title, "--body", body, "--label", "autopilot",
                                "--label", "claude"], check=False, timeout=60)
            except Exception:  # noqa: BLE001
                pass

    # -- launcher -------------------------------------------------------------------------------
    def launch(self, kind: str, jobs: list[dict], est_minutes: float):
        """Run jobs (dicts with fold/seed) on the configured backend, with a budget check."""
        backend = self.cfg["compute"]["backend"]
        if backend == "local":
            for j in jobs:
                self._local_job(kind, j)
            return
        rate = self.cfg["compute"][backend]["est_usd_per_hour"]
        est = rate * est_minutes / 60.0 * len(jobs)
        month = dt.date.today().strftime("%Y-%m")
        spent = self.state["spend_usd"].get(month, 0.0)
        cap = self.cfg["compute"]["monthly_budget_usd"]
        if spent + est > cap:
            raise RuntimeError(f"budget: launching {len(jobs)} {kind} jobs (~${est:.2f}) would exceed ${cap}/month "
                               f"(spent ${spent:.2f})")
        if backend == "modal":
            from .cloud.modal_launcher import run_on_modal
            run_on_modal(self, kind, jobs)
        elif backend == "digitalocean":
            from .cloud.do_launcher import run_on_droplet
            run_on_droplet(self, kind, jobs)
        else:
            raise ValueError(backend)
        self.state["spend_usd"][month] = spent + est
        self.save()

    def _local_job(self, kind: str, j: dict):
        from .train import walk_forward as wf
        folds = {f["k"]: f for f in self.folds()}
        src = self.cfg["project"]["data_source"]
        if kind == "pretrain":
            wf.run_pretrain(self.F, self.cfg, folds[j["fold"]], log=self.log)
        elif kind == "member":
            wf.run_member(self.F, self.cfg, folds[j["fold"]], j["seed"], src, log=self.log)
        elif kind == "lgbm":
            wf.run_lgbm(self.F, self.cfg, folds[j["fold"]], log=self.log)

    # -- stages ------------------------------------------------------------------------------------
    def stage_ingest(self) -> dict:
        raw = self.A / "raw"
        if self.cfg["project"]["data_source"] == "sim":
            if not (raw / "market.npz").exists():
                from .sim.market import simulate
                simulate(self.cfg, raw, log=self.log)
        else:
            from .data.ercot import fetch
            fetch(self.cfg, raw, log=self.log)
        meta = self.meta()
        return {"days": len(meta["dates"]), "first": meta["dates"][0], "last": meta["dates"][-1],
                "nodes": len(meta["grid"]["nodes"]), "lines": len(meta["grid"]["lines"]), "source": meta["source"]}

    def stage_validate(self) -> dict:
        from .data.build import leak_test, load_raw
        raw, meta = load_raw(self.A / "raw")
        da = raw["da_lmp"]
        rt = raw["rt_lmp"][:-1]          # the last day is "tomorrow": RT not yet known for real data
        missing = float((~np.isfinite(da)).mean() * 0.5 + (~np.isfinite(rt)).mean() * 0.5)
        lt = leak_test(raw, meta, self.cfg, n_checks=3)
        return {"missing_frac": missing, "leak_test": lt, "da_mean": float(np.nanmean(da)),
                "rt_mean": float(np.nanmean(rt)), "spread_p99": float(np.nanpercentile(rt - da[:-1], 99))}

    def stage_features(self) -> dict:
        from .data.build import build_features, load_raw, save_features
        raw, meta = load_raw(self.A / "raw")
        F = build_features(raw, meta, self.cfg)
        save_features(F, self.A / "features")
        self._F = None
        return {"node_features": int(F["node"].shape[-1]), "edge_features": int(F["edge"].shape[-1]),
                "days": int(F["node"].shape[0]), "first_trainable_day": str(F["dates"][int(F["first_day"])]),
                "folds": [{"k": f["k"], "train": int(len(f["train"])), "val": int(len(f["val"])),
                           "test": [str(self.F["dates"][f["test"][0]]), str(self.F["dates"][f["test"][-1]])]}
                          for f in self.folds()]}

    def stage_pretrain(self) -> dict:
        jobs = [{"fold": f["k"]} for f in self.folds()]
        self.launch("pretrain", jobs, est_minutes=10)
        out = {}
        for f in self.folds():
            import torch
            info = torch.load(self.A / "models" / f"fold{f['k']}_pretrain.pt", map_location="cpu")["info"]
            out[f"fold{f['k']}_final_masked_mse"] = info["pretrain_loss"][-1]
            out[f"fold{f['k']}_first_masked_mse"] = info["pretrain_loss"][0]
        return out

    def stage_baselines(self) -> dict:
        from .train.walk_forward import assemble, run_naive
        for f in self.folds():
            run_naive(self.F, self.cfg, f)
        self.launch("lgbm", [{"fold": f["k"]} for f in self.folds()], est_minutes=8)
        folds = self.folds()
        return {"naive": assemble(self.F, self.cfg, "naive", folds)["overall"],
                "lgbm": assemble(self.F, self.cfg, "lgbm", folds)["overall"]}

    def stage_train(self) -> dict:
        jobs = [{"fold": f["k"], "seed": s} for f in self.folds() for s in self.cfg["train"]["ensemble_seeds"]]
        self.launch("member", jobs, est_minutes=25)
        from .train.walk_forward import assemble
        folds = self.folds()
        g = assemble(self.F, self.cfg, "gnn", folds)
        l = assemble(self.F, self.cfg, "lgbm", folds)
        return {"gnn": g["overall"], "lgbm": l["overall"],
                "pinball_improvement_vs_lgbm": 1 - g["overall"]["pinball"] / l["overall"]["pinball"]}

    def stage_calibrate(self) -> dict:
        from .train.walk_forward import assemble
        from .export.store import save_assembled
        folds = self.folds()
        out = {}
        for m in ["gnn", "lgbm", "naive"]:
            res = assemble(self.F, self.cfg, m, folds)
            save_assembled(res, self.A / "preds" / f"{m}_assembled")
            out[m] = res["overall"] | {"folds": res["stats"]}
        return out["gnn"] | {"lgbm": out["lgbm"], "naive": out["naive"]}

    def stage_backtest(self) -> dict:
        from .trade.backtest import run_backtests
        return run_backtests(self)

    def stage_signals(self) -> dict:
        from .trade.backtest import live_signals
        return live_signals(self)

    def stage_publish(self) -> dict:
        from .export.dashboard_data import export_dashboard
        return export_dashboard(self)
