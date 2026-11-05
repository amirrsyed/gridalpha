"""Walk-forward training: folds x seeds, each a separate resumable job.

Every (fold, seed) member writes its own checkpoint and predictions, so the same code
runs sequentially on a laptop or fans out across cloud GPUs (see ``cloud/modal_app.py``),
and a crashed run resumes where it stopped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..config import artifacts
from ..models.baselines import fit_lgbm, naive_quantiles, predict_lgbm
from .calibrate import apply_conformal, coverage, ensemble, fit_conformal, pit
from .train import attributions, build_model, device_of, fit, physics_for, pinball_np, predict, pretrain


def plan_folds(F: dict, cfg: dict) -> list[dict]:
    wf = cfg["walk_forward"]
    D = F["node"].shape[0]
    first = int(F["first_day"])
    test0 = D - wf["test_days"]
    edges = np.linspace(test0, D, wf["folds"] + 1).astype(int)
    folds = []
    for k in range(wf["folds"]):
        t0, t1 = edges[k], edges[k + 1]
        train_end = t0 - wf["purge_days"]
        val0 = train_end - wf["val_days"]
        folds.append({
            "k": k,
            "train": np.arange(first, val0),
            "val": np.arange(val0, train_end),
            "test": np.arange(t0, t1),
        })
    return folds


def _paths(cfg):
    A = artifacts(cfg)
    m = A / "models"
    p = A / "preds"
    m.mkdir(exist_ok=True, parents=True)
    p.mkdir(exist_ok=True, parents=True)
    return A, m, p


def run_pretrain(F: dict, cfg: dict, fold: dict, log=print) -> Path:
    _, M, _ = _paths(cfg)
    path = M / f"fold{fold['k']}_pretrain.pt"
    if path.exists():
        return path
    torch.manual_seed(cfg["project"]["seed"])
    dev = device_of(cfg)
    model = build_model(F, cfg).to(dev)
    info = pretrain(model, F, fold["train"], cfg, log=log)
    torch.save({"state": model.state_dict(), "info": info}, path)
    return path


def run_member(F: dict, cfg: dict, fold: dict, seed: int, source: str, log=print) -> Path:
    _, M, P = _paths(cfg)
    out = P / f"gnn_fold{fold['k']}_seed{seed}.npz"
    if out.exists():
        return out
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device_of(cfg)
    model = build_model(F, cfg).to(dev)
    pre = M / f"fold{fold['k']}_pretrain.pt"
    if pre.exists():
        # transfer only what pretraining actually learned; heads and encoders keep this
        # seed's fresh initialization so ensemble members stay diverse
        keep = ("node_emb.", "adapt.", "layers.")
        state = {k: v for k, v in torch.load(pre, map_location=dev)["state"].items() if k.startswith(keep)}
        model.load_state_dict(state, strict=False)
    phys = physics_for(F, cfg, source)
    if phys is not None:
        phys = phys.to(dev)
    info = fit(model, F, fold["train"], fold["val"], cfg, seed, phys, log=log)
    torch.save({"state": model.state_dict(), "info": info}, M / f"gnn_fold{fold['k']}_seed{seed}.pt")
    pv = predict(model, F, fold["val"], cfg)
    pt = predict(model, F, fold["test"], cfg)
    extra = {}
    if seed == cfg["train"]["ensemble_seeds"][0]:
        att, names = attributions(model, F, fold["test"], cfg, fold["train"])
        extra = {"attrib": att, "attrib_names": np.array(names)}
        if model.adapt is not None:
            extra["adaptive_adj"] = model.adapt().detach().cpu().numpy()
    np.savez(out, val_q=pv["q"], test_q=pt["q"], test_da=pt["da"], test_bind=pt["bind"], test_cong=pt["cong"],
             val_days=fold["val"], test_days=fold["test"], curve=json.dumps(info["curve"]), **extra)
    return out


def run_lgbm(F: dict, cfg: dict, fold: dict, log=print) -> Path:
    _, _, P = _paths(cfg)
    out = P / f"lgbm_fold{fold['k']}.npz"
    if out.exists():
        return out
    taus = cfg["model"]["quantiles"]
    models = fit_lgbm(F, np.concatenate([fold["train"]]), seed=cfg["project"]["seed"], log=log)
    np.savez(out, val_q=predict_lgbm(models, F, fold["val"], taus), test_q=predict_lgbm(models, F, fold["test"], taus),
             val_days=fold["val"], test_days=fold["test"])
    return out


def run_naive(F: dict, cfg: dict, fold: dict) -> Path:
    _, _, P = _paths(cfg)
    out = P / f"naive_fold{fold['k']}.npz"
    taus = cfg["model"]["quantiles"]
    np.savez(out, val_q=naive_quantiles(F, fold["val"], taus), test_q=naive_quantiles(F, fold["test"], taus),
             val_days=fold["val"], test_days=fold["test"])
    return out


def assemble(F: dict, cfg: dict, model: str, folds: list[dict]) -> dict:
    """Ensemble members, conformal-calibrate on each fold's validation window, stitch the test period."""
    _, _, P = _paths(cfg)
    taus = cfg["model"]["quantiles"]
    seeds = cfg["train"]["ensemble_seeds"]
    y = F["y_spread"]
    parts = {k: [] for k in ["days", "q_raw", "q_cal", "da", "bind", "members_med", "fold", "attrib"]}
    stats, node_corr, curves, adj = [], [], [], None
    attrib_names = None
    for f in folds:
        if model == "gnn":
            mem = [np.load(P / f"gnn_fold{f['k']}_seed{s}.npz") for s in seeds]
            q_val = ensemble([m["val_q"] for m in mem])
            q_test = ensemble([m["test_q"] for m in mem])
            mid = taus.index(0.5)
            parts["members_med"].append(np.stack([m["test_q"][..., mid] for m in mem], -1))
            parts["da"].append(np.mean([m["test_da"] for m in mem], 0))
            parts["bind"].append(np.mean([m["test_bind"] for m in mem], 0))
            parts["attrib"].append(mem[0]["attrib"])
            attrib_names = [str(x) for x in mem[0]["attrib_names"]]
            curves.append([json.loads(str(m["curve"])) for m in mem])
            if "adaptive_adj" in mem[0].files:
                adj = mem[0]["adaptive_adj"]
        else:
            z = np.load(P / f"{model}_fold{f['k']}.npz")
            q_val, q_test = z["val_q"], z["test_q"]
            gnn = np.load(P / f"gnn_fold{f['k']}_seed{seeds[0]}.npz") if (P / f"gnn_fold{f['k']}_seed{seeds[0]}.npz").exists() else None
            parts["da"].append(np.mean([np.load(P / f"gnn_fold{f['k']}_seed{s}.npz")["test_da"] for s in seeds], 0)
                               if gnn is not None else F["y_da"][f["test"]] * np.nan)
        shifts = fit_conformal(q_val, y[f["val"]], taus)
        q_val_cal = apply_conformal(q_val, shifts)
        q_cal = apply_conformal(q_test, shifts)
        node_corr.append(np.nan_to_num(pit(q_val_cal[-30:], y[f["val"]][-30:], taus), nan=0.5))
        parts["days"].append(f["test"])
        parts["q_raw"].append(q_test)
        parts["q_cal"].append(q_cal)
        parts["fold"].append(np.full(len(f["test"]), f["k"]))
        yt = y[f["test"]]
        stats.append({
            "fold": f["k"], "test_start": str(F["dates"][f["test"][0]]), "test_end": str(F["dates"][f["test"][-1]]),
            "train_days": int(len(f["train"])), "pinball_raw": pinball_np(q_test, yt, taus),
            "pinball": pinball_np(q_cal, yt, taus),
            "pinball_median": pinball_np(q_cal[..., taus.index(0.5):taus.index(0.5) + 1], yt, [0.5]),
            "coverage80_raw": coverage(q_test, yt, taus), "coverage80": coverage(q_cal, yt, taus),
            "val_coverage80_raw": coverage(q_val, y[f["val"]], taus),
        })
    res = {k: np.concatenate(v) for k, v in parts.items() if v}
    res["stats"] = stats
    res["pit_val"] = node_corr
    res["curves"] = curves
    res["adaptive_adj"] = adj
    res["attrib_names"] = attrib_names
    yt = y[res["days"]]
    res["overall"] = {
        "pinball": pinball_np(res["q_cal"], yt, taus),
        "pinball_median": pinball_np(res["q_cal"][..., taus.index(0.5):taus.index(0.5) + 1], yt, [0.5]),
        "coverage80": coverage(res["q_cal"], yt, taus),
        "coverage80_raw": coverage(res["q_raw"], yt, taus),
    }
    return res
