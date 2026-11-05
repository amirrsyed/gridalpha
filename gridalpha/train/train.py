"""Pretraining, supervised fine-tuning, prediction and attribution for the ST-GNN."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from ..data.build import DRIVER_GROUPS, EDGE_FEATURES, GLOBAL_FEATURES, NODE_FEATURES
from ..models.gnn import STGNN
from ..models.losses import PhysicsLoss, masked_bce, masked_huber, pinball


def device_of(cfg: dict) -> torch.device:
    want = cfg["train"].get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def build_model(F: dict, cfg: dict) -> STGNN:
    m = cfg["model"]
    rel = list(m["relations"])
    return STGNN(
        n_nodes=F["node"].shape[2], lines=torch.tensor(F["lines"]), fn=F["node"].shape[-1],
        fg=F["glob"].shape[-1], fe=F["edge"].shape[-1], fh=F["hist"].shape[-1], hidden=m["hidden"],
        layers=m["layers"], node_emb=m["node_emb"], quantiles=m["quantiles"], adaptive_topk=m["adaptive_topk"],
        relations=rel, dropout=m["dropout"],
    )


def batch(F: dict, idx: np.ndarray, dev: torch.device, labels: bool = True) -> dict:
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a[idx]), dtype=torch.float32, device=dev)
    out = {k: t(F[k]) for k in ["node", "glob", "hist", "edge"]}
    if labels:
        for k in ["y_spread", "y_da", "y_rtcong", "y_bind", "y_inj"]:
            if k in F:
                out[k] = t(F[k])
    return out


def to_target(y: torch.Tensor, cfg: dict) -> torch.Tensor:
    s = cfg["loss"]["target_scale"]
    return torch.asinh(y / s) if cfg["loss"].get("target_transform") == "asinh" else y / s


def to_usd(q: torch.Tensor, cfg: dict) -> torch.Tensor:
    s = cfg["loss"]["target_scale"]
    return s * torch.sinh(q) if cfg["loss"].get("target_transform") == "asinh" else s * q


def _mask(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = torch.isfinite(y).float()
    return torch.nan_to_num(y), m


def pretrain(model: STGNN, F: dict, days: np.ndarray, cfg: dict, log=print) -> dict:
    """Masked-node reconstruction of RT congestion on historical hours (no labels needed)."""
    pc = cfg["pretrain"]
    dev = next(model.parameters()).device
    rng = np.random.default_rng(cfg["project"]["seed"])
    cong = np.arcsinh(np.nan_to_num(F["y_rtcong"]) / 10.0).astype(np.float32)
    hours = np.array([(d, h) for d in days for h in range(24)])
    opt = torch.optim.AdamW(model.parameters(), lr=pc["lr"], weight_decay=1e-4)
    hist = []
    for ep in range(pc["epochs"]):
        rng.shuffle(hours)
        tot, n = 0.0, 0
        for i in range(0, len(hours), pc["batch_hours"]):
            hb = hours[i:i + pc["batch_hours"]]
            v = torch.tensor(cong[hb[:, 0], hb[:, 1]], device=dev)
            e = torch.tensor(F["edge"][hb[:, 0], hb[:, 1]], device=dev)
            m = (torch.rand_like(v) < pc["mask_frac"]).float()
            rec = model.pretrain_forward(v, m, torch.tensor(hb[:, 1], device=dev), e)
            loss = ((rec - v).pow(2) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            n += 1
        hist.append(tot / max(n, 1))
        log(f"    pretrain epoch {ep + 1}/{pc['epochs']}: masked MSE {hist[-1]:.4f}")
    return {"pretrain_loss": hist}


def compute_loss(model, out, b, cfg, phys: PhysicsLoss | None):
    lc = cfg["loss"]
    ys, ms = _mask(to_target(b["y_spread"], cfg))
    L = {"quantile": pinball(out["q"], ys, model.q, ms)}
    yd, md = _mask(torch.asinh(b["y_da"] / 30.0))
    L["da"] = masked_huber(out["da"], yd, md)
    yc, mc = _mask(torch.asinh(b["y_rtcong"] / 10.0))
    L["cong"] = masked_huber(out["cong"], yc, mc)
    yb, mb = _mask(b["y_bind"])
    alive = 1.0 - b["edge"][..., 1].clamp(0, 1)
    L["bind"] = masked_bce(out["bind"], yb, mb * alive)
    total = (lc["w_quantile"] * L["quantile"] + lc["w_da_level"] * L["da"] + lc["w_rt_cong"] * L["cong"]
             + lc["w_bind"] * L["bind"])
    if phys is not None:
        yi, mi = (None, None)
        if "y_inj" in b:
            yi, mi = _mask(b["y_inj"] / 1000.0)
        P = phys(out["theta"], out["inj"], yb, mb, alive, yi, mi)
        L.update({f"phys_{k}": v for k, v in P.items()})
        total = total + lc["w_phys_balance"] * P["balance"] + lc["w_phys_limit"] * P["limit"] + lc["w_phys_inj"] * P["inj"]
    L["total"] = total
    return L


def fit(model: STGNN, F: dict, train_days: np.ndarray, val_days: np.ndarray, cfg: dict, seed: int,
        phys: PhysicsLoss | None = None, log=print) -> dict:
    tc = cfg["train"]
    dev = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    best, best_state, bad = np.inf, None, 0
    curve = []
    for ep in range(tc["epochs"]):
        model.train()
        order = rng.permutation(train_days)
        t0 = time.time()
        tl = []
        for i in range(0, len(order), tc["batch_days"]):
            b = batch(F, order[i:i + tc["batch_days"]], dev)
            out = model(b["node"], b["glob"], b["hist"], b["edge"])
            L = compute_loss(model, out, b, cfg, phys)
            opt.zero_grad()
            L["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(L["total"].item())
        sched.step()
        val = evaluate(model, F, val_days, cfg)
        curve.append({"epoch": ep + 1, "train_loss": float(np.mean(tl)), "val_pinball": val["pinball"],
                      "val_pinball_median": val["pinball_median"], "secs": time.time() - t0})
        log(f"    seed {seed} epoch {ep + 1}: train {np.mean(tl):.4f}  val pinball ${val['pinball']:.3f}"
            f"  ({time.time() - t0:.0f}s)")
        if val["pinball"] < best - 1e-4:
            best, bad = val["pinball"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    return {"curve": curve, "best_val_pinball": float(best)}


@torch.no_grad()
def predict(model: STGNN, F: dict, days: np.ndarray, cfg: dict, bs: int = 16) -> dict:
    model.eval()
    dev = next(model.parameters()).device
    s = cfg["loss"]["target_scale"]
    qs, da, cong, bind = [], [], [], []
    for i in range(0, len(days), bs):
        b = batch(F, days[i:i + bs], dev, labels=False)
        out = model(b["node"], b["glob"], b["hist"], b["edge"])
        qs.append(to_usd(out["q"], cfg).cpu().numpy())
        da.append((30.0 * torch.sinh(out["da"])).cpu().numpy())
        cong.append((10.0 * torch.sinh(out["cong"])).cpu().numpy())
        bind.append(torch.sigmoid(out["bind"]).cpu().numpy())
    return {"q": np.concatenate(qs), "da": np.concatenate(da), "cong": np.concatenate(cong),
            "bind": np.concatenate(bind)}


def pinball_np(q: np.ndarray, y: np.ndarray, taus) -> float:
    taus = np.asarray(taus)
    m = np.isfinite(y)
    e = y[m][:, None] - q[m]
    return float(np.maximum(taus * e, (taus - 1) * e).mean())


@torch.no_grad()
def evaluate(model, F, days, cfg) -> dict:
    p = predict(model, F, days, cfg)
    y = F["y_spread"][days]
    taus = cfg["model"]["quantiles"]
    mid = taus.index(0.5)
    return {"pinball": pinball_np(p["q"], y, taus),
            "pinball_median": pinball_np(p["q"][..., mid:mid + 1], y, [0.5])}


@torch.no_grad()
def attributions(model: STGNN, F: dict, days: np.ndarray, cfg: dict, base_days: np.ndarray) -> tuple[np.ndarray, list]:
    """Occlusion: change in predicted median spread when a feature group is set to its training mean.

    Returns [D,H,N,G] ($/MWh; positive = the group pushes the spread up) and the group names.
    """
    model.eval()
    dev = next(model.parameters()).device
    s = cfg["loss"]["target_scale"]
    mid = cfg["model"]["quantiles"].index(0.5)
    mean_node = F["node"][base_days].mean(axis=(0, 1, 2))
    mean_glob = F["glob"][base_days].mean(axis=(0, 1))
    mean_hist = F["hist"][base_days].mean(axis=(0, 1, 2, 3))
    names = list(DRIVER_GROUPS)
    out = np.zeros((len(days), 24, F["node"].shape[2], len(names)), np.float32)
    for i in range(0, len(days), 16):
        idx = days[i:i + 16]
        b = batch(F, idx, dev, labels=False)
        ref = to_usd(model(b["node"], b["glob"], b["hist"], b["edge"])["q"][..., mid], cfg)
        for gi, g in enumerate(names):
            nf, gf = DRIVER_GROUPS[g]
            node, glob, hist = b["node"].clone(), b["glob"].clone(), b["hist"].clone()
            for f in nf:
                j = NODE_FEATURES.index(f)
                node[..., j] = float(mean_node[j])
            for f in gf:
                j = GLOBAL_FEATURES.index(f)
                glob[..., j] = float(mean_glob[j])
            if g == "Recent spreads":
                hist[:] = torch.tensor(mean_hist, device=dev)
            alt = to_usd(model(node, glob, hist, b["edge"])["q"][..., mid], cfg)
            out[i:i + len(idx), ..., gi] = (ref - alt).cpu().numpy()
    return out, names


def physics_for(F: dict, cfg: dict, source: str) -> PhysicsLoss | None:
    if "physical" not in cfg["model"]["relations"] or source != "sim":
        return None
    return PhysicsLoss(F["lines"], F["reactance"], F["rating"], F["node"].shape[2])


__all__ = ["build_model", "pretrain", "fit", "predict", "evaluate", "attributions", "physics_for",
           "device_of", "pinball_np", "EDGE_FEATURES"]
