"""Losses: pinball for quantiles, plus the DC power-flow physics terms."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pinball(pred: torch.Tensor, y: torch.Tensor, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred [...,Q], y [...], q [Q], mask [...] -> mean pinball over valid entries."""
    e = y[..., None] - pred
    l = torch.maximum(q * e, (q - 1) * e).mean(-1)
    return (l * mask).sum() / mask.sum().clamp(min=1)


def masked_huber(pred, y, mask, delta=1.0):
    l = F.huber_loss(pred, y, reduction="none", delta=delta)
    return (l * mask).sum() / mask.sum().clamp(min=1)


def masked_bce(logit, y, mask):
    l = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    return (l * mask).sum() / mask.sum().clamp(min=1)


class PhysicsLoss(torch.nn.Module):
    """DC power flow consistency on the physical sub-graph.

    theta, inj: [B,H,N] (inj in GW). balance: B_bus @ theta = inj; sum(inj) = 0;
    limits: |b_l (theta_i - theta_j)| <= F_l; flows that reach the limit should line
    up with binding labels; optional supervised injections where labels exist.
    """

    def __init__(self, lines: np.ndarray, reactance: np.ndarray, rating: np.ndarray, n: int):
        super().__init__()
        b = 1.0 / np.asarray(reactance, dtype=float)
        b = b / b.mean()
        A = np.zeros((len(lines), n))
        A[np.arange(len(lines)), lines[:, 0]] = 1.0
        A[np.arange(len(lines)), lines[:, 1]] = -1.0
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float32))
        self.register_buffer("Bbus", torch.tensor(A.T @ (b[:, None] * A), dtype=torch.float32))
        self.register_buffer("F", torch.tensor(np.asarray(rating) / 1000.0, dtype=torch.float32))

    def forward(self, theta, inj, y_bind=None, bind_mask=None, alive=None, y_inj=None, inj_mask=None):
        bal = ((theta @ self.Bbus.T) - inj).pow(2).mean() + inj.sum(-1).pow(2).mean() * 1e-2
        flow = (theta @ self.A.T) * self.b                               # [B,H,L] GW
        over = F.relu(flow.abs() - self.F)
        if alive is not None:
            over = over * alive
        lim = over.pow(2).mean()
        fb = torch.tensor(0.0, device=theta.device)
        if y_bind is not None:
            logit = 20.0 * (flow.abs() / self.F - 0.97)
            m = bind_mask if bind_mask is not None else torch.ones_like(y_bind)
            if alive is not None:
                m = m * alive
            fb = masked_bce(logit, y_bind, m)
        sup = torch.tensor(0.0, device=theta.device)
        if y_inj is not None:
            sup = ((inj - y_inj).pow(2) * inj_mask).sum() / inj_mask.sum().clamp(min=1)
        return {"balance": bal, "limit": lim + 0.1 * fb, "inj": sup}
