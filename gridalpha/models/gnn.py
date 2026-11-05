"""Physics-informed spatio-temporal GNN.

Plain PyTorch (no PyG dependency) so it runs the same on a laptop CPU, a Modal GPU or
a DigitalOcean GPU Droplet. Message passing uses index_add over an edge list.

Relations:
  physical  edge-conditioned attention over transmission lines (or, for real ERCOT data,
            binding constraints attached to the nodes on each side); outaged lines carry
            no messages
  adaptive  a learned dense adjacency (top-k sparsified) from node embeddings, the
            "latent grid" inferred from congestion co-movement
  global    a virtual system node (mean pool -> MLP -> broadcast) for long-range flow
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i: int, h: int, o: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))


class AdaptiveAdjacency(nn.Module):
    def __init__(self, n: int, dim: int = 16, topk: int = 8):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(n, dim) * 0.3)
        self.e2 = nn.Parameter(torch.randn(n, dim) * 0.3)
        self.topk = min(topk, n)

    def forward(self) -> torch.Tensor:
        s = F.relu(self.e1 @ self.e2.T)
        if self.topk < s.shape[0]:
            thr = s.topk(self.topk, dim=1).values[:, -1:]
            s = s.masked_fill(s < thr, float("-inf"))
        return torch.softmax(s, dim=1)


class SpatialLayer(nn.Module):
    def __init__(self, d: int, fe: int, relations: list[str], heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.rel = relations
        self.heads = heads
        self.dh = d // heads
        if "physical" in relations:
            self.ws = nn.Linear(d, d, bias=False)
            self.wd = nn.Linear(d, d, bias=False)
            self.we = nn.Linear(fe, d, bias=False)
            self.att = nn.Parameter(torch.randn(heads, self.dh) / math.sqrt(self.dh))
            self.wm = nn.Linear(d, d)
            self.gate = nn.Sequential(nn.Linear(fe, d), nn.Sigmoid())
        if "adaptive" in relations:
            self.wa = nn.Linear(d, d)
        if "global" in relations:
            self.wg = mlp(d, d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = mlp(d, 2 * d, d, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, z, eattr, src, dst, alive, A):
        # z [G,N,d]; eattr [G,E,fe] (directed, both ways); alive [G,E]; A [N,N]
        Gn, N, d = z.shape
        msg = torch.zeros_like(z)
        if "physical" in self.rel and src.numel():
            hs, hd = z[:, src], z[:, dst]
            pre = F.leaky_relu(self.ws(hs) + self.wd(hd) + self.we(eattr), 0.2)
            score = (pre.view(Gn, -1, self.heads, self.dh) * self.att).sum(-1)       # [G,E,heads]
            score = score - score.amax(dim=1, keepdim=True).detach()
            w = torch.exp(score) * alive[..., None]
            den = torch.zeros(Gn, N, self.heads, device=z.device).index_add_(1, dst, w)
            alpha = w / (den[:, dst] + 1e-9)
            m = (self.wm(hs) * self.gate(eattr)).view(Gn, -1, self.heads, self.dh) * alpha[..., None]
            msg = msg + torch.zeros(Gn, N, self.heads, self.dh, device=z.device).index_add_(1, dst, m).view(Gn, N, d)
        if "adaptive" in self.rel and A is not None:
            msg = msg + torch.einsum("nm,gmd->gnd", A, self.wa(z))
        if "global" in self.rel:
            msg = msg + self.wg(z.mean(1, keepdim=True)).expand(-1, N, -1)
        z = self.norm1(z + self.drop(msg))
        return self.norm2(z + self.drop(self.ffn(z)))


class STGNN(nn.Module):
    def __init__(self, n_nodes: int, lines: torch.Tensor, fn: int, fg: int, fe: int, fh: int = 3, *,
                 hidden: int = 64, layers: int = 3, node_emb: int = 16, quantiles=(0.1, 0.5, 0.9),
                 adaptive_topk: int = 8, relations=("physical", "adaptive", "global"), dropout: float = 0.1):
        super().__init__()
        self.n = n_nodes
        self.register_buffer("lines", lines.long())
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32))
        self.relations = list(relations)
        d = hidden
        self.node_emb = nn.Embedding(n_nodes, node_emb)
        self.hist_gru = nn.GRU(fh, d // 2, batch_first=True)
        self.hour_mix = nn.Conv1d(d // 2, d // 2, kernel_size=3, padding=1)
        self.inp = mlp(fn + fg + d // 2 + node_emb, d, d, dropout)
        self.tconv = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d)
        self.tpoint = nn.Linear(d, d)
        self.adapt = AdaptiveAdjacency(n_nodes, node_emb, adaptive_topk) if "adaptive" in relations else None
        self.layers = nn.ModuleList([SpatialLayer(d, fe, self.relations, dropout=dropout) for _ in range(layers)])
        nq = len(quantiles)
        self.q_head = mlp(d, d, nq)
        self.da_head = mlp(d, d // 2, 1)
        self.cong_head = mlp(d, d // 2, 1)
        self.bind_head = mlp(2 * d + fe, d, 1)
        self.theta_head = mlp(d, d // 2, 1)
        self.inj_head = mlp(d, d // 2, 1)
        # self-supervised pretraining head and input projection
        self.pre_in = mlp(2 + 2 + node_emb, d, d)
        self.pre_out = mlp(d, d, 1)

    # -- graph plumbing -------------------------------------------------------
    def _edges(self, edge: torch.Tensor):
        """edge [G,L,fe] -> directed copies + alive mask (planned outage = feature 1)."""
        a, b = self.lines[:, 0], self.lines[:, 1]
        src = torch.cat([a, b])
        dst = torch.cat([b, a])
        e2 = torch.cat([edge, edge], dim=1)
        alive = 1.0 - e2[..., 1].clamp(0, 1)
        return e2, src, dst, alive

    def _spatial(self, z, edge):
        e2, src, dst, alive = self._edges(edge)
        A = self.adapt() if self.adapt is not None else None
        for layer in self.layers:
            z = layer(z, e2, src, dst, alive, A)
        return z

    # -- supervised forward ------------------------------------------------------
    def forward(self, node, glob, hist, edge):
        """node [B,H,N,fn] glob [B,H,fg] hist [B,K,H,N,fh] edge [B,H,L,fe]."""
        B, H, N, _ = node.shape
        K = hist.shape[1]
        d = self.inp[-1].out_features
        hseq = hist.permute(0, 2, 3, 1, 4).reshape(B * H * N, K, -1)
        _, hh = self.hist_gru(hseq)
        hh = hh[-1].view(B, H, N, -1)
        hh = hh.permute(0, 2, 3, 1).reshape(B * N, -1, H)
        hh = (hh + self.hour_mix(hh)).view(B, N, -1, H).permute(0, 3, 1, 2)
        emb = self.node_emb.weight[None, None].expand(B, H, -1, -1)
        g = glob[:, :, None, :].expand(-1, -1, N, -1)
        z = self.inp(torch.cat([node, g, hh, emb], dim=-1))                      # [B,H,N,d]
        zt = z.permute(0, 2, 3, 1).reshape(B * N, d, H)
        zt = self.tconv(zt).view(B, N, d, H).permute(0, 3, 1, 2)
        z = z + self.tpoint(F.gelu(zt))
        z = self._spatial(z.reshape(B * H, N, d), edge.reshape(B * H, -1, edge.shape[-1]))
        raw = self.q_head(z)
        q = torch.cat([raw[..., :1], raw[..., :1] + torch.cumsum(F.softplus(raw[..., 1:]), -1)], -1)
        a, b = self.lines[:, 0], self.lines[:, 1]
        ez = torch.cat([z[:, a], z[:, b], edge.reshape(B * H, -1, edge.shape[-1])], dim=-1)
        out = {
            "q": q.view(B, H, N, -1),
            "da": self.da_head(z).view(B, H, N),
            "cong": self.cong_head(z).view(B, H, N),
            "bind": self.bind_head(ez).view(B, H, -1),
            "theta": self.theta_head(z).view(B, H, N),
            "inj": self.inj_head(z).view(B, H, N),
            "z": z.view(B, H, N, -1),
        }
        return out

    # -- self-supervised masked-node reconstruction --------------------------------
    def pretrain_forward(self, value, mask, hour, edge):
        """value/mask [G,N]; hour [G]; edge [G,L,fe] -> reconstruction [G,N]."""
        Gn, N = value.shape
        ang = 2 * math.pi * hour.float() / 24.0
        hs = torch.stack([torch.sin(ang), torch.cos(ang)], -1)[:, None, :].expand(-1, N, -1)
        emb = self.node_emb.weight[None].expand(Gn, -1, -1)
        x = torch.cat([(value * (1 - mask))[..., None], mask[..., None], hs, emb], -1)
        z = self._spatial(self.pre_in(x), edge)
        return self.pre_out(z).squeeze(-1)
