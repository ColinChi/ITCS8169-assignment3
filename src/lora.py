"""LoRA on the fused qkv projections of a timm ViT (CLIP-LoRA, Zanella & Ben Ayed, CVPRW 2024).

CLIP-LoRA adapts W_q, W_k, W_v of every attention layer with separate rank-r updates B A (A Kaiming-uniform,
B zero, scaling alpha / r, dropout on the LoRA input). timm fuses q, k, v into one Linear whose output is
[q | k | v], so the three updates are stacked along the output dimension.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRAQKV(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 2, alpha: float = 1.0, dropout: float = 0.25):
        super().__init__()
        self.base = base
        d_in, d_out = base.in_features, base.out_features // 3
        self.A = nn.Parameter(torch.empty(3, r, d_in))
        self.B = nn.Parameter(torch.zeros(3, d_out, r))
        for k in range(3):
            nn.init.kaiming_uniform_(self.A[k], a=math.sqrt(5))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.einsum("...i,kri->...kr", self.dropout(x), self.A.to(x.dtype))
        delta = torch.einsum("...kr,kdr->...kd", h, self.B.to(x.dtype)).flatten(-2)
        return self.base(x) + self.scaling * delta


def add_lora(trunk: nn.Module, **kw) -> list[LoRAQKV]:
    """Wrap blocks[i].attn.qkv of a timm VisionTransformer; returns the LoRA modules."""
    mods = []
    for blk in trunk.blocks:
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, **kw)
        mods.append(blk.attn.qkv)
    return mods


def lora_state(mods: list[LoRAQKV]) -> dict[str, torch.Tensor]:
    return {f"{i}.{n}": p.detach().cpu() for i, m in enumerate(mods) for n, p in (("A", m.A), ("B", m.B))}


def load_lora_state(mods: list[LoRAQKV], state: dict[str, torch.Tensor]) -> None:
    for i, m in enumerate(mods):
        m.A.data.copy_(state[f"{i}.A"])
        m.B.data.copy_(state[f"{i}.B"])
