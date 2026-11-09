"""Shared AdaptFormer bottleneck: down, ReLU, dropout, up, scale."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Adapter(nn.Module):
    def __init__(
        self,
        dim: int,
        bottleneck: int,
        scale: float = 1.0,
        dropout: float = 0.0,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.scale, self.dropout = float(scale), dropout
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.layer_norm(features)
        hidden = F.dropout(
            F.relu(self.down(features)), p=self.dropout, training=self.training
        )
        return self.scale * self.up(hidden)
