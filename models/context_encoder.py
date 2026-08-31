"""静态城市对属性和动态上下文的轻量编码。"""
from __future__ import annotations

import torch
import torch.nn as nn


class StaticPairEncoder(nn.Module):
    """将标准化城市对属性映射到隐藏空间。"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.net(pair_features)


class DynamicContextEncoder(nn.Module):
    """编码日历、天气以及起终点动态特征。"""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)
