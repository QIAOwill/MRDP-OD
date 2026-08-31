"""沿每条 OD 序列建模局部和周尺度变化的时间残差块。"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    """使用普通卷积和膨胀卷积形成多尺度时间感受野。"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.local = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.weekly = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3, padding=3)
        self.mix = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """输入和输出均为 [B,L,E,D]。"""
        b, l, e, d = h.shape
        x = h.permute(0, 2, 3, 1).reshape(b * e, d, l)
        local = self.local(x).transpose(1, 2)
        weekly = self.weekly(x).transpose(1, 2)
        mixed = self.mix(torch.cat([local, weekly], dim=-1))
        mixed = mixed.reshape(b, e, l, d).permute(0, 2, 1, 3)
        return self.norm(h + mixed)
