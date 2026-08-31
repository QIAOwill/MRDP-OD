"""掩码归一化的多尺度时间先验。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiScaleTemporalPrior(nn.Module):
    """使用三个可配置时间窗口的掩码平均卷积构造局部时间先验。"""

    def __init__(self, scales: list[int] | tuple[int, ...] = (3, 7, 15), gate_hidden: int = 12) -> None:
        super().__init__()
        self.scales = tuple(int(scale) for scale in scales)
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("时间先验尺度必须为正整数")
        self.scale_gate = nn.Sequential(
            nn.Linear(len(self.scales), gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, len(self.scales)),
        )
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)

    @staticmethod
    def _masked_average(values: torch.Tensor, mask: torch.Tensor, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
        """对 [B,L,E] 执行按 OD 分组的掩码平均卷积。"""
        b, l, e = values.shape
        value_series = (values * mask).permute(0, 2, 1).reshape(b * e, 1, l)
        mask_series = mask.permute(0, 2, 1).reshape(b * e, 1, l)
        kernel = torch.ones(1, 1, scale, device=values.device, dtype=values.dtype)
        # 偶数窗口采用左少一格、右多一格的确定性非对称填充，输出长度仍为 L。
        left = (scale - 1) // 2
        right = scale // 2
        numerator = F.conv1d(F.pad(value_series, (left, right)), kernel)
        denominator = F.conv1d(F.pad(mask_series, (left, right)), kernel)
        estimate = numerator / denominator.clamp_min(1e-6)
        support = denominator / float(scale)
        estimate = estimate.reshape(b, e, l).permute(0, 2, 1)
        support = support.reshape(b, e, l).permute(0, 2, 1).clamp(0.0, 1.0)
        return estimate, support

    def forward(self, observed_data: torch.Tensor, cond_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        estimates: list[torch.Tensor] = []
        supports: list[torch.Tensor] = []
        for scale in self.scales:
            estimate, support = self._masked_average(observed_data, cond_mask, scale)
            estimates.append(estimate)
            supports.append(support)
        estimate_stack = torch.stack(estimates, dim=-1)
        support_stack = torch.stack(supports, dim=-1)
        logits = self.scale_gate(support_stack)
        logits = logits.masked_fill(support_stack <= 0, -1e4)
        all_empty = support_stack.sum(dim=-1, keepdim=True) <= 0
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(all_empty, torch.zeros_like(weights), weights)
        prior = (weights * estimate_stack).sum(dim=-1)
        prior = cond_mask * observed_data + (1.0 - cond_mask) * prior
        return {"prior": prior, "support": support_stack, "scale_weights": weights}
