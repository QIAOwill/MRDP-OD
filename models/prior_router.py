"""根据缺失结构估计时间先验和关系先验的可靠性。"""
from __future__ import annotations

import torch
import torch.nn as nn


class MissingnessReliabilityRouter(nn.Module):
    """以六个 missingness-support 特征生成逐位置时间先验权重。"""

    def __init__(self, hidden_dim: int = 16, feature_mask: list[float] | None = None) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("router hidden_dim 必须大于 0")
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # 仅用于令训练初始 gate=0.5；不存在任何人工 support 加权公式。
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        values = feature_mask if feature_mask is not None else [1.0] * 6
        if len(values) != 6:
            raise ValueError("router_feature_mask 必须包含 6 个值")
        self.register_buffer("feature_mask", torch.tensor(values, dtype=torch.float32), persistent=True)

    @staticmethod
    def _endpoint_support(
        cond_mask: torch.Tensor,
        endpoint_idx: torch.Tensor,
        num_cities: int,
    ) -> torch.Tensor:
        """按起点或终点聚合 [B,L,E] 条件掩码。"""
        b, l, e = cond_mask.shape
        expanded = endpoint_idx.view(1, 1, e).expand(b, l, e)
        total = torch.zeros(b, l, num_cities, device=cond_mask.device, dtype=cond_mask.dtype)
        total.scatter_add_(2, expanded, cond_mask)
        counts = torch.bincount(endpoint_idx, minlength=num_cities).to(cond_mask.dtype).clamp_min(1.0)
        support_by_city = total / counts.view(1, 1, num_cities)
        return torch.gather(support_by_city, 2, expanded)

    def forward(
        self,
        cond_mask: torch.Tensor,
        temporal_support: torch.Tensor,
        origin_idx: torch.Tensor,
        destination_idx: torch.Tensor,
        num_cities: int,
    ) -> dict[str, torch.Tensor]:
        """返回 ``gate`` 和严格由六个观测支持度组成的 ``router_features``。"""
        if temporal_support.shape[-1] != 3:
            raise ValueError("router 需要三个时间支持度特征（3、7、15 日）")
        pair_support = cond_mask.mean(dim=1, keepdim=True).expand_as(cond_mask)
        origin_support = self._endpoint_support(cond_mask, origin_idx, num_cities)
        destination_support = self._endpoint_support(cond_mask, destination_idx, num_cities)
        features = torch.cat(
            [
                temporal_support,
                pair_support.unsqueeze(-1),
                origin_support.unsqueeze(-1),
                destination_support.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = features * self.feature_mask.to(features).view(1, 1, 1, 6)
        gate = torch.sigmoid(self.mlp(features).squeeze(-1))
        # 冻结硬规则：完整窗口内该 OD 对无任何条件观测时，只使用关系先验。
        gate = torch.where(pair_support <= 0, torch.zeros_like(gate), gate)
        return {"gate": gate, "router_features": features}
