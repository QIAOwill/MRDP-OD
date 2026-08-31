"""有向起点—终点角色编码与可选 OD 对标识编码。"""
from __future__ import annotations

import torch
import torch.nn as nn


class DirectedODPairTokenizer(nn.Module):
    """区分城市作为起点和终点时的角色，并生成 OD-pair token。"""

    def __init__(
        self,
        num_cities: int,
        num_pairs: int,
        hidden_dim: int,
        pair_emb_dim: int = 32,
        use_direction: bool = True,
        use_pair_id: bool = False,
        city_feature_dim: int = 0,
        city_token_mode: str = "transductive",
    ) -> None:
        super().__init__()
        self.use_direction = bool(use_direction)
        self.use_pair_id = bool(use_pair_id)
        self.city_token_mode = str(city_token_mode)
        if self.city_token_mode not in {"transductive", "inductive"}:
            raise ValueError("city_token_mode 只能是 transductive 或 inductive")
        if self.city_token_mode == "inductive" and self.use_pair_id:
            raise ValueError("inductive tokenizer 不能使用 pair identity embedding")
        role_dim = max(8, hidden_dim // 3)
        if self.city_token_mode == "transductive" and self.use_direction:
            self.origin_embedding: nn.Module | None = nn.Embedding(num_cities, role_dim)
            self.destination_embedding: nn.Module | None = nn.Embedding(num_cities, role_dim)
            self.origin_role = None
            self.destination_role = None
        elif self.city_token_mode == "transductive":
            shared = nn.Embedding(num_cities, role_dim)
            self.origin_embedding = shared
            self.destination_embedding = shared
            self.origin_role = None
            self.destination_role = None
        else:
            self.origin_embedding = None
            self.destination_embedding = None
            self.origin_role = nn.Parameter(torch.zeros(role_dim))
            if self.use_direction:
                self.destination_role = nn.Parameter(torch.zeros(role_dim))
            else:
                self.destination_role = self.origin_role
            nn.init.normal_(self.origin_role, std=0.02)
            if self.destination_role is not self.origin_role:
                nn.init.normal_(self.destination_role, std=0.02)
        self.city_feature_encoder = (
            nn.Sequential(nn.Linear(city_feature_dim, role_dim), nn.GELU(), nn.LayerNorm(role_dim))
            if city_feature_dim > 0 else None
        )
        input_dim = role_dim * 2
        if self.use_pair_id:
            self.pair_embedding = nn.Embedding(num_pairs, pair_emb_dim)
            input_dim += pair_emb_dim
        else:
            self.pair_embedding = None
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        origin_idx: torch.Tensor,
        destination_idx: torch.Tensor,
        city_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回 [E,D] 的有向 OD token。"""
        if self.city_token_mode == "transductive":
            if self.origin_embedding is None or self.destination_embedding is None:
                raise RuntimeError("transductive tokenizer 缺少城市 identity embedding")
            origin = self.origin_embedding(origin_idx)
            destination = self.destination_embedding(destination_idx)
        else:
            if self.city_feature_encoder is None or city_features is None:
                raise ValueError("inductive tokenizer 必须提供城市静态属性")
            encoded = self.city_feature_encoder(city_features)
            origin = encoded[origin_idx] + self.origin_role.view(1, -1)
            destination = encoded[destination_idx] + self.destination_role.view(1, -1)
        if self.city_feature_encoder is not None and city_features is not None:
            if self.city_token_mode == "transductive":
                encoded_city = self.city_feature_encoder(city_features)
                origin = origin + encoded_city[origin_idx]
                destination = destination + encoded_city[destination_idx]
        parts = [origin, destination]
        if self.pair_embedding is not None:
            pair_idx = torch.arange(origin_idx.numel(), device=origin_idx.device)
            parts.append(self.pair_embedding(pair_idx))
        return self.projection(torch.cat(parts, dim=-1))
