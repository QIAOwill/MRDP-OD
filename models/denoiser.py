"""先验中心残差扩散的条件噪声预测网络。"""
from __future__ import annotations

import math
from typing import Dict, Tuple
import torch
import torch.nn as nn

from .context_encoder import DynamicContextEncoder, StaticPairEncoder
from .graph_encoder import SparseMultiRelationGraphEncoder
from .od_tokenizer import DirectedODPairTokenizer
from .temporal import TemporalConvBlock


def sinusoidal_embedding(steps: torch.Tensor, dim: int) -> torch.Tensor:
    """生成扩散步正弦余弦编码。"""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=steps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    values = steps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(values), torch.cos(values)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PriorCenteredDenoiser(nn.Module):
    """融合 noisy residual、条件值、双先验、router 和有向多关系图。"""

    def __init__(
        self,
        num_cities: int,
        num_pairs: int,
        pair_feature_dim: int,
        city_feature_dim: int,
        context_dim: int,
        relations: list[str],
        cfg: dict,
    ) -> None:
        super().__init__()
        hidden = int(cfg.get("hidden_dim", 96))
        dropout = float(cfg.get("dropout", 0.1))
        num_layers = int(cfg.get("num_layers", 2))
        time_dim = int(cfg.get("time_emb_dim", hidden))
        self.input_projection = nn.Linear(5, hidden)
        self.tokenizer = DirectedODPairTokenizer(
            num_cities=num_cities,
            num_pairs=num_pairs,
            hidden_dim=hidden,
            pair_emb_dim=int(cfg.get("pair_emb_dim", 32)),
            use_direction=bool(cfg.get("use_direction", True)),
            use_pair_id=bool(cfg.get("use_pair_id", False)),
            city_feature_dim=city_feature_dim,
            city_token_mode=str(cfg.get("city_token_mode", "transductive")),
        )
        self.pair_encoder = StaticPairEncoder(pair_feature_dim, hidden, dropout)
        self.context_encoder = DynamicContextEncoder(context_dim, hidden, dropout)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.time_dim = time_dim
        self.temporal_layers = nn.ModuleList([TemporalConvBlock(hidden, dropout) for _ in range(num_layers)])
        self.graph_layers = nn.ModuleList([
            SparseMultiRelationGraphEncoder(hidden, relations, dropout=dropout, num_layers=1)
            for _ in range(num_layers)
        ])
        self.film = nn.Linear(hidden, hidden * 2)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        noisy_residual: torch.Tensor,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        prior: torch.Tensor,
        gate: torch.Tensor,
        context: torch.Tensor,
        diffusion_step: torch.Tensor,
        pair_features: torch.Tensor,
        city_features: torch.Tensor,
        origin_idx: torch.Tensor,
        destination_idx: torch.Tensor,
        relation_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        channels = torch.stack(
            [
                noisy_residual * (1.0 - cond_mask),
                observed_data * cond_mask,
                cond_mask,
                prior,
                gate,
            ],
            dim=-1,
        )
        h = self.input_projection(channels)
        token = self.tokenizer(origin_idx, destination_idx, city_features)
        pair = self.pair_encoder(pair_features)
        dynamic = self.context_encoder(context)
        step = self.time_mlp(sinusoidal_embedding(diffusion_step, self.time_dim))
        h = (
            h
            + token.view(1, 1, token.shape[0], -1)
            + pair.view(1, 1, pair.shape[0], -1)
            + dynamic
            + step.view(step.shape[0], 1, 1, -1)
        )
        gamma, beta = self.film(pair).chunk(2, dim=-1)
        h = (1.0 + torch.tanh(gamma).view(1, 1, pair.shape[0], -1)) * h + beta.view(1, 1, pair.shape[0], -1)
        for temporal, graph in zip(self.temporal_layers, self.graph_layers):
            h = temporal(h)
            h = graph(h, relation_edges)
        return self.output(h).squeeze(-1)

    def relation_weights(self) -> dict[str, dict[str, float]]:
        """返回每层去噪图编码器的关系权重。"""
        return {f"denoiser_layer_{i + 1}": layer.relation_weights() for i, layer in enumerate(self.graph_layers)}
