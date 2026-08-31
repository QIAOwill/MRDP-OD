"""由有向 OD token、静态属性、动态上下文和多关系图生成关系先验。"""
from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn

from .context_encoder import DynamicContextEncoder, StaticPairEncoder
from .graph_encoder import SparseMultiRelationGraphEncoder
from .od_tokenizer import DirectedODPairTokenizer
from .temporal import TemporalConvBlock


class RelationalMobilityPrior(nn.Module):
    """在局部时间证据不足时提供可归纳的粗粒度 mobility 先验。"""

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
        hidden = int(cfg.get("prior_hidden_dim", 64))
        dropout = float(cfg.get("dropout", 0.1))
        self.input_projection = nn.Linear(2, hidden)
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
        self.temporal = TemporalConvBlock(hidden, dropout)
        self.graph = SparseMultiRelationGraphEncoder(
            hidden,
            relations,
            dropout=dropout,
            num_layers=int(cfg.get("prior_graph_layers", 1)),
        )
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(
        self,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        context: torch.Tensor,
        pair_features: torch.Tensor,
        city_features: torch.Tensor,
        origin_idx: torch.Tensor,
        destination_idx: torch.Tensor,
        relation_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        numerical = torch.stack([observed_data * cond_mask, cond_mask], dim=-1)
        h = self.input_projection(numerical)
        token = self.tokenizer(origin_idx, destination_idx, city_features)
        pair = self.pair_encoder(pair_features)
        dynamic = self.context_encoder(context)
        h = h + token.view(1, 1, token.shape[0], -1) + pair.view(1, 1, pair.shape[0], -1) + dynamic
        h = self.temporal(h)
        h = self.graph(h, relation_edges)
        prior = self.output(h).squeeze(-1)
        return cond_mask * observed_data + (1.0 - cond_mask) * prior

    def relation_weights(self) -> dict[str, float]:
        return self.graph.relation_weights()
