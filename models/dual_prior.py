"""组织时间先验、关系先验和缺失结构路由。"""
from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn

from .prior_router import MissingnessReliabilityRouter
from .relational_prior import RelationalMobilityPrior
from .temporal_prior import MaskedMultiScaleTemporalPrior


VALID_VARIANTS = {"direct", "temporal_only", "relational_only", "fixed_fusion", "full"}


class RoutedDualPrior(nn.Module):
    """按模型变体只实例化实际参与前向传播的先验模块。"""

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
        self.variant = str(cfg.get("variant", "full"))
        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"未知模型变体：{self.variant}")
        self.temporal_scales = tuple(int(value) for value in cfg.get("temporal_scales", [3, 7, 15]))
        if len(self.temporal_scales) != 3 or any(value <= 0 for value in self.temporal_scales):
            raise ValueError("MRDP-OD 时间先验必须包含三个正整数尺度")
        allow_scale_sensitivity = bool(cfg.get("allow_temporal_scale_sensitivity", False))
        if self.temporal_scales != (3, 7, 15) and not allow_scale_sensitivity:
            raise ValueError(
                "主实验的 MRDP-OD 时间先验尺度必须为 [3, 7, 15]；"
                "六个论文实验均使用冻结的时间先验尺度"
            )

        use_temporal = self.variant in {"full", "fixed_fusion", "temporal_only"}
        use_relational = self.variant in {"full", "fixed_fusion", "relational_only"}
        use_router = self.variant == "full"

        self.temporal: MaskedMultiScaleTemporalPrior | None = (
            MaskedMultiScaleTemporalPrior(self.temporal_scales) if use_temporal else None
        )
        self.relational: RelationalMobilityPrior | None = (
            RelationalMobilityPrior(
                num_cities,
                num_pairs,
                pair_feature_dim,
                city_feature_dim,
                context_dim,
                relations,
                cfg,
            )
            if use_relational
            else None
        )
        self.router: MissingnessReliabilityRouter | None = (
            MissingnessReliabilityRouter(
                int(cfg.get("router_hidden_dim", 16)),
                feature_mask=list(cfg.get("router_feature_mask", [1.0] * 6)),
            ) if use_router else None
        )

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
    ) -> dict[str, torch.Tensor]:
        zeros = torch.zeros_like(observed_data)
        zero_support = torch.zeros(
            *observed_data.shape,
            len(self.temporal_scales),
            device=observed_data.device,
            dtype=observed_data.dtype,
        )

        if self.temporal is not None:
            temporal_output = self.temporal(observed_data, cond_mask)
        else:
            temporal_output = {"prior": zeros, "support": zero_support, "scale_weights": zero_support}

        if self.relational is not None:
            relational_prior = self.relational(
                observed_data,
                cond_mask,
                context,
                pair_features,
                city_features,
                origin_idx,
                destination_idx,
                relation_edges,
            )
        else:
            relational_prior = zeros

        empty_structural_support = torch.cat(
            [zero_support, zeros.unsqueeze(-1), zeros.unsqueeze(-1), zeros.unsqueeze(-1)], dim=-1
        )
        if self.variant == "full":
            if self.router is None:
                raise RuntimeError("full 变体缺少 router")
            router_output = self.router(
                cond_mask,
                temporal_output["support"],
                origin_idx,
                destination_idx,
                int(city_features.shape[0]),
            )
            gate = router_output["gate"]
            router_features = router_output["router_features"]
            prior = gate * temporal_output["prior"] + (1.0 - gate) * relational_prior
        elif self.variant == "direct":
            gate = zeros
            router_features = empty_structural_support
            prior = zeros
        elif self.variant == "temporal_only":
            gate = torch.ones_like(zeros)
            router_features = torch.cat(
                [temporal_output["support"], zeros.unsqueeze(-1), zeros.unsqueeze(-1), zeros.unsqueeze(-1)],
                dim=-1,
            )
            prior = temporal_output["prior"]
        elif self.variant == "relational_only":
            gate = zeros
            router_features = empty_structural_support
            prior = relational_prior
        elif self.variant == "fixed_fusion":
            gate = torch.full_like(zeros, 0.5)
            router_features = torch.cat(
                [temporal_output["support"], zeros.unsqueeze(-1), zeros.unsqueeze(-1), zeros.unsqueeze(-1)],
                dim=-1,
            )
            prior = 0.5 * temporal_output["prior"] + 0.5 * relational_prior
        else:  # pragma: no cover - 构造函数已检查
            raise RuntimeError(f"未处理的模型变体：{self.variant}")

        # 条件位置始终等于真实观测；辅助损失只在 target mask 上计算。
        prior = cond_mask * observed_data + (1.0 - cond_mask) * prior
        return {
            "temporal_prior": temporal_output["prior"],
            "relational_prior": relational_prior,
            "temporal_support": temporal_output["support"],
            "scale_weights": temporal_output["scale_weights"],
            "router_features": router_features,
            "gate": gate,
            "prior": prior,
        }

    def relation_weights(self) -> dict[str, float]:
        """无关系先验的消融返回空字典。"""
        return self.relational.relation_weights() if self.relational is not None else {}

    def parameter_breakdown(self) -> dict[str, int]:
        """返回先验子模块的真实实例化参数量。"""
        def count(module: nn.Module | None) -> int:
            return 0 if module is None else sum(parameter.numel() for parameter in module.parameters())

        return {
            "temporal_prior": count(self.temporal),
            "relational_prior": count(self.relational),
            "router": count(self.router),
        }
