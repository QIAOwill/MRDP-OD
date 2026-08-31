"""纯 PyTorch 稀疏多关系 OD-pair 图编码器。"""
from __future__ import annotations
from contextlib import nullcontext

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMultiRelationGraphEncoder(nn.Module):
    """
    对 [B,L,E,D] 执行多关系消息传播。

    稀疏邻接左乘被转换为 [E,E] @ [E,B*L*D]，避免复制 batch*time 份边。
    """

    def __init__(self, hidden_dim: int, relations: list[str], dropout: float = 0.1, num_layers: int = 1) -> None:
        super().__init__()
        self.relations = list(relations)
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self.rel_logits = nn.Parameter(torch.zeros(len(self.relations)))
        self.linears = nn.ModuleList([
            nn.ModuleDict({name: nn.Linear(hidden_dim, hidden_dim) for name in self.relations})
            for _ in range(self.num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])
        self._sparse_cache: dict[tuple[str, str, int], torch.Tensor] = {}

    def _adjacency(
        self,
        name: str,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """在目标设备上构造并缓存 FP32 稀疏邻接矩阵。"""
        key = (name, str(device), num_nodes)
        cached = self._sparse_cache.get(key)
        if cached is not None:
            return cached

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"关系 {name} 的 edge_index 形状应为 [2, num_edges]，"
                f"实际为 {tuple(edge_index.shape)}"
            )

        if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"关系 {name} 的 edge_weight 数量与边数量不一致："
                f"{edge_weight.shape[0]} vs {edge_index.shape[1]}"
            )

        src = edge_index[0].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        dst = edge_index[1].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        values = edge_weight.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        if src.numel() > 0:
            minimum = int(torch.minimum(src.min(), dst.min()).item())
            maximum = int(torch.maximum(src.max(), dst.max()).item())
            if minimum < 0 or maximum >= num_nodes:
                raise ValueError(
                    f"关系 {name} 的节点索引越界："
                    f"合法范围为 [0, {num_nodes - 1}]，实际为 [{minimum}, {maximum}]"
                )

        indices = torch.stack([dst, src], dim=0)

        amp_disabled = (
            torch.autocast(device_type="cuda", enabled=False)
            if device.type == "cuda"
            else nullcontext()
        )

        with amp_disabled:
            adjacency = torch.sparse_coo_tensor(
                indices=indices,
                values=values,
                size=(num_nodes, num_nodes),
                dtype=torch.float32,
                device=device,
                check_invariants=False,
            ).coalesce()

        self._sparse_cache[key] = adjacency
        return adjacency

    @staticmethod
    def _propagate(
        adjacency: torch.Tensor,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行多关系图传播。

        稀疏矩阵乘法固定使用 FP32，传播结果再恢复为输入 hidden state
        的数据类型，使其余网络层仍可使用 BF16 混合精度。
        """
        del edge_index, edge_weight

        b, l, e, d = h.shape
        output_dtype = h.dtype

        amp_disabled = (
            torch.autocast(device_type="cuda", enabled=False)
            if h.device.type == "cuda"
            else nullcontext()
        )

        with amp_disabled:
            matrix = (
                h.float()
                .permute(2, 0, 1, 3)
                .contiguous()
                .reshape(e, b * l * d)
            )

            adjacency_fp32 = adjacency.float()

            if h.device.type == "cpu" and e <= 128:
                propagated = adjacency_fp32.to_dense().matmul(matrix)
            else:
                propagated = torch.sparse.mm(adjacency_fp32, matrix)

        propagated = propagated.to(dtype=output_dtype)

        return propagated.reshape(e, b, l, d).permute(1, 2, 0, 3)

    def forward(self, h: torch.Tensor, relation_edges: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        weights = torch.softmax(self.rel_logits, dim=0)
        out = h
        for layer_index in range(self.num_layers):
            mixed = torch.zeros_like(out)
            active_weight = torch.zeros((), device=out.device)
            for relation_index, name in enumerate(self.relations):
                if name not in relation_edges:
                    continue
                edge_index, edge_weight = relation_edges[name]
                adjacency = self._adjacency(name, edge_index, edge_weight, out.shape[2], out.device)
                message = self._propagate(adjacency, out, edge_index, edge_weight)
                message = self.linears[layer_index][name](message)
                mixed = mixed + weights[relation_index] * message
                active_weight = active_weight + weights[relation_index]
            mixed = mixed / active_weight.clamp_min(1e-6)
            mixed = F.gelu(mixed)
            mixed = F.dropout(mixed, p=self.dropout, training=self.training)
            out = self.norms[layer_index](out + mixed)
        return out

    def relation_weights(self) -> dict[str, float]:
        """返回当前全局关系权重。"""
        weights = torch.softmax(self.rel_logits.detach().cpu(), dim=0)
        return {name: float(value) for name, value in zip(self.relations, weights)}
