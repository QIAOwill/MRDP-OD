"""构造有向 OD-pair 的六类稀疏关系图。"""
from __future__ import annotations

from typing import Dict
import numpy as np
import torch


def _normalize_topk(matrix: np.ndarray, topk: int, self_loop: bool) -> np.ndarray:
    """每行保留 top-k，并执行行归一化。"""
    matrix = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = matrix.shape[0]
    if self_loop:
        np.fill_diagonal(matrix, np.maximum(np.diag(matrix), 1.0))
    if 0 < topk < n:
        indices = np.argpartition(-matrix, kth=topk - 1, axis=1)[:, :topk]
        keep = np.zeros_like(matrix, dtype=bool)
        keep[np.arange(n)[:, None], indices] = True
        matrix = np.where(keep, matrix, 0.0)
    sums = matrix.sum(axis=1, keepdims=True)
    sums[sums <= 1e-12] = 1.0
    return matrix / sums


def _rbf(features: np.ndarray, sigma: float) -> np.ndarray:
    """从标准化特征构造 RBF 相似度。"""
    features = np.nan_to_num(features.astype(np.float32))
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    z = (features - mean) / std
    squared = ((z[:, None, :] - z[None, :, :]) ** 2).mean(axis=-1)
    return np.exp(-squared / max(float(sigma), 1e-6)).astype(np.float32)


def build_relation_edges(pair_frame, cfg: dict) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """返回关系名到 edge_index、edge_weight 的字典。"""
    origins = pair_frame["origin_id"].to_numpy()
    destinations = pair_frame["destination_id"].to_numpy()
    e = len(pair_frame)
    relations: dict[str, np.ndarray] = {
        "shared_origin": (origins[:, None] == origins[None, :]).astype(np.float32),
        "shared_destination": (destinations[:, None] == destinations[None, :]).astype(np.float32),
        "reverse": ((origins[:, None] == destinations[None, :]) & (destinations[:, None] == origins[None, :])).astype(np.float32),
    }
    geo_cols = [c for c in ["distance_line", "distance_road", "distance_railway", "is_adjacent", "same_province"] if c in pair_frame]
    hsr_cols = [c for c in ["hsr_direct_flag", "hsr_train_count", "hsr_min_travel_time", "hsr_avg_travel_time", "hsr_service_intensity"] if c in pair_frame]
    socio_cols = [c for c in ["gdp_gap", "income_gap", "population_gap", "industry_structure_similarity", "poi_structure_similarity"] if c in pair_frame]
    sigma = float(cfg.get("similarity_sigma", 1.0))
    relations["geo"] = _rbf(pair_frame[geo_cols].to_numpy(float), sigma) if geo_cols else np.eye(e, dtype=np.float32)
    relations["hsr"] = _rbf(pair_frame[hsr_cols].to_numpy(float), sigma) if hsr_cols else np.eye(e, dtype=np.float32)
    relations["socio"] = _rbf(pair_frame[socio_cols].to_numpy(float), sigma) if socio_cols else np.eye(e, dtype=np.float32)

    requested = cfg.get("relations", list(relations))
    topk = int(cfg.get("topk", 24))
    self_loop = bool(cfg.get("self_loop", True))
    output: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in requested:
        if name not in relations:
            raise ValueError(f"未知图关系：{name}")
        adjacency = _normalize_topk(relations[name], topk, self_loop)
        dst, src = np.nonzero(adjacency)
        edge_index = torch.from_numpy(np.vstack([src, dst]).astype(np.int64))
        edge_weight = torch.from_numpy(adjacency[dst, src].astype(np.float32))
        output[name] = (edge_index, edge_weight)
    return output
