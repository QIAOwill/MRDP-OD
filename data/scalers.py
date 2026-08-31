"""组合多个源区域 scaler，并在目标区域只执行 transform。"""
from __future__ import annotations

from typing import Iterable, Any
import numpy as np

from .tensor_builder import RegionTensorData, scaler_metadata


def _pooled_vector(states: list[dict[str, Any]], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.asarray([item["mean"] for item in states], dtype=np.float64)
    stds = np.asarray([item["std"] for item in states], dtype=np.float64)
    medians = np.asarray([item["imputation_median"] for item in states], dtype=np.float64)
    normalised = weights / weights.sum()
    mean = np.sum(means * normalised[:, None], axis=0)
    variance = np.sum(normalised[:, None] * (stds ** 2 + (means - mean) ** 2), axis=0)
    # 多源中位数不能由摘要精确恢复；使用仅依赖源域的加权中位数摘要，并在 metadata 标注。
    median = np.sum(medians * normalised[:, None], axis=0)
    return median.astype(np.float32), mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)


def combine_source_scalers(regions: Iterable[RegionTensorData]) -> dict[str, Any]:
    """用源区域统计量构造不读取目标域分布的共享 scaler。"""
    items = list(regions)
    if not items:
        raise ValueError("至少需要一个源区域")
    metadata = [scaler_metadata(item) for item in items]
    target_weights = np.asarray([item.native_mask[item.train_idx].sum() for item in items], dtype=np.float64)
    target_means = np.asarray([item.x_mean for item in items], dtype=np.float64)
    target_stds = np.asarray([item.x_std for item in items], dtype=np.float64)
    target_weight = target_weights / target_weights.sum()
    target_mean = float(np.sum(target_means * target_weight))
    target_variance = float(np.sum(target_weight * (target_stds ** 2 + (target_means - target_mean) ** 2)))
    result: dict[str, Any] = {
        "target": {"mean": target_mean, "std": float(np.sqrt(max(target_variance, 1e-12)))},
        "fit_scope": "source_regions_only",
        "source_regions": [item.region for item in items],
        "median_combination": "weighted_source_summary",
    }
    section_weights = {
        "context": np.asarray([len(item.train_idx) * item.num_pairs for item in items], dtype=np.float64),
        "pair_static": np.asarray([item.num_pairs for item in items], dtype=np.float64),
        "city_static": np.asarray([item.num_cities for item in items], dtype=np.float64),
    }
    for section, weights in section_weights.items():
        states = [value[section] for value in metadata]
        names = list(states[0]["feature_names"])
        if any(list(state["feature_names"]) != names for state in states[1:]):
            raise ValueError(f"多源 {section} feature schema 不一致")
        median, mean, std = _pooled_vector(states, weights)
        result[section] = {
            "feature_names": names,
            "imputation_median": median.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    return result
