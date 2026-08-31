"""为 unseen-OD 与 leave-city-out 实验生成可复现的实体留出划分。"""
from __future__ import annotations

from typing import Any
import numpy as np


def held_out_pair_mask(
    origin_idx: np.ndarray,
    destination_idx: np.ndarray,
    num_cities: int,
    spec: dict[str, Any] | None,
) -> np.ndarray:
    """返回长度为 E 的布尔数组；True 表示该 OD 在训练和验证中不可见。"""
    count = int(len(origin_idx))
    result = np.zeros(count, dtype=bool)
    if not spec or not bool(spec.get("enabled", True)):
        return result
    kind = str(spec.get("type", "none"))
    if kind in {"", "none"}:
        return result
    rng = np.random.RandomState(int(spec.get("seed", 32024)))
    if kind == "unseen_od":
        fraction = float(spec.get("fraction", 0.2))
        requested = int(spec.get("pair_count") or round(count * fraction))
        requested = min(max(1, requested), max(1, count - 1))
        selected = rng.choice(count, size=requested, replace=False)
        result[selected] = True
    elif kind == "leave_city_out":
        explicit = spec.get("city_indices")
        if explicit is not None:
            cities = np.asarray([int(value) for value in explicit], dtype=np.int64)
        else:
            city_count = min(max(1, int(spec.get("city_count", 1))), max(1, int(num_cities) - 1))
            cities = rng.choice(int(num_cities), size=city_count, replace=False)
        result = np.isin(origin_idx, cities) | np.isin(destination_idx, cities)
    else:
        raise ValueError(f"未知实体划分类型：{kind}")
    if not np.any(result) or np.all(result):
        raise ValueError(f"实体划分 {kind} 产生空留出集或留出了全部 OD")
    return result.astype(bool)


def entity_split_manifest(mask: np.ndarray, spec: dict[str, Any] | None) -> dict[str, Any]:
    """返回可写入运行元数据的实体划分摘要。"""
    values = np.asarray(mask, dtype=bool)
    return {
        "type": str((spec or {}).get("type", "none")),
        "seed": (spec or {}).get("seed"),
        "held_out_pair_count": int(values.sum()),
        "available_pair_count": int((~values).sum()),
        "held_out_pair_indices": np.where(values)[0].astype(int).tolist(),
    }
