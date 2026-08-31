"""生成只作用于原生可观测位置的实验缺失掩码。"""
from __future__ import annotations

import numpy as np


def _normalized_score(values: np.ndarray) -> np.ndarray:
    """把任意 driver 转成稳定的 [0,1] 排名分数，避免量纲影响抽样。"""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(flat)
    result = np.zeros_like(flat)
    if finite.any():
        order = np.argsort(np.argsort(flat[finite], kind="mergesort"), kind="mergesort")
        result[finite] = order / max(len(order) - 1, 1)
        result[~finite] = float(np.median(result[finite]))
    return result.reshape(np.asarray(values).shape)


def driver_weighted_mask(
    native_mask: np.ndarray,
    rate: float,
    rng: np.random.RandomState,
    driver: np.ndarray,
    strength: float = 3.0,
) -> np.ndarray:
    """按协变量/MNAR driver 加权且精确抽取目标数量。"""
    native = np.asarray(native_mask) > 0
    score = np.broadcast_to(np.asarray(driver, dtype=np.float64), native.shape)
    rank = _normalized_score(score)
    positions = np.argwhere(native)
    count = _target_count(native_mask, rate)
    target = np.zeros_like(native_mask, dtype=np.float32)
    if count <= 0:
        return target
    weights = np.exp(np.clip(float(strength) * (rank[native] - 0.5), -20.0, 20.0))
    weights = weights / weights.sum()
    chosen = rng.choice(len(positions), size=count, replace=False, p=weights)
    target[tuple(positions[chosen].T)] = 1.0
    return target


def native_like_mask(
    native_mask: np.ndarray,
    rate: float,
    rng: np.random.RandomState,
    native_reference: np.ndarray | None,
) -> np.ndarray:
    """仅从训练/验证原生掩码学习日级故障和 OD 特异倾向，再回溯遮挡有真值位置。"""
    if native_reference is None or np.asarray(native_reference).ndim != 2:
        return random_mask(native_mask, rate, rng)
    reference = np.asarray(native_reference, dtype=np.float32)
    pair_missing = 1.0 - reference.mean(axis=0)
    pair_driver = np.broadcast_to(pair_missing.reshape(1, -1), native_mask.shape)
    target = np.zeros_like(native_mask, dtype=np.float32)
    desired = _target_count(native_mask, rate)

    # 先复现平台级整日/连续故障所占比例，再用 OD 特异倾向补足精确数量。
    full_days = np.where(reference.sum(axis=1) == 0)[0]
    system_fraction = float(len(full_days) / max(len(reference), 1))
    system_budget = min(desired, int(round(native_mask.size * min(rate, system_fraction))))
    length = native_mask.shape[0]
    attempts = 0
    while int(target.sum()) < system_budget and attempts < max(8, length * 2):
        block = 1
        if len(full_days) > 1:
            diffs = np.diff(full_days)
            block = int(max(1, min(length, np.quantile(diffs[diffs > 0], 0.25) if np.any(diffs > 0) else 1)))
        start = int(rng.randint(0, max(1, length - block + 1)))
        target[start : start + block] = native_mask[start : start + block]
        attempts += 1

    remaining = (np.asarray(native_mask) > 0) & (target == 0)
    need = desired - int(target.sum())
    if need > 0 and remaining.any():
        positions = np.argwhere(remaining)
        rank = _normalized_score(pair_driver)[remaining]
        weights = np.exp(np.clip(4.0 * (rank - 0.5), -20.0, 20.0))
        weights /= weights.sum()
        chosen = rng.choice(len(positions), size=min(need, len(positions)), replace=False, p=weights)
        target[tuple(positions[chosen].T)] = 1.0
    return (target * native_mask).astype(np.float32)


def _target_count(native_mask: np.ndarray, rate: float) -> int:
    """根据原生可观测单元计算目标遮挡数量。"""
    observed = int(native_mask.sum())
    if observed <= 1:
        return 0
    return min(observed - 1, max(1, int(round(observed * float(rate)))))


def random_mask(native_mask: np.ndarray, rate: float, rng: np.random.RandomState) -> np.ndarray:
    """从真实观测单元中精确采样随机点缺失。"""
    target = np.zeros_like(native_mask, dtype=np.float32)
    positions = np.argwhere(native_mask > 0)
    count = _target_count(native_mask, rate)
    if count:
        chosen = rng.choice(len(positions), size=count, replace=False)
        target[tuple(positions[chosen].T)] = 1.0
    return target


def independent_temporal_block_mask(
    native_mask: np.ndarray,
    rate: float,
    rng: np.random.RandomState,
    block_len: int,
) -> np.ndarray:
    """为不同 OD 对独立生成连续时间块，并将实际遮挡量逼近目标缺失率。"""
    length, num_pairs = native_mask.shape
    target = np.zeros_like(native_mask, dtype=np.float32)
    desired = _target_count(native_mask, rate)
    block_len = max(1, min(int(block_len), length))
    attempts = 0
    max_attempts = max(100, desired * 4)
    while int(target.sum()) < desired and attempts < max_attempts:
        pair = int(rng.randint(0, num_pairs))
        start = int(rng.randint(0, max(1, length - block_len + 1)))
        target[start : start + block_len, pair] = native_mask[start : start + block_len, pair]
        attempts += 1
    if int(target.sum()) < desired:
        remaining = np.argwhere((native_mask > 0) & (target == 0))
        count = min(desired - int(target.sum()), len(remaining))
        if count > 0:
            chosen = rng.choice(len(remaining), size=count, replace=False)
            target[tuple(remaining[chosen].T)] = 1.0
    return target.astype(np.float32)


def system_temporal_block_mask(
    native_mask: np.ndarray,
    rate: float,
    rng: np.random.RandomState,
    block_len: int,
) -> np.ndarray:
    """生成平台级连续时间故障，所有 OD 同期缺失。"""
    length, _ = native_mask.shape
    target = np.zeros_like(native_mask, dtype=np.float32)
    desired = _target_count(native_mask, rate)
    block_len = max(1, min(int(block_len), length))
    attempts = 0
    while int(target.sum()) < desired and attempts < length * 4:
        start = int(rng.randint(0, max(1, length - block_len + 1)))
        target[start : start + block_len] = native_mask[start : start + block_len]
        attempts += 1
    return target


def persistent_od_mask(native_mask: np.ndarray, rate: float, rng: np.random.RandomState) -> np.ndarray:
    """随机选择若干 OD 对，并遮挡其窗口内全部真实观测。"""
    _, num_pairs = native_mask.shape
    counts = native_mask.sum(axis=0)
    candidates = np.where(counts > 0)[0]
    rng.shuffle(candidates)
    desired = _target_count(native_mask, rate)
    selected: list[int] = []
    covered = 0
    for pair in candidates:
        if covered >= desired and selected:
            break
        selected.append(int(pair))
        covered += int(counts[pair])
    target = np.zeros_like(native_mask, dtype=np.float32)
    if selected:
        target[:, selected] = native_mask[:, selected]
    return target


def city_level_mask(
    native_mask: np.ndarray,
    origin_idx: np.ndarray,
    destination_idx: np.ndarray,
    rng: np.random.RandomState,
    rate: float,
    city_count: int | None,
) -> np.ndarray:
    """遮挡选中城市的全部流入和流出。"""
    cities = np.unique(np.concatenate([origin_idx, destination_idx]))
    cities = cities.copy()
    rng.shuffle(cities)
    selected: list[int] = []
    if city_count is not None and int(city_count) > 0:
        selected = [int(v) for v in cities[: min(int(city_count), len(cities))]]
    else:
        desired = _target_count(native_mask, rate)
        covered = 0
        for city in cities:
            selected.append(int(city))
            hit = np.isin(origin_idx, selected) | np.isin(destination_idx, selected)
            covered = int(native_mask[:, hit].sum())
            if covered >= desired:
                break
    hit = np.isin(origin_idx, selected) | np.isin(destination_idx, selected)
    target = np.zeros_like(native_mask, dtype=np.float32)
    target[:, hit] = native_mask[:, hit]
    return target


def make_target_mask(
    native_mask: np.ndarray,
    missing_type: str,
    rate: float,
    rng: np.random.RandomState,
    origin_idx: np.ndarray,
    destination_idx: np.ndarray,
    block_len: int = 7,
    city_count: int | None = None,
    driver: np.ndarray | None = None,
    native_reference: np.ndarray | None = None,
    driver_strength: float = 3.0,
) -> np.ndarray:
    """按缺失类型生成 target mask；1 表示模型需要恢复并参与评价。"""
    if missing_type == "random":
        target = random_mask(native_mask, rate, rng)
    elif missing_type in {"independent_temporal_block", "temporal_block"}:
        target = independent_temporal_block_mask(native_mask, rate, rng, block_len)
    elif missing_type == "system_temporal_block":
        target = system_temporal_block_mask(native_mask, rate, rng, block_len)
    elif missing_type in {"persistent_od", "od_pair_block"}:
        target = persistent_od_mask(native_mask, rate, rng)
    elif missing_type in {"city_level", "city_level_block"}:
        target = city_level_mask(native_mask, origin_idx, destination_idx, rng, rate, city_count)
    elif missing_type in {"mar_weather", "mar_calendar", "mar_distance", "mnar_high_flow"}:
        if driver is None:
            raise ValueError(f"{missing_type} 缺失机制需要 driver")
        target = driver_weighted_mask(native_mask, rate, rng, driver, driver_strength)
    elif missing_type == "mnar_low_flow":
        if driver is None:
            raise ValueError("mnar_low_flow 缺失机制需要 flow driver")
        target = driver_weighted_mask(native_mask, rate, rng, -np.asarray(driver), driver_strength)
    elif missing_type == "native_like":
        target = native_like_mask(native_mask, rate, rng, native_reference)
    else:
        raise ValueError(f"未知缺失模式：{missing_type}")
    target = (target * native_mask).astype(np.float32)
    if target.sum() == 0 and native_mask.sum() > 1:
        return random_mask(native_mask, min(max(rate, 0.01), 0.5), rng)
    if target.sum() >= native_mask.sum():
        positions = np.argwhere(target > 0)
        chosen = positions[int(rng.randint(0, len(positions)))]
        target[tuple(chosen)] = 0.0
    return target
