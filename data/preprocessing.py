"""数值列识别、训练期插补、标准化和时间切分。"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd

NON_FEATURE_COLUMNS = {
    "date", "city_id", "origin_id", "destination_id", "city_name", "origin_city",
    "destination_city", "region", "urban_agglomeration", "holiday_type",
}


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    """返回适合作为模型输入的数值列。"""
    excluded = NON_FEATURE_COLUMNS | set(exclude)
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def fill_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """用当前静态表的列中位数填充数值特征缺失。"""
    out = df.copy()
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        median = float(finite.median()) if len(finite) else 0.0
        out[column] = values.fillna(median)
    return out


def date_split(dates: pd.DatetimeIndex, split_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按日期边界或时间比例生成互不重叠的训练、验证和测试索引。"""
    if split_cfg.get("mode", "ratio") == "date":
        train_end = pd.Timestamp(split_cfg["train_end"])
        valid_end = pd.Timestamp(split_cfg["valid_end"])
        if valid_end <= train_end:
            raise ValueError("split.valid_end 必须晚于 split.train_end")
        train = np.where(dates <= train_end)[0]
        valid = np.where((dates > train_end) & (dates <= valid_end))[0]
        test = np.where(dates > valid_end)[0]
    else:
        ratios = split_cfg.get("ratio", [0.7, 0.1, 0.2])
        if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
            raise ValueError("split.ratio 必须包含三个且总和为 1 的比例")
        if any(float(value) <= 0 for value in ratios):
            raise ValueError("split.ratio 的三个比例都必须大于 0")
        n = len(dates)
        n_train = int(n * ratios[0])
        n_valid = int(n * ratios[1])
        train = np.arange(0, n_train)
        valid = np.arange(n_train, n_train + n_valid)
        test = np.arange(n_train + n_valid, n)
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f"时间切分产生空集合：train={len(train)}, valid={len(valid)}, test={len(test)}")
    if not (train[-1] < valid[0] and valid[-1] < test[0]):
        raise RuntimeError("时间切分不是严格按时间顺序排列")
    return train.astype(np.int64), valid.astype(np.int64), test.astype(np.int64)


def standardize_observed(
    values: np.ndarray,
    observed_mask: np.ndarray,
    train_indices: np.ndarray,
    method: str = "standard",
) -> tuple[np.ndarray, float, float]:
    """仅用训练期真实观测单元拟合 standard 或 robust log-flow scaler。"""
    reference = values[train_indices]
    reference_mask = observed_mask[train_indices].astype(bool)
    observed = reference[reference_mask]
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        raise ValueError("训练期不存在真实观测 OD 流量")
    method = str(method).lower()
    if method == "standard":
        mean = float(np.mean(observed))
        std = float(np.std(observed))
    elif method == "robust":
        mean = float(np.median(observed))
        # 1.4826*MAD 在近似正态时与标准差同量纲，但不引入分布假设到评价中。
        std = float(1.4826 * np.median(np.abs(observed - mean)))
    else:
        raise ValueError("data.target_scaler 只能是 standard 或 robust")
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    scaled = np.zeros_like(values, dtype=np.float32)
    valid = observed_mask.astype(bool) & np.isfinite(values)
    scaled[valid] = ((values[valid] - mean) / std).astype(np.float32)
    return scaled, mean, std


def _column_train_medians(reference: np.ndarray) -> np.ndarray:
    """逐特征拟合训练期中位数；训练期全缺失的特征回退为 0。"""
    medians = np.zeros(reference.shape[-1], dtype=np.float64)
    for feature_index in range(reference.shape[-1]):
        column = reference[:, feature_index]
        finite = column[np.isfinite(column)]
        medians[feature_index] = float(np.median(finite)) if finite.size else 0.0
    return medians


def standardize_time_features(
    values: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    仅用训练时间段拟合动态上下文的插补中位数、均值和标准差。

    返回 ``scaled, imputation_median, mean, std``。验证集和测试集只应用训练期参数，
    因而不会通过缺失值填充或标准化引入未来统计量。
    """
    numeric = np.asarray(values, dtype=np.float64).copy()
    numeric[~np.isfinite(numeric)] = np.nan
    reference = numeric[train_indices].reshape(-1, numeric.shape[-1])
    medians = _column_train_medians(reference)
    filled = np.where(np.isfinite(numeric), numeric, medians.reshape(1, 1, -1))
    train_filled = filled[train_indices].reshape(-1, filled.shape[-1])
    mean = np.mean(train_filled, axis=0)
    std = np.std(train_filled, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    scaled = (filled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return (
        np.nan_to_num(scaled).astype(np.float32),
        medians.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def standardize_static(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """标准化区域内部固定的城市或城市对静态特征。"""
    values = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    scaled = (values - mean) / std
    return np.nan_to_num(scaled).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def apply_observed_scaler(
    values: np.ndarray,
    observed_mask: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """应用已在源域拟合的 target scaler；原生缺失位置保持占位 0。"""
    scaled = np.zeros_like(values, dtype=np.float32)
    valid = observed_mask.astype(bool) & np.isfinite(values)
    safe_std = float(std) if np.isfinite(std) and float(std) > 1e-6 else 1.0
    scaled[valid] = ((values[valid] - float(mean)) / safe_std).astype(np.float32)
    return scaled


def apply_feature_scaler(
    values: np.ndarray,
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """应用可序列化的逐特征填补与标准化参数。"""
    numeric = np.asarray(values, dtype=np.float64).copy()
    numeric[~np.isfinite(numeric)] = np.nan
    medians = np.asarray(medians, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if numeric.shape[-1] != len(mean) or len(mean) != len(std) or len(mean) != len(medians):
        raise ValueError("外部 scaler 维度与特征维度不一致")
    filled = np.where(np.isfinite(numeric), numeric, medians.reshape((1,) * (numeric.ndim - 1) + (-1,)))
    safe_std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    shape = (1,) * (numeric.ndim - 1) + (-1,)
    return np.nan_to_num((filled - mean.reshape(shape)) / safe_std.reshape(shape)).astype(np.float32)
