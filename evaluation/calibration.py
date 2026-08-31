"""只用验证集拟合的多级区间缩放与测试集校准指标。"""
from __future__ import annotations

from typing import Iterable
import numpy as np


LEVELS = (50, 80, 90, 95)


def fit_quantile_scaling(
    y_true: np.ndarray,
    samples: np.ndarray,
    levels: Iterable[int] = LEVELS,
) -> dict[str, float]:
    """围绕样本中位数搜索区间缩放因子，使验证覆盖率接近 nominal coverage。"""
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != truth.size:
        raise ValueError("samples 必须为 [S,N] 且与 y_true 对齐")
    median = np.median(values, axis=0)
    result: dict[str, float] = {}
    for level in levels:
        alpha = 1.0 - int(level) / 100.0
        lower = np.quantile(values, alpha / 2.0, axis=0)
        upper = np.quantile(values, 1.0 - alpha / 2.0, axis=0)
        left_width = np.maximum(median - lower, 1e-9)
        right_width = np.maximum(upper - median, 1e-9)
        low, high = 0.05, 20.0
        target = int(level) / 100.0
        for _ in range(60):
            factor = 0.5 * (low + high)
            calibrated_lower = median - factor * left_width
            calibrated_upper = median + factor * right_width
            coverage = np.mean((truth >= calibrated_lower) & (truth <= calibrated_upper))
            if coverage < target:
                low = factor
            else:
                high = factor
        result[str(int(level))] = float(0.5 * (low + high))
    return result


def calibrated_interval_metrics(
    y_true: np.ndarray,
    samples: np.ndarray,
    factors: dict[str, float],
    levels: Iterable[int] = LEVELS,
) -> dict[str, float]:
    """应用 validation factors；只改变区间，不改变点预测。"""
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    values = np.asarray(samples, dtype=np.float64)
    median = np.median(values, axis=0)
    result: dict[str, float] = {}
    errors: list[float] = []
    weighted_scores: list[tuple[float, np.ndarray]] = []
    for level in levels:
        alpha = 1.0 - int(level) / 100.0
        lower = np.quantile(values, alpha / 2.0, axis=0)
        upper = np.quantile(values, 1.0 - alpha / 2.0, axis=0)
        factor = float(factors[str(int(level))])
        lower = median - factor * np.maximum(median - lower, 0.0)
        upper = median + factor * np.maximum(upper - median, 0.0)
        coverage = float(np.mean((truth >= lower) & (truth <= upper)))
        interval = (
            upper - lower
            + (2.0 / alpha) * (lower - truth) * (truth < lower)
            + (2.0 / alpha) * (truth - upper) * (truth > upper)
        )
        result[f"CAL_PICP{level}"] = coverage
        result[f"CAL_MPIW{level}"] = float(np.mean(upper - lower))
        result[f"CAL_INTERVAL_SCORE{level}"] = float(np.mean(interval))
        errors.append(abs(coverage - int(level) / 100.0))
        weighted_scores.append((alpha, interval))
    wis = 0.5 * np.abs(median - truth)
    for alpha, interval in weighted_scores:
        wis = wis + (alpha / 2.0) * interval
    result["CAL_WIS"] = float(np.mean(wis / (len(weighted_scores) + 0.5)))
    result["CAL_CALIBRATION_ERROR"] = float(np.mean(errors))
    return result
