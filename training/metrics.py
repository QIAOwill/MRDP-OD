"""点预测、概率预测和分组评价指标。"""
from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """在已经筛选的目标向量上计算全局指标。"""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {name: float("nan") for name in ["MAE", "RMSE", "MAPE", "WMAPE", "SMAPE", "TOP20_WMAPE", "CPC"]}
    error = y_pred - y_true
    absolute = np.abs(error)
    mae = absolute.mean()
    rmse = np.sqrt(np.mean(error ** 2))
    nonzero = np.abs(y_true) > 1e-6
    mape = np.mean(absolute[nonzero] / np.abs(y_true[nonzero])) * 100.0 if np.any(nonzero) else float("nan")
    wmape = absolute.sum() / max(np.abs(y_true).sum(), 1e-9)
    smape = np.mean(2.0 * absolute / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-9))
    threshold = np.quantile(y_true, 0.8) if y_true.size > 1 else y_true[0]
    top = y_true >= threshold
    top_wmape = absolute[top].sum() / max(np.abs(y_true[top]).sum(), 1e-9)
    cpc = 2.0 * np.minimum(np.maximum(y_true, 0.0), np.maximum(y_pred, 0.0)).sum()
    cpc /= max(np.maximum(y_true, 0.0).sum() + np.maximum(y_pred, 0.0).sum(), 1e-9)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "WMAPE": float(wmape),
        "SMAPE": float(smape),
        "TOP20_WMAPE": float(top_wmape),
        "CPC": float(cpc),
    }


def probabilistic_metrics(y_true: np.ndarray, sample_matrix: np.ndarray) -> dict[str, float]:
    """
    根据 [S,N] 经验样本计算 CRPS、90% 区间覆盖率、区间宽度和误差—不确定性相关性。
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    samples = np.asarray(sample_matrix, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != y_true.size or samples.shape[0] < 2:
        names = ["CRPS", "NCRPS", "WIS", "CALIBRATION_ERROR", "UNCERTAINTY_ERROR_CORR", "UNCERTAINTY_ERROR_SPEARMAN", "AURC"]
        names += [f"{prefix}{level}" for level in (50, 80, 90, 95) for prefix in ("PICP", "MPIW", "INTERVAL_SCORE")]
        return {name: float("nan") for name in names}
    first = np.mean(np.abs(samples - y_true[None, :]), axis=0)
    sorted_samples = np.sort(samples, axis=0)
    s = sorted_samples.shape[0]
    coefficients = (2.0 * np.arange(1, s + 1) - s - 1.0).reshape(s, 1)
    second = np.sum(coefficients * sorted_samples, axis=0) / (s ** 2)
    crps = np.mean(first - second)
    uncertainty = np.std(samples, axis=0)
    point_error = np.abs(np.median(samples, axis=0) - y_true)
    if np.std(uncertainty) < 1e-12 or np.std(point_error) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(uncertainty, point_error)[0, 1])
    result = {
        "CRPS": float(crps),
        "NCRPS": float(crps / max(np.mean(np.abs(y_true)), 1e-9)),
        "UNCERTAINTY_ERROR_CORR": corr,
    }
    coverage_errors: list[float] = []
    interval_scores: list[tuple[float, np.ndarray]] = []
    for level in (50, 80, 90, 95):
        alpha = 1.0 - level / 100.0
        lower = np.quantile(samples, alpha / 2.0, axis=0)
        upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        width = float(np.mean(upper - lower))
        interval = (
            upper - lower
            + (2.0 / alpha) * (lower - y_true) * (y_true < lower)
            + (2.0 / alpha) * (y_true - upper) * (y_true > upper)
        )
        result[f"PICP{level}"] = coverage
        result[f"MPIW{level}"] = width
        result[f"INTERVAL_SCORE{level}"] = float(np.mean(interval))
        coverage_errors.append(abs(coverage - level / 100.0))
        interval_scores.append((alpha, interval))
    median_error = np.abs(np.median(samples, axis=0) - y_true)
    numerator = 0.5 * median_error
    for alpha, interval in interval_scores:
        numerator = numerator + (alpha / 2.0) * interval
    result["WIS"] = float(np.mean(numerator / (len(interval_scores) + 0.5)))
    result["CALIBRATION_ERROR"] = float(np.mean(coverage_errors))
    result["UNCERTAINTY_ERROR_SPEARMAN"] = _spearman(uncertainty, point_error)
    order = np.argsort(uncertainty, kind="mergesort")
    cumulative_risk = np.cumsum(point_error[order]) / np.arange(1, len(order) + 1)
    result["AURC"] = float(np.mean(cumulative_risk))
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """不依赖 scipy 的平均秩实现。"""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_ranks(np.asarray(left).reshape(-1))
    right_rank = _average_ranks(np.asarray(right).reshape(-1))
    if left_rank.size < 2 or np.std(left_rank) < 1e-12 or np.std(right_rank) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def consecutive_target_lengths(mask: np.ndarray) -> np.ndarray:
    """为 [L,E] 中每个 target 单元标注其所在连续缺失段长度。"""
    mask = np.asarray(mask).astype(bool)
    lengths = np.zeros_like(mask, dtype=np.int16)
    for pair in range(mask.shape[1]):
        start = None
        for time in range(mask.shape[0] + 1):
            active = time < mask.shape[0] and mask[time, pair]
            if active and start is None:
                start = time
            elif not active and start is not None:
                lengths[start:time, pair] = time - start
                start = None
    return lengths
