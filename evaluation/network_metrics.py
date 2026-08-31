"""把日–OD 补全结果转成城市流入/流出和走廊恢复指标。"""
from __future__ import annotations

import numpy as np


def completed_flow_matrix(region_data, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 truth、completed prediction 和参与评价的日期；重复窗口目标取预测中位数。"""
    truth = region_data.inverse_target(region_data.x).astype(np.float64)
    completed = truth.copy()
    date_index = np.asarray(arrays["date_index"], dtype=np.int64)
    pair_index = np.asarray(arrays["pair_index"], dtype=np.int64)
    prediction = np.asarray(arrays["y_pred"], dtype=np.float64)
    buckets: dict[tuple[int, int], list[float]] = {}
    for date, pair, value in zip(date_index, pair_index, prediction):
        buckets.setdefault((int(date), int(pair)), []).append(float(value))
    for (date, pair), values in buckets.items():
        completed[date, pair] = float(np.median(values))
    dates = np.asarray(sorted({int(value) for value in date_index}), dtype=np.int64)
    return truth, completed, dates


def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_pred - y_true).sum() / max(np.abs(y_true).sum(), 1e-9))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.argsort(np.argsort(np.asarray(left), kind="mergesort"), kind="mergesort").astype(float)
    b = np.argsort(np.argsort(np.asarray(right), kind="mergesort"), kind="mergesort").astype(float)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def network_recovery_metrics(region_data, arrays: dict[str, np.ndarray], top_fraction: float = 0.2) -> dict[str, float]:
    truth, completed, dates = completed_flow_matrix(region_data, arrays)
    origin = np.asarray(region_data.origin_idx, dtype=np.int64)
    destination = np.asarray(region_data.destination_idx, dtype=np.int64)
    city_count = int(region_data.num_cities)
    true_in = np.zeros((len(dates), city_count), dtype=np.float64)
    pred_in = np.zeros_like(true_in)
    true_out = np.zeros_like(true_in)
    pred_out = np.zeros_like(true_in)
    for row, date in enumerate(dates):
        np.add.at(true_out[row], origin, truth[date])
        np.add.at(pred_out[row], origin, completed[date])
        np.add.at(true_in[row], destination, truth[date])
        np.add.at(pred_in[row], destination, completed[date])
    k = max(1, int(round(region_data.num_pairs * float(top_fraction))))
    recalls: list[float] = []
    for date in dates:
        true_top = set(np.argpartition(truth[date], -k)[-k:].tolist())
        pred_top = set(np.argpartition(completed[date], -k)[-k:].tolist())
        recalls.append(len(true_top & pred_top) / k)
    true_values = truth[dates].reshape(-1)
    pred_values = completed[dates].reshape(-1)
    cpc = 2.0 * np.minimum(np.maximum(true_values, 0), np.maximum(pred_values, 0)).sum()
    cpc /= max(np.maximum(true_values, 0).sum() + np.maximum(pred_values, 0).sum(), 1e-9)
    return {
        "NETWORK_INFLOW_WMAPE": _wmape(true_in, pred_in),
        "NETWORK_OUTFLOW_WMAPE": _wmape(true_out, pred_out),
        "NETWORK_INFLOW_RANK_CORR": _spearman(true_in.reshape(-1), pred_in.reshape(-1)),
        "NETWORK_OUTFLOW_RANK_CORR": _spearman(true_out.reshape(-1), pred_out.reshape(-1)),
        "TOP_CORRIDOR_RECALL": float(np.mean(recalls)),
        "NETWORK_CPC": float(cpc),
        "NETWORK_DATE_COUNT": int(len(dates)),
    }
