"""固定的逐 OD ridge-AR 下游预测器，用于隔离补全输入的影响。"""
from __future__ import annotations

import numpy as np

from .network_metrics import completed_flow_matrix


def _fit_pairwise_ar(values: np.ndarray, native: np.ndarray, train_idx: np.ndarray, lags: int, ridge: float) -> np.ndarray:
    pair_count = values.shape[1]
    coefficients = np.zeros((pair_count, lags + 1), dtype=np.float64)
    for pair in range(pair_count):
        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        allowed = set(int(v) for v in train_idx)
        for date in train_idx:
            date = int(date)
            history = np.arange(date - lags, date)
            if date - lags < 0 or any(int(v) not in allowed for v in history):
                continue
            if native[date, pair] <= 0 or np.any(native[history, pair] <= 0):
                continue
            x_rows.append(np.r_[1.0, values[history, pair]])
            y_rows.append(float(values[date, pair]))
        if not x_rows:
            coefficients[pair, 0] = float(np.nanmean(values[train_idx, pair]))
            continue
        design = np.vstack(x_rows)
        target = np.asarray(y_rows)
        penalty = np.eye(lags + 1) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients[pair] = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return coefficients


def downstream_forecast_metrics(
    region_data,
    arrays: dict[str, np.ndarray],
    lags: int = 7,
    ridge: float = 1e-3,
) -> dict[str, float]:
    truth, completed, _ = completed_flow_matrix(region_data, arrays)
    native = np.asarray(region_data.native_mask, dtype=np.float64)
    train_mean = np.divide(
        (truth[region_data.train_idx] * native[region_data.train_idx]).sum(axis=0),
        native[region_data.train_idx].sum(axis=0),
        out=np.zeros(region_data.num_pairs, dtype=np.float64),
        where=native[region_data.train_idx].sum(axis=0) > 0,
    )
    completed[native <= 0] = np.broadcast_to(train_mean, completed.shape)[native <= 0]
    coefficients = _fit_pairwise_ar(truth, native, region_data.train_idx, int(lags), float(ridge))
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for date in region_data.test_idx:
        date = int(date)
        if date - lags < 0:
            continue
        history = completed[date - lags : date].T
        design = np.concatenate([np.ones((region_data.num_pairs, 1)), history], axis=1)
        prediction = np.maximum(np.sum(coefficients * design, axis=1), 0.0)
        mask = native[date] > 0
        predictions.append(prediction[mask])
        targets.append(truth[date, mask])
    if not targets:
        return {"FORECAST_WMAPE": float("nan"), "FORECAST_RMSE": float("nan"), "FORECAST_COUNT": 0}
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {
        "FORECAST_WMAPE": float(np.abs(y_pred - y_true).sum() / max(np.abs(y_true).sum(), 1e-9)),
        "FORECAST_RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "FORECAST_COUNT": int(y_true.size),
    }
