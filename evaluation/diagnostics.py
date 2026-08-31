"""生成失败边界、网络案例和结果可复现性所需的原始诊断文件。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def failure_subgroup_metrics(diagnostic: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    按流量、距离、观测支持度和连续缺失长度汇总失败边界。

    参数
    ----
    diagnostic:
        ``target_diagnostics.csv`` 对应的数据；每一行是一处被遮蔽目标。

    返回
    ----
    subgroup_table:
        长表，每行给出一个维度内一个分组的 MAE、WMAPE 和失败率。
    definition:
        失败事件和分组边界的机器可读定义。
    """
    required = {"y_true", "y_pred", "absolute_error", "gap_length", "distance_line"}
    missing = required - set(diagnostic)
    if missing:
        raise KeyError(f"失败诊断缺少字段：{sorted(missing)}")
    data = diagnostic.copy()
    if data.empty:
        raise ValueError("失败诊断不能使用空数据")
    truth_scale = max(float(np.mean(np.abs(data["y_true"]))), 1e-9)
    data["normalized_absolute_error"] = data["absolute_error"] / truth_scale
    failure_threshold = float(data["normalized_absolute_error"].quantile(0.90))
    data["failure"] = data["normalized_absolute_error"] >= failure_threshold
    global_wmape = float(data["absolute_error"].sum() / max(data["y_true"].abs().sum(), 1e-9))

    dimensions: list[tuple[str, pd.Series]] = []
    for name, column in (
        ("flow_quantile", "y_true"),
        ("distance_quantile", "distance_line"),
        ("support_quantile", "support_mean"),
    ):
        if column not in data or data[column].nunique(dropna=True) < 2:
            continue
        labels = pd.qcut(
            data[column], q=min(4, int(data[column].nunique(dropna=True))),
            labels=False, duplicates="drop",
        )
        dimensions.append((name, labels.map(lambda value: f"Q{int(value) + 1}" if pd.notna(value) else "missing")))
    gap_labels = pd.cut(
        data["gap_length"], [0, 3, 7, 14, np.inf],
        labels=["1-3", "4-7", "8-14", ">14"], include_lowest=True,
    ).astype(str)
    dimensions.append(("gap_length", gap_labels))

    rows: list[dict[str, Any]] = []
    for dimension, labels in dimensions:
        for group, part in data.assign(_group=labels).groupby("_group", observed=False):
            if part.empty or str(group) in {"nan", "missing"}:
                continue
            denominator = max(float(part["y_true"].abs().sum()), 1e-9)
            group_wmape = float(part["absolute_error"].sum() / denominator)
            rows.append({
                "dimension": dimension,
                "group": str(group),
                "count": int(len(part)),
                "MAE": float(part["absolute_error"].mean()),
                "WMAPE": group_wmape,
                "WMAPE_relative_to_run": float(group_wmape / max(global_wmape, 1e-9) - 1.0),
                "failure_rate": float(part["failure"].mean()),
                "normalized_absolute_error_mean": float(part["normalized_absolute_error"].mean()),
                "truth_mean": float(part["y_true"].mean()),
                "distance_mean": float(part["distance_line"].mean()),
                "support_mean": float(part["support_mean"].mean()) if "support_mean" in part else np.nan,
                "gap_length_mean": float(part["gap_length"].mean()),
            })
    definition = {
        "failure_event": "normalized_absolute_error >= within-run 90th percentile",
        "normalized_absolute_error": "absolute_error / mean(abs(y_true)) within the run",
        "failure_threshold": failure_threshold,
        "global_wmape": global_wmape,
        "quantile_groups": "Q1 (lowest) through Q4 (highest), computed within each run",
        "gap_groups": ["1-3", "4-7", "8-14", ">14"],
    }
    return pd.DataFrame(rows), definition


def network_case_snapshot(
    region_data,
    diagnostic: pd.DataFrame,
    selection_rule: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    从预注册规则选择一个日期，保存网络可视化所需的完整 OD 对快照。

    ``max_target_count_then_earliest`` 只依赖固定测试掩码，不读取任何模型误差，
    因而所有模型会选中同一个案例，避免事后挑选有利样本。
    """
    if selection_rule != "max_target_count_then_earliest":
        raise ValueError(f"未知网络案例选择规则：{selection_rule}")
    required = {"date_index", "pair_index", "y_pred"}
    missing = required - set(diagnostic)
    if missing:
        raise KeyError(f"网络案例缺少字段：{sorted(missing)}")
    counts = diagnostic.groupby("date_index")["pair_index"].nunique().sort_index()
    if counts.empty:
        raise ValueError("网络案例没有可用目标日期")
    maximum = int(counts.max())
    selected_date_index = int(counts[counts == maximum].index.min())
    selected = diagnostic[diagnostic["date_index"] == selected_date_index]
    predictions = selected.groupby("pair_index")["y_pred"].median()

    pair_frame = region_data.pair_frame.copy().reset_index(drop=True)
    pair_frame.insert(0, "pair_index", np.arange(region_data.num_pairs, dtype=np.int32))
    truth_matrix = region_data.inverse_target(region_data.x)
    truth = truth_matrix[selected_date_index].astype(float)
    native = region_data.native_mask[selected_date_index].astype(bool)
    target = np.zeros(region_data.num_pairs, dtype=bool)
    target[predictions.index.to_numpy(dtype=int)] = True
    masked = truth.copy()
    masked[~native | target] = np.nan
    completed = masked.copy()
    for pair_index, value in predictions.items():
        completed[int(pair_index)] = float(value)
    pair_frame["truth_flow"] = truth
    pair_frame["native_observed"] = native.astype(np.uint8)
    pair_frame["targeted"] = target.astype(np.uint8)
    pair_frame["masked_flow"] = masked
    pair_frame["completed_flow"] = completed
    pair_frame["model_prediction"] = [predictions.get(index, np.nan) for index in range(region_data.num_pairs)]

    city_frame = region_data.city_frame.copy().reset_index(drop=True)
    manifest = {
        "region": str(region_data.region),
        "selection_rule": selection_rule,
        "selected_date_index": selected_date_index,
        "selected_date": str(region_data.dates[selected_date_index].date()),
        "target_pair_count": maximum,
        "selection_uses_model_error": False,
        "pair_table": "network_case_pairs.tsv",
        "city_table": "network_case_cities.tsv",
    }
    return pair_frame, city_frame, manifest
