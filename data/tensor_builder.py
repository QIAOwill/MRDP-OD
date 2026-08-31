"""把区域原始表转换为 MRDP-OD 使用的时间—OD 张量。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch

from .graph_builder import build_relation_edges
from .entity_splits import held_out_pair_mask, entity_split_manifest
from .preprocessing import (
    date_split,
    fill_numeric,
    numeric_columns,
    standardize_observed,
    standardize_static,
    standardize_time_features,
    apply_observed_scaler,
    apply_feature_scaler,
)


@dataclass
class RegionTensorData:
    """单个城市群的模型输入、掩码、图和标准化信息。"""
 
    region: str
    dates: pd.DatetimeIndex
    pair_frame: pd.DataFrame
    city_frame: pd.DataFrame
    city_ids: np.ndarray
    x: np.ndarray
    native_mask: np.ndarray
    context: np.ndarray
    pair_features: np.ndarray
    city_features: np.ndarray
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    x_mean: float
    x_std: float
    context_imputation_median: np.ndarray
    context_mean: np.ndarray
    context_std: np.ndarray
    pair_imputation_median: np.ndarray
    pair_mean: np.ndarray
    pair_std: np.ndarray
    city_imputation_median: np.ndarray
    city_mean: np.ndarray
    city_std: np.ndarray
    relation_edges: Dict[str, tuple[torch.Tensor, torch.Tensor]]
    origin_idx: np.ndarray
    destination_idx: np.ndarray
    pair_feature_names: list[str]
    context_feature_names: list[str]
    city_feature_names: list[str]
    held_out_pair_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    target_scaler_method: str = "standard"

    def inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        """将标准化 log1p 结果还原到非负原始流量。"""
        log_values = scaled * self.x_std + self.x_mean
        raw = np.expm1(np.clip(log_values, -20.0, 20.0))
        return np.maximum(raw, 0.0)

    @property
    def num_pairs(self) -> int:
        return int(self.x.shape[1])

    @property
    def num_cities(self) -> int:
        return int(len(self.city_ids))


def _complete_date_axis(data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """使用所有动态表的最小—最大日期构造连续日历轴。"""
    all_dates = pd.concat(
        [data["od"]["date"], data["city_dynamic"]["date"], data["pair_weather"]["date"]],
        ignore_index=True,
    )
    start = pd.Timestamp(all_dates.min())
    end = pd.Timestamp(all_dates.max())
    return pd.date_range(start, end, freq="D")


def _build_pair_features(
    city_static: pd.DataFrame,
    pair_static: pd.DataFrame,
    external_medians: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str], np.ndarray]:
    """拼接城市对属性、起终点属性、差值和乘积。"""
    city_cols = numeric_columns(city_static)
    pair_cols = numeric_columns(pair_static)

    if external_medians is None:
        city_static = fill_numeric(city_static, city_cols)
        pair_static = fill_numeric(pair_static, pair_cols)
    else:
        # zero-shot 目标域不能先用自身中位数填补；保留 NaN，待完整 pair
        # feature schema 构造后统一应用 source-only medians。
        city_static = city_static.copy()
        pair_static = pair_static.copy()
        for column in city_cols:
            city_static[column] = pd.to_numeric(city_static[column], errors="coerce")
        for column in pair_cols:
            pair_static[column] = pd.to_numeric(pair_static[column], errors="coerce")

    origin = city_static[["city_id"] + city_cols].rename(
        columns={
            "city_id": "origin_id",
            **{column: f"origin_{column}" for column in city_cols},
        }
    )

    destination = city_static[["city_id"] + city_cols].rename(
        columns={
            "city_id": "destination_id",
            **{column: f"destination_{column}" for column in city_cols},
        }
    )

    frame = (
        pair_static
        .merge(origin, on="origin_id", how="left")
        .merge(destination, on="destination_id", how="left")
    )

    origin_feature_names = [
        f"origin_{column}" for column in city_cols
    ]
    destination_feature_names = [
        f"destination_{column}" for column in city_cols
    ]

    derived_columns: dict[str, pd.Series] = {}
    derived_feature_names: list[str] = []

    for column in city_cols:
        origin_column = f"origin_{column}"
        destination_column = f"destination_{column}"

        difference_name = f"difference_{column}"
        product_name = f"product_{column}"

        derived_columns[difference_name] = (
            frame[origin_column] - frame[destination_column]
        )
        derived_columns[product_name] = (
            frame[origin_column] * frame[destination_column]
        )

        derived_feature_names.extend([
            difference_name,
            product_name,
        ])

    if derived_columns:
        derived_frame = pd.DataFrame(
            derived_columns,
            index=frame.index,
        )
        frame = pd.concat(
            [frame, derived_frame],
            axis=1,
        )

    features = (
        list(pair_cols)
        + origin_feature_names
        + destination_feature_names
        + derived_feature_names
    )

    cleaned_features = (
        frame.loc[:, features]
        .replace([np.inf, -np.inf], np.nan)
    )

    if external_medians is None:
        medians = cleaned_features.median(numeric_only=True).reindex(features).fillna(0.0)
    else:
        values = np.asarray(external_medians, dtype=np.float64)
        if values.size != len(features):
            raise ValueError("pair external medians 与 feature schema 不一致")
        medians = pd.Series(values, index=features)
    cleaned_features = (
        cleaned_features
        .fillna(medians)
        .fillna(0.0)
        .astype(np.float32)
    )

    non_feature_columns = [
        column for column in frame.columns
        if column not in features
    ]

    # 重新拼接为连续内存块，避免 DataFrame 碎片化。
    frame = pd.concat(
        [
            frame.loc[:, non_feature_columns].reset_index(drop=True),
            cleaned_features.reset_index(drop=True),
        ],
        axis=1,
    )

    values = cleaned_features.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    return values, frame, features, medians.to_numpy(dtype=np.float32)


def _build_city_features(
    city_static: pd.DataFrame,
    external_medians: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str], np.ndarray]:
    """构造按 city_id 排序的城市属性矩阵及静态插补参数。"""
    columns = numeric_columns(city_static)
    frame = city_static.sort_values("city_id").reset_index(drop=True).copy()
    cleaned = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if external_medians is None:
        medians = cleaned.median(numeric_only=True).reindex(columns).fillna(0.0)
    else:
        values = np.asarray(external_medians, dtype=np.float64)
        if values.size != len(columns):
            raise ValueError("city external medians 与 feature schema 不一致")
        medians = pd.Series(values, index=columns)
    cleaned = cleaned.fillna(medians).fillna(0.0).astype(np.float32)
    for column in columns:
        frame[column] = cleaned[column].to_numpy(dtype=np.float32)
    values = cleaned.to_numpy(dtype=np.float32, copy=True)
    return values, frame, columns, medians.to_numpy(dtype=np.float32)


def _holiday_dummies(values: pd.Series, prefix: str) -> pd.DataFrame:
    """将节假日类型转为稳定 one-hot。"""
    categories = ["none", "元旦", "春节", "清明节", "劳动节", "端午节", "中秋节", "国庆节"]
    text = values.fillna("none").astype(str)
    categorical = pd.Categorical(text, categories=categories)
    return pd.get_dummies(categorical, prefix=prefix, dtype=np.float32)


def _build_context(
    data: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    pair_frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """构造时间、天气、起点动态和终点动态上下文。"""
    pairs = pair_frame[["origin_id", "destination_id"]].copy().reset_index(drop=True)
    pairs["pair_pos"] = np.arange(len(pairs))
    base = pd.MultiIndex.from_product([dates, pairs["pair_pos"]], names=["date", "pair_pos"]).to_frame(index=False)
    base = base.merge(pairs, on="pair_pos", how="left")
    frame = base.merge(data["pair_weather"], on=["date", "origin_id", "destination_id"], how="left", suffixes=("", "_weather"))

    dynamic = data["city_dynamic"].copy()
    dynamic_columns = [
        "is_weekend", "is_holiday", "month", "day_of_week", "temp_mean",
        "wind_speed", "snow_flag", "dewpoint_temperature", "holiday_type",
    ]
    origin = dynamic[["date", "city_id"] + dynamic_columns].rename(
        columns={"city_id": "origin_id", **{c: f"origin_{c}" for c in dynamic_columns}}
    )
    destination = dynamic[["date", "city_id"] + dynamic_columns].rename(
        columns={"city_id": "destination_id", **{c: f"destination_{c}" for c in dynamic_columns}}
    )
    frame = frame.merge(origin, on=["date", "origin_id"], how="left")
    frame = frame.merge(destination, on=["date", "destination_id"], how="left")

    numeric = [c for c in ["temp_diff", "wind_max", "snow_any"] if c in frame]
    numeric.extend(
        c for c in frame.columns
        if (c.startswith("origin_") or c.startswith("destination_"))
        and not c.endswith("holiday_type")
        and c not in {"origin_id", "destination_id", "origin_city", "destination_city"}
        and pd.api.types.is_numeric_dtype(frame[c])
    )
    numeric_frame = (
        frame[numeric]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )
    oh = _holiday_dummies(frame.get("origin_holiday_type", pd.Series("none", index=frame.index)), "origin_holiday")
    dh = _holiday_dummies(frame.get("destination_holiday_type", pd.Series("none", index=frame.index)), "destination_holiday")
    features = pd.concat([numeric_frame, oh, dh], axis=1)
    names = list(features.columns)
    values = features.to_numpy(dtype=np.float32, copy=True).reshape(len(dates), len(pairs), -1)
    return values, names


def build_region_tensor(
    region: str,
    data: Dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    scaler_override: dict[str, Any] | None = None,
) -> RegionTensorData:
    """构建单区域完整张量对象。"""
    city_limit = cfg.get("data", {}).get("city_limit")
    if city_limit is not None:
        selected = set(sorted(data["city_static"]["city_id"].unique())[: int(city_limit)])
        local: Dict[str, pd.DataFrame] = {}
        local["city_static"] = data["city_static"][data["city_static"]["city_id"].isin(selected)].copy()
        local["city_dynamic"] = data["city_dynamic"][data["city_dynamic"]["city_id"].isin(selected)].copy()
        for name in ("od", "pair_static", "pair_weather"):
            local[name] = data[name][
                data[name]["origin_id"].isin(selected) & data[name]["destination_id"].isin(selected)
            ].copy()
        data = local
    pair_frame = data["pair_static"].sort_values(["origin_id", "destination_id"]).reset_index(drop=True)
    dates = _complete_date_axis(data)
    pair_index = {(int(row.origin_id), int(row.destination_id)): i for i, row in pair_frame.iterrows()}
    date_index = {pd.Timestamp(date): i for i, date in enumerate(dates)}

    raw = np.full((len(dates), len(pair_frame)), np.nan, dtype=np.float32)
    native_mask = np.zeros_like(raw, dtype=np.float32)
    for row in data["od"][["date", "origin_id", "destination_id", "od_flow"]].itertuples(index=False):
        value = float(row.od_flow)
        if np.isnan(value):
            # 显式 NaN 与记录缺席具有相同语义：原生未观测。
            continue
        if not np.isfinite(value) or value < 0:
            raise ValueError("OD 流量只能是非负有限值或 NaN")
        ti = date_index[pd.Timestamp(row.date)]
        ei = pair_index[(int(row.origin_id), int(row.destination_id))]
        raw[ti, ei] = value
        native_mask[ti, ei] = 1.0
    log_raw = np.full_like(raw, np.nan)
    observed = native_mask.astype(bool)
    log_raw[observed] = np.log1p(raw[observed])

    train_idx, valid_idx, test_idx = date_split(dates, cfg["data"]["split"])
    preliminary_city_ids = np.sort(data["city_static"]["city_id"].unique()).astype(np.int64)
    preliminary_city_map = {int(city_id): i for i, city_id in enumerate(preliminary_city_ids)}
    preliminary_origin = np.asarray(
        [preliminary_city_map[int(value)] for value in pair_frame["origin_id"]], dtype=np.int64
    )
    preliminary_destination = np.asarray(
        [preliminary_city_map[int(value)] for value in pair_frame["destination_id"]], dtype=np.int64
    )
    held_out = held_out_pair_mask(
        preliminary_origin,
        preliminary_destination,
        len(preliminary_city_ids),
        cfg.get("data", {}).get("entity_split"),
    )
    scaler_fit_mask = native_mask.copy()
    if held_out.any():
        scaler_fit_mask[:, held_out] = 0.0
    if scaler_override is None:
        target_scaler_method = str(cfg.get("data", {}).get("target_scaler", "standard"))
        x, x_mean, x_std = standardize_observed(
            log_raw, scaler_fit_mask, train_idx, method=target_scaler_method
        )
        # scaler 用可见实体拟合，但 transform 仍覆盖所有有真值单元，供 test entity target 使用。
        x = apply_observed_scaler(log_raw, native_mask, x_mean, x_std)
    else:
        target_scaler = scaler_override["target"]
        target_scaler_method = str(target_scaler.get("scaler", "standard"))
        x_mean = float(target_scaler["mean"])
        x_std = float(target_scaler["std"])
        x = apply_observed_scaler(log_raw, native_mask, x_mean, x_std)

    pair_override = None if scaler_override is None else scaler_override["pair_static"]
    pair_values, enriched_pairs, pair_names, pair_imputation_median = _build_pair_features(
        data["city_static"], pair_frame,
        None if pair_override is None else np.asarray(pair_override["imputation_median"]),
    )
    if pair_override is None:
        pair_values, pair_mean, pair_std = standardize_static(pair_values)
    else:
        if list(pair_override["feature_names"]) != pair_names:
            raise ValueError("pair feature schema 与外部 scaler 不一致")
        pair_mean = np.asarray(pair_override["mean"], dtype=np.float32)
        pair_std = np.asarray(pair_override["std"], dtype=np.float32)
        pair_values = apply_feature_scaler(pair_values, pair_imputation_median, pair_mean, pair_std)
    city_override = None if scaler_override is None else scaler_override["city_static"]
    city_values, city_frame, city_names, city_imputation_median = _build_city_features(
        data["city_static"],
        None if city_override is None else np.asarray(city_override["imputation_median"]),
    )
    if city_override is None:
        city_values, city_mean, city_std = standardize_static(city_values)
    else:
        if list(city_override["feature_names"]) != city_names:
            raise ValueError("city feature schema 与外部 scaler 不一致")
        city_mean = np.asarray(city_override["mean"], dtype=np.float32)
        city_std = np.asarray(city_override["std"], dtype=np.float32)
        city_values = apply_feature_scaler(city_values, city_imputation_median, city_mean, city_std)
    context, context_names = _build_context(data, dates, pair_frame)
    context_override = None if scaler_override is None else scaler_override["context"]
    if context_override is None:
        context, context_imputation_median, context_mean, context_std = standardize_time_features(
            context, train_idx
        )
    else:
        if list(context_override["feature_names"]) != context_names:
            raise ValueError("context feature schema 与外部 scaler 不一致")
        context_imputation_median = np.asarray(context_override["imputation_median"], dtype=np.float32)
        context_mean = np.asarray(context_override["mean"], dtype=np.float32)
        context_std = np.asarray(context_override["std"], dtype=np.float32)
        context = apply_feature_scaler(
            context, context_imputation_median, context_mean, context_std
        )

    # 特征缺失敏感性：在完成训练期 scaler 后把预注册比例的特征置为标准化均值 0。
    # 同一配置在 train/valid/test 使用同一列集合，不会借测试信息选择列。
    dropout = cfg.get("data", {}).get("feature_dropout") or {}
    dropout_seed = int(dropout.get("seed", 52024))
    rng = np.random.RandomState(dropout_seed)
    for key, values in (
        ("context", context), ("pair_static", pair_values), ("city_static", city_values)
    ):
        fraction = float(dropout.get(key, 0.0))
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"data.feature_dropout.{key} 必须位于 [0,1)")
        count = int(round(values.shape[-1] * fraction))
        if count > 0:
            columns = rng.choice(values.shape[-1], size=count, replace=False)
            values[..., columns] = 0.0

    city_ids = city_frame["city_id"].to_numpy(np.int64)
    city_map = {int(city_id): i for i, city_id in enumerate(city_ids)}
    origin_idx = np.array([city_map[int(v)] for v in pair_frame["origin_id"]], dtype=np.int64)
    destination_idx = np.array([city_map[int(v)] for v in pair_frame["destination_id"]], dtype=np.int64)
    relation_edges = build_relation_edges(enriched_pairs, cfg["graph"])

    return RegionTensorData(
        region=region,
        dates=dates,
        pair_frame=pair_frame,
        city_frame=city_frame,
        city_ids=city_ids,
        x=x,
        native_mask=native_mask,
        context=context,
        pair_features=pair_values,
        city_features=city_values,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        x_mean=x_mean,
        x_std=x_std,
        context_imputation_median=context_imputation_median,
        context_mean=context_mean,
        context_std=context_std,
        pair_imputation_median=pair_imputation_median,
        pair_mean=pair_mean,
        pair_std=pair_std,
        city_imputation_median=city_imputation_median,
        city_mean=city_mean,
        city_std=city_std,
        relation_edges=relation_edges,
        origin_idx=origin_idx,
        destination_idx=destination_idx,
        pair_feature_names=pair_names,
        context_feature_names=context_names,
        city_feature_names=city_names,
        held_out_pair_mask=held_out,
        target_scaler_method=target_scaler_method,
    )


def data_manifest(region_data: RegionTensorData) -> dict[str, Any]:
    """返回可写入运行目录的数据规模与切分摘要。"""
    return {
        "region": region_data.region,
        "date_start": str(region_data.dates.min().date()),
        "date_end": str(region_data.dates.max().date()),
        "num_dates": len(region_data.dates),
        "num_cities": region_data.num_cities,
        "num_pairs": region_data.num_pairs,
        "native_observed_cells": int(region_data.native_mask.sum()),
        "native_missing_cells": int((1.0 - region_data.native_mask).sum()),
        "native_observation_rate": float(region_data.native_mask.mean()),
        "train_dates": len(region_data.train_idx),
        "valid_dates": len(region_data.valid_idx),
        "test_dates": len(region_data.test_idx),
        "pair_feature_dim": int(region_data.pair_features.shape[-1]),
        "city_feature_dim": int(region_data.city_features.shape[-1]),
        "context_dim": int(region_data.context.shape[-1]),
        "relations": list(region_data.relation_edges),
        "target_mean": region_data.x_mean,
        "target_std": region_data.x_std,
        "entity_split": entity_split_manifest(
            region_data.held_out_pair_mask,
            None,
        ),
    }

def scaler_metadata(region_data: RegionTensorData) -> dict[str, Any]:
    """返回可复现实验所需的插补器和标准化器参数。"""
    return {
        "target": {
            "transform": "log1p",
            "scaler": region_data.target_scaler_method,
            "mean": float(region_data.x_mean),
            "std": float(region_data.x_std),
            "fit_scope": "train_native_observed_only",
            "inverse_nonnegative_clip": True,
        },
        "context": {
            "feature_names": region_data.context_feature_names,
            "imputation": "train_median",
            "imputation_median": region_data.context_imputation_median.tolist(),
            "mean": region_data.context_mean.tolist(),
            "std": region_data.context_std.tolist(),
            "fit_scope": "train_dates_only",
        },
        "pair_static": {
            "feature_names": region_data.pair_feature_names,
            "imputation": "regional_static_median",
            "imputation_median": region_data.pair_imputation_median.tolist(),
            "mean": region_data.pair_mean.tolist(),
            "std": region_data.pair_std.tolist(),
        },
        "city_static": {
            "feature_names": region_data.city_feature_names,
            "imputation": "regional_static_median",
            "imputation_median": region_data.city_imputation_median.tolist(),
            "mean": region_data.city_mean.tolist(),
            "std": region_data.city_std.tolist(),
        },
    }
