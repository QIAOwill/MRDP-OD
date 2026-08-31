"""统一模型验证、测试、概率样本保存和可选诊断。"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import consecutive_target_lengths, probabilistic_metrics, regression_metrics


def observation_support_features(
    cond_mask: torch.Tensor,
    origin_idx: torch.Tensor,
    destination_idx: torch.Tensor,
    num_cities: int,
    scales: tuple[int, int, int] = (3, 7, 15),
) -> torch.Tensor:
    """为所有模型统一计算三个时间、OD 对、起点和终点观测支持度。"""
    b, length, pairs = cond_mask.shape
    series = cond_mask.permute(0, 2, 1).reshape(b * pairs, 1, length)
    temporal: list[torch.Tensor] = []
    for scale in scales:
        kernel = torch.ones(1, 1, scale, device=cond_mask.device, dtype=cond_mask.dtype)
        left, right = (scale - 1) // 2, scale // 2
        denominator = F.conv1d(F.pad(series, (left, right)), kernel) / float(scale)
        temporal.append(denominator.reshape(b, pairs, length).permute(0, 2, 1).clamp(0.0, 1.0))
    pair_support = cond_mask.mean(dim=1, keepdim=True).expand_as(cond_mask)

    def endpoint_support(indices: torch.Tensor) -> torch.Tensor:
        expanded = indices.view(1, 1, pairs).expand(b, length, pairs)
        totals = torch.zeros(b, length, num_cities, device=cond_mask.device, dtype=cond_mask.dtype)
        totals.scatter_add_(2, expanded, cond_mask)
        counts = torch.bincount(indices, minlength=num_cities).to(cond_mask.dtype).clamp_min(1.0)
        return torch.gather(totals / counts.view(1, 1, num_cities), 2, expanded)

    origin_support = endpoint_support(origin_idx)
    destination_support = endpoint_support(destination_idx)
    return torch.stack(
        [*temporal, pair_support, origin_support, destination_support], dim=-1,
    )


@contextmanager
def _evaluation_rng(device: torch.device, random_seed: int | None):
    """在不改变后续训练随机状态的前提下固定一次验证或测试采样。"""
    if random_seed is None:
        yield
        return
    devices: list[int] = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(random_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(random_seed))
        yield


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _impute_for_evaluation(model, batch: dict[str, Any], n_samples: int) -> dict[str, Any]:
    """支持的概率模型逐样本生成后立即卸载到 CPU，避免在 GPU 堆积 [B,S,L,E]。"""
    kwargs: dict[str, Any] = {"n_samples": n_samples}
    if bool(getattr(model, "supports_sample_cpu_offload", False)):
        kwargs["sample_output_device"] = "cpu"
    return model.impute(batch, **kwargs)


def _autocast_context(device: torch.device, enabled: bool, dtype_name: str):
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def _evaluate_loss_impl(model, loader: DataLoader, device: torch.device, amp_enabled: bool, amp_dtype: str, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        with _autocast_context(device, amp_enabled, amp_dtype):
            losses = model.loss(batch)
        batch_size = int(batch["observed_data"].shape[0])
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("验证集没有处理任何 batch")
    return {key: value / count for key, value in sums.items()}


def evaluate_loss(
    model,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: str,
    max_batches: int | None = None,
    random_seed: int | None = None,
) -> dict[str, float]:
    with _evaluation_rng(device, random_seed):
        return _evaluate_loss_impl(model, loader, device, amp_enabled, amp_dtype, max_batches)


@torch.no_grad()
def _evaluate_point_metrics_impl(
    model,
    loader: DataLoader,
    region_data,
    device: torch.device,
    n_samples: int,
    amp_enabled: bool,
    amp_dtype: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    """仅计算一次验证点指标，不写入预测大数组。"""
    model.eval()
    model.use_evaluation_weights() if hasattr(model, "use_evaluation_weights") else None
    truth: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    try:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch(batch, device)
            with _autocast_context(device, amp_enabled, amp_dtype):
                output = _impute_for_evaluation(model, batch, n_samples)
            samples = output["samples"].float().cpu().numpy()
            y = region_data.inverse_target(batch["observed_data"].float().cpu().numpy())
            pred = np.median(region_data.inverse_target(samples), axis=1)
            mask = batch["target_mask"].bool().cpu().numpy()
            for item in range(mask.shape[0]):
                truth.append(y[item][mask[item]])
                prediction.append(pred[item][mask[item]])
            del output, samples, y, pred, mask, batch
    finally:
        model.restore_training_weights() if hasattr(model, "restore_training_weights") else None
    if not truth:
        raise RuntimeError("验证集没有 target entries")
    return regression_metrics(np.concatenate(truth), np.concatenate(prediction))


def evaluate_point_metrics(
    model,
    loader: DataLoader,
    region_data,
    device: torch.device,
    n_samples: int,
    amp_enabled: bool,
    amp_dtype: str,
    max_batches: int | None = None,
    random_seed: int | None = None,
) -> dict[str, float]:
    with _evaluation_rng(device, random_seed):
        return _evaluate_point_metrics_impl(
            model,
            loader,
            region_data,
            device,
            n_samples,
            amp_enabled,
            amp_dtype,
            max_batches,
        )


def _target_aligned_tensor(value: Any, batch_shape: tuple[int, int, int]) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    if tuple(value.shape[:3]) == batch_shape:
        return value
    return None


@torch.no_grad()
def _evaluate_model_impl(
    model,
    loader: DataLoader,
    region_data,
    device: torch.device,
    n_samples: int,
    amp_enabled: bool,
    amp_dtype: str,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray], pd.DataFrame]:
    """在 target entries 上计算统一点指标和模型能力允许的概率指标。"""
    model.eval()
    model.use_evaluation_weights() if hasattr(model, "use_evaluation_weights") else None
    truth_list: list[np.ndarray] = []
    prediction_list: list[np.ndarray] = []
    sample_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    gap_list: list[np.ndarray] = []
    date_index_list: list[np.ndarray] = []
    diagnostics_lists: dict[str, list[np.ndarray]] = {}
    native_prediction_list: list[np.ndarray] = []
    native_std_list: list[np.ndarray] = []
    native_q05_list: list[np.ndarray] = []
    native_q50_list: list[np.ndarray] = []
    native_q95_list: list[np.ndarray] = []
    native_date_list: list[np.ndarray] = []
    native_pair_list: list[np.ndarray] = []
    actual_rates: list[float] = []
    configured_rates: list[float] = []
    target_cell_count = 0.0
    native_observed_cell_count = 0.0
    total_cell_count = 0.0
    distances = region_data.pair_frame.get("distance_line", pd.Series(np.arange(region_data.num_pairs))).to_numpy(float)
    origin_indices = torch.as_tensor(region_data.origin_idx, device=device, dtype=torch.long)
    destination_indices = torch.as_tensor(region_data.destination_idx, device=device, dtype=torch.long)
    support_names = (
        "support_temporal_local", "support_temporal_weekly", "support_temporal_long",
        "support_pair", "support_origin", "support_destination",
    )

    try:
        for batch_index, batch in enumerate(tqdm(loader, desc="测试", leave=False)):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch(batch, device)
            with _autocast_context(device, amp_enabled, amp_dtype):
                output = _impute_for_evaluation(model, batch, n_samples)
            samples_scaled = output["samples"].float().cpu().numpy()
            if samples_scaled.ndim != 4:
                raise ValueError(f"model.impute()['samples'] 必须为 [B,S,L,E]，实际 {samples_scaled.shape}")
            truth_scaled = batch["observed_data"].float().cpu().numpy()
            target_mask = batch["target_mask"].float().cpu().numpy()
            native_mask = batch["native_mask"].float().cpu().numpy()
            native_query_mask = batch.get("native_query_mask")
            native_query_mask = (
                native_query_mask.float().cpu().numpy()
                if native_query_mask is not None else np.zeros_like(native_mask)
            )
            generation_mask = batch.get("generation_mask")
            if generation_mask is not None:
                native_query_mask *= generation_mask.float().cpu().numpy()
            samples_raw = region_data.inverse_target(samples_scaled)
            truth_raw = region_data.inverse_target(truth_scaled)
            prediction_raw = np.median(samples_raw, axis=1)

            optional: dict[str, Any] = dict(output.get("diagnostics", {}))
            # 兼容 MRDP-OD 既有顶层诊断输出。
            for key in ("prior", "temporal_prior", "relational_prior", "gate", "router_features"):
                if key in output:
                    optional[key] = output[key]

            batch_shape = tuple(batch["observed_data"].shape)
            aligned: dict[str, np.ndarray] = {}
            for name, value in optional.items():
                tensor = _target_aligned_tensor(value, batch_shape)
                if tensor is not None:
                    array = tensor.float().cpu().numpy()
                    if name in {"prior", "temporal_prior", "relational_prior", "preliminary", "guide_prior", "graph_informed_source"}:
                        array = region_data.inverse_target(array)
                    aligned[name] = array
                elif isinstance(value, torch.Tensor) and value.ndim == 4 and tuple(value.shape[:3]) == batch_shape:
                    aligned[name] = value.float().cpu().numpy()

            # 与模型实现无关的统一诊断，使所有基线都能进入失败边界比较。
            universal_support = observation_support_features(
                batch["cond_mask"], origin_indices, destination_indices, region_data.num_cities,
            ).float().cpu().numpy()
            for feature, name in enumerate(support_names):
                aligned[name] = universal_support[..., feature]

            for item in range(target_mask.shape[0]):
                mask = target_mask[item].astype(bool)
                gaps = consecutive_target_lengths(mask)
                local_time, _ = np.where(mask)
                truth_list.append(truth_raw[item][mask])
                prediction_list.append(prediction_raw[item][mask])
                sample_list.append(samples_raw[item][:, mask])
                mask_list.append(mask.astype(np.uint8))
                gap_list.append(gaps[mask])
                start_index = int(batch["start_index"][item].detach().cpu())
                date_index_list.append((start_index + local_time).astype(np.int32))
                native_query = native_query_mask[item].astype(bool)
                if native_query.any():
                    native_time, native_pair = np.where(native_query)
                    native_values = samples_raw[item][:, native_query]
                    native_prediction_list.append(np.median(native_values, axis=0))
                    native_std_list.append(np.std(native_values, axis=0))
                    native_q05_list.append(np.quantile(native_values, 0.05, axis=0))
                    native_q50_list.append(np.quantile(native_values, 0.50, axis=0))
                    native_q95_list.append(np.quantile(native_values, 0.95, axis=0))
                    native_date_list.append((start_index + native_time).astype(np.int32))
                    native_pair_list.append(native_pair.astype(np.int32))
                for name, array in aligned.items():
                    values = array[item][mask]
                    if values.ndim == 1:
                        diagnostics_lists.setdefault(name, []).append(values)
                    else:
                        for feature in range(values.shape[-1]):
                            diagnostics_lists.setdefault(f"{name}_{feature}", []).append(values[:, feature])
            actual_rates.extend(batch["actual_missing_rate"].float().cpu().numpy().tolist())
            configured_rates.extend(batch["configured_missing_rate"].float().cpu().numpy().tolist())
            target_cell_count += float(target_mask.sum())
            native_observed_cell_count += float(native_mask.sum())
            total_cell_count += float(native_mask.size)
            del output, batch, aligned, universal_support
    finally:
        model.restore_training_weights() if hasattr(model, "restore_training_weights") else None

    if not truth_list or target_cell_count <= 0:
        raise RuntimeError("测试集没有 target entries")
    y_true = np.concatenate(truth_list)
    y_pred = np.concatenate(prediction_list)
    samples = np.concatenate(sample_list, axis=1)
    gap = np.concatenate(gap_list)
    date_index = np.concatenate(date_index_list)
    pair_index = np.concatenate([np.where(mask)[1] for mask in mask_list])

    metrics = regression_metrics(y_true, y_pred)
    if bool(getattr(model, "is_probabilistic", False)):
        metrics.update(probabilistic_metrics(y_true, samples))
    metrics.update({
        "TARGET_COUNT": int(y_true.size),
        "CONFIGURED_MISSING_RATE_MEAN": float(np.mean(configured_rates)),
        "ACTUAL_MISSING_RATE": float(target_cell_count / max(native_observed_cell_count, 1.0)),
        "ACTUAL_MISSING_RATE_MEAN": float(np.mean(actual_rates)),
        "ACTUAL_MISSING_RATE_STD": float(np.std(actual_rates)),
        "NATIVE_OBSERVATION_RATE": float(native_observed_cell_count / max(total_cell_count, 1.0)),
    })

    diagnostic_data: dict[str, Any] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "absolute_error": np.abs(y_pred - y_true),
        "gap_length": gap,
        "pair_index": pair_index,
        "date_index": date_index,
        "distance_line": distances[pair_index],
        "sample_std": np.std(samples, axis=0),
        "q025": np.quantile(samples, 0.025, axis=0),
        "q05": np.quantile(samples, 0.05, axis=0),
        "q10": np.quantile(samples, 0.10, axis=0),
        "q25": np.quantile(samples, 0.25, axis=0),
        "q50": np.quantile(samples, 0.50, axis=0),
        "q75": np.quantile(samples, 0.75, axis=0),
        "q90": np.quantile(samples, 0.90, axis=0),
        "q95": np.quantile(samples, 0.95, axis=0),
        "q975": np.quantile(samples, 0.975, axis=0),
    }
    pair_frame = region_data.pair_frame.reset_index(drop=True)
    origin_ids = pair_frame["origin_id"].to_numpy()
    destination_ids = pair_frame["destination_id"].to_numpy()
    diagnostic_data.update({
        "origin_index": np.asarray(region_data.origin_idx, dtype=np.int32)[pair_index],
        "destination_index": np.asarray(region_data.destination_idx, dtype=np.int32)[pair_index],
        "origin_id": origin_ids[pair_index],
        "destination_id": destination_ids[pair_index],
        "hsr_direct_flag": pair_frame.get(
            "hsr_direct_flag", pd.Series(np.zeros(region_data.num_pairs))
        ).to_numpy(float)[pair_index],
    })
    arrays: dict[str, np.ndarray] = {
        "y_true": y_true.astype(np.float32),
        "y_pred": y_pred.astype(np.float32),
        "samples": samples.astype(np.float32),
        "q05": diagnostic_data["q05"].astype(np.float32),
        "q50": diagnostic_data["q50"].astype(np.float32),
        "q95": diagnostic_data["q95"].astype(np.float32),
        "gap_length": gap.astype(np.int16),
        "pair_index": pair_index.astype(np.int32),
        "date_index": date_index.astype(np.int32),
    }
    for quantile in ("q025", "q10", "q25", "q75", "q90", "q975"):
        arrays[quantile] = diagnostic_data[quantile].astype(np.float32)
    if native_prediction_list:
        arrays.update({
            "native_query_prediction": np.concatenate(native_prediction_list).astype(np.float32),
            "native_query_std": np.concatenate(native_std_list).astype(np.float32),
            "native_query_q05": np.concatenate(native_q05_list).astype(np.float32),
            "native_query_q50": np.concatenate(native_q50_list).astype(np.float32),
            "native_query_q95": np.concatenate(native_q95_list).astype(np.float32),
            "native_query_date_index": np.concatenate(native_date_list).astype(np.int32),
            "native_query_pair_index": np.concatenate(native_pair_list).astype(np.int32),
        })
        metrics["NATIVE_QUERY_COUNT"] = int(sum(len(values) for values in native_prediction_list))
    for name, parts in diagnostics_lists.items():
        if parts:
            values = np.concatenate(parts)
            diagnostic_data[name] = values
            arrays[name] = values.astype(np.float32)
    available_support = [name for name in support_names if name in diagnostic_data]
    if available_support:
        support_mean = np.mean(
            np.column_stack([np.asarray(diagnostic_data[name], dtype=float) for name in available_support]),
            axis=1,
        )
        diagnostic_data["support_mean"] = support_mean
        arrays["support_mean"] = support_mean.astype(np.float32)
    for name in ("distance_line", "origin_index", "destination_index", "origin_id", "destination_id", "hsr_direct_flag"):
        values = np.asarray(diagnostic_data[name])
        arrays[name] = values.astype(np.float32 if values.dtype.kind == "f" else np.int64)

    # MRDP-OD 专属摘要只在相关诊断存在时生成。
    if "gate" in diagnostic_data:
        metrics["GATE_MEAN"] = float(np.mean(diagnostic_data["gate"]))
        metrics["GATE_STD"] = float(np.std(diagnostic_data["gate"]))
    for name, metric_name in [
        ("temporal_prior", "TEMPORAL_PRIOR_MAE"),
        ("relational_prior", "RELATIONAL_PRIOR_MAE"),
        ("prior", "FUSED_PRIOR_MAE"),
        ("preliminary", "PRELIMINARY_MAE"),
    ]:
        if name in diagnostic_data:
            metrics[metric_name] = float(np.mean(np.abs(diagnostic_data[name] - y_true)))

    return metrics, arrays, pd.DataFrame(diagnostic_data)


def evaluate_model(
    model,
    loader: DataLoader,
    region_data,
    device: torch.device,
    n_samples: int,
    amp_enabled: bool,
    amp_dtype: str,
    max_batches: int | None = None,
    random_seed: int | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray], pd.DataFrame]:
    with _evaluation_rng(device, random_seed):
        return _evaluate_model_impl(
            model,
            loader,
            region_data,
            device,
            n_samples,
            amp_enabled,
            amp_dtype,
            max_batches,
        )
