"""支持联合多场景验证、分阶段训练、早停和冻结 checkpoint 测试的训练器。"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.io import ensure_dir, write_dataframe, write_json
from utils.logger import log
from evaluation.diagnostics import failure_subgroup_metrics, network_case_snapshot
from .evaluator import _autocast_context, evaluate_loss, evaluate_model, evaluate_point_metrics, move_batch


def accumulation_group_size(batch_index: int, planned_batches: int, accumulation: int) -> int:
    if planned_batches <= 0:
        raise ValueError("planned_batches 必须大于 0")
    accumulation = max(1, int(accumulation))
    group_start = (int(batch_index) // accumulation) * accumulation
    group_end = min(group_start + accumulation, int(planned_batches))
    return max(1, group_end - group_start)


def should_validate(epoch: int, total_epochs: int, interval: int) -> bool:
    """判断当前 epoch 是否需要验证，并保证每个训练阶段的最后一轮一定验证。"""
    epoch = int(epoch)
    total_epochs = int(total_epochs)
    interval = int(interval)
    if epoch <= 0 or total_epochs <= 0:
        raise ValueError("epoch 和 total_epochs 必须大于 0")
    if interval <= 0:
        raise ValueError("validation_interval 必须大于 0")
    return epoch % interval == 0 or epoch == total_epochs


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    """对多个场景共有的数值指标做等权宏平均。"""
    if not rows:
        raise ValueError("无法对空验证结果求平均")
    keys = set.intersection(*(set(row) for row in rows))
    result: dict[str, float] = {}
    for key in sorted(keys):
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        if np.isfinite(values).any():
            result[key] = float(np.nanmean(values))
    return result


def _scenario_seed(base_seed: int, scenario_name: str) -> int:
    """由固定评价种子和场景名生成稳定且互不干扰的随机流。"""
    digest = hashlib.sha256(f"{int(base_seed)}::{scenario_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


class Trainer:
    """Train and evaluate MRDP-OD with checkpointed early stopping."""

    def __init__(
        self,
        model,
        train_set,
        valid_set,
        test_set,
        region_data,
        cfg: dict[str, Any],
        run_dir: str | Path,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_set = train_set
        if isinstance(valid_set, dict):
            self.valid_sets = dict(valid_set)
        elif valid_set is None:
            self.valid_sets = {}
        else:
            name = str(getattr(valid_set, "scenario_name", "validation"))
            self.valid_sets = {name: valid_set}
        self.valid_set = next(iter(self.valid_sets.values()), None)
        self.test_set = test_set
        self.region_data = region_data
        self.cfg = cfg
        self.run_dir = ensure_dir(run_dir)
        self.device = device
        amp_cfg = cfg["training"].get("amp", {})
        self.amp_enabled = bool(amp_cfg.get("enabled", True)) and device.type == "cuda"
        self.amp_dtype = str(amp_cfg.get("dtype", "bfloat16"))
        scaler_enabled = self.amp_enabled and self.amp_dtype == "float16"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.best_path = self.run_dir / "checkpoint_best.pt"
        self.last_path = self.run_dir / "checkpoint_last.pt"

    def _loader(self, dataset, shuffle: bool, batch_size: int | None = None) -> DataLoader:
        if dataset is None:
            raise ValueError("DataLoader 对应的数据集不能为空")
        training = self.cfg["training"]
        workers = int(self.cfg["project"].get("num_workers", 0))
        return DataLoader(
            dataset,
            batch_size=int(batch_size or training.get("batch_size", 8)),
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
            drop_last=False,
        )

    def _phase_config(self, phase: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self.cfg["training"])
        merged.pop("phases", None)
        merged.update({key: value for key, value in phase.items() if key != "name"})
        return merged

    def _optimizer(self, phase_cfg: dict[str, Any]):
        parameters = list(
            self.model.trainable_parameters()
            if hasattr(self.model, "trainable_parameters")
            else (parameter for parameter in self.model.parameters() if parameter.requires_grad)
        )
        if not parameters:
            raise RuntimeError("当前训练阶段没有可训练参数")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(phase_cfg.get("lr", 1e-3)),
            weight_decay=float(phase_cfg.get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(phase_cfg.get("lr_factor", 0.5)),
            patience=int(phase_cfg.get("lr_patience", 5)),
            min_lr=float(phase_cfg.get("min_lr", 1e-5)),
        )
        return optimizer, scheduler

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        phase: str,
        validation_summary: dict[str, float],
        validation_by_scenario: dict[str, dict[str, float]],
        optimizer,
        scheduler,
    ) -> None:
        torch.save(
            {
                "epoch": int(epoch),
                "phase": phase,
                "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_validation_score": float(validation_summary["SELECTION_SCORE"]),
                "best_validation_mae": float(validation_summary.get("MAE", np.nan)),
                "validation_summary": validation_summary,
                "validation_by_scenario": validation_by_scenario,
                "cfg": self.cfg,
            },
            path,
        )

    def _validation(
        self,
        phase_cfg: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
        if not self.valid_sets:
            raise RuntimeError("训练阶段至少需要一个验证场景")
        max_batches = phase_cfg.get("max_valid_batches")
        max_batches_value = int(max_batches) if max_batches is not None else None
        evaluation = self.cfg["evaluation"]
        validation_samples = (
            int(evaluation.get("validation_n_samples", 1))
            if bool(getattr(self.model, "is_probabilistic", False))
            else 1
        )
        metric_name = str(evaluation.get("validation_metric", "MAE")).upper()
        base_seed = int(evaluation.get("validation_seed", 12024))
        loss_rows: list[dict[str, float]] = []
        point_rows: list[dict[str, float]] = []
        by_scenario: dict[str, dict[str, float]] = {}

        for scenario_name, dataset in self.valid_sets.items():
            loader = self._loader(dataset, False)
            seed = _scenario_seed(base_seed, scenario_name)
            losses = evaluate_loss(
                self.model,
                loader,
                self.device,
                self.amp_enabled,
                self.amp_dtype,
                max_batches=max_batches_value,
                random_seed=seed,
            )
            point = evaluate_point_metrics(
                self.model,
                loader,
                self.region_data,
                self.device,
                n_samples=validation_samples,
                amp_enabled=self.amp_enabled,
                amp_dtype=self.amp_dtype,
                max_batches=max_batches_value,
                random_seed=seed,
            )
            if metric_name not in point:
                raise KeyError(f"验证指标 {metric_name} 不在模型点指标中")
            loss_rows.append(losses)
            point_rows.append(point)
            by_scenario[scenario_name] = {**losses, **point}

        aggregate_losses = _mean_dict(loss_rows)
        aggregate_points = _mean_dict(point_rows)
        aggregate_points["SELECTION_SCORE"] = float(aggregate_points[metric_name])
        return aggregate_losses, aggregate_points, by_scenario

    @staticmethod
    def _history_scenario_fields(by_scenario: dict[str, dict[str, float]]) -> dict[str, float]:
        fields: dict[str, float] = {}
        for scenario_name, metrics in by_scenario.items():
            safe_name = scenario_name.replace("-", "_").replace(" ", "_")
            for metric_name, value in metrics.items():
                fields[f"valid_{safe_name}_{metric_name}"] = float(value)
        return fields

    def fit(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """仅用训练集拟合，并按固定多场景验证宏平均指标选择 checkpoint。"""
        started = perf_counter()
        history: list[dict[str, Any]] = []
        metric_name = str(self.cfg["evaluation"].get("validation_metric", "MAE")).upper()

        if not bool(getattr(self.model, "requires_training", True)):
            losses, summary, by_scenario = self._validation(self.cfg["training"])
            record = {
                "phase": "no_training",
                "epoch": 0,
                **{f"valid_{key}": value for key, value in losses.items()},
                **{f"valid_{key}": value for key, value in summary.items()},
                **self._history_scenario_fields(by_scenario),
            }
            history.append(record)
            self._save_checkpoint(
                self.best_path, 0, "no_training", summary, by_scenario, None, None
            )
            shutil.copy2(self.best_path, self.last_path)
            frame = pd.DataFrame(history)
            write_dataframe(frame, self.run_dir / "training_history.csv")
            write_json(by_scenario, self.run_dir / "best_validation_by_scenario.json")
            return frame, {
                "training_seconds": float(perf_counter() - started),
                "epochs_completed": 0,
                "best_validation_score": float(summary["SELECTION_SCORE"]),
                "best_validation_mae": float(summary.get("MAE", np.nan)),
                "best_validation_wmape": float(summary.get("WMAPE", np.nan)),
                "validation_metric": metric_name,
                "best_validation_by_scenario": by_scenario,
            }

        if self.train_set is None:
            raise RuntimeError("可训练模型缺少训练集")

        total_epochs = 0
        final_summary: dict[str, float] = {}
        final_by_scenario: dict[str, dict[str, float]] = {}
        active_progress = None
        phases = self.model.training_phases() if hasattr(self.model, "training_phases") else [{"name": "main"}]
        try:
            for phase_index, phase in enumerate(phases):
                phase_name = str(phase.get("name", f"phase_{phase_index + 1}"))
                phase_cfg = self._phase_config(phase)
                self.model.set_training_phase(phase_name) if hasattr(self.model, "set_training_phase") else None
                optimizer, scheduler = self._optimizer(phase_cfg)
                train_loader = self._loader(
                    self.train_set,
                    True,
                    batch_size=int(phase_cfg.get("batch_size", self.cfg["training"].get("batch_size", 8))),
                )
                epochs = int(phase_cfg.get("epochs", 100))
                patience = int(phase_cfg.get("patience", 15))
                validation_interval = int(phase_cfg.get("validation_interval", 1))
                if validation_interval <= 0:
                    raise ValueError("training.validation_interval 必须大于 0")
                grad_clip = float(phase_cfg.get("grad_clip", 1.0))
                accumulation = max(1, int(phase_cfg.get("grad_accumulation_steps", 1)))
                max_train_batches = phase_cfg.get("max_train_batches")
                best_score = float("inf")
                best_summary: dict[str, float] = {}
                best_by_scenario: dict[str, dict[str, float]] = {}
                bad_epochs = 0
                phase_best = self.run_dir / f"checkpoint_{phase_name}_best.pt"
                phase_last = self.run_dir / f"checkpoint_{phase_name}_last.pt"
                log(
                    f"开始训练阶段 {phase_name}：epochs={epochs}, batch={phase_cfg.get('batch_size')}, "
                    f"每 {validation_interval} epoch 验证一次，"
                    f"多场景验证选择指标=macro-{metric_name}"
                )

                for epoch in range(1, epochs + 1):
                    # 上一轮的 100% 进度条一直保留到新 epoch 真正开始，再在原位置清掉。
                    if active_progress is not None:
                        active_progress.close()
                        active_progress = None
                    total_epochs += 1
                    self.train_set.set_epoch(total_epochs)
                    self.model.train()
                    sums: dict[str, float] = {}
                    samples_seen = 0
                    optimizer.zero_grad(set_to_none=True)
                    planned_batches = len(train_loader)
                    if max_train_batches is not None:
                        planned_batches = min(planned_batches, int(max_train_batches))
                    if planned_batches <= 0:
                        raise RuntimeError("训练配置没有允许处理任何 batch")
                    progress = tqdm(
                        total=planned_batches,
                        desc=f"{phase_name} {epoch}/{epochs}",
                        leave=False,
                        dynamic_ncols=True,
                    )
                    active_progress = progress
                    for batch_index, batch in enumerate(train_loader):
                        if batch_index >= planned_batches:
                            break
                        batch = move_batch(batch, self.device)
                        group_size = accumulation_group_size(batch_index, planned_batches, accumulation)
                        with _autocast_context(self.device, self.amp_enabled, self.amp_dtype):
                            losses = self.model.loss(batch)
                            scaled_loss = losses["loss"] / group_size
                        if self.scaler.is_enabled():
                            self.scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        group_start = (batch_index // accumulation) * accumulation
                        group_end = min(group_start + accumulation, planned_batches)
                        if (batch_index + 1) == group_end:
                            if self.scaler.is_enabled():
                                self.scaler.unscale_(optimizer)
                            if grad_clip > 0:
                                parameters_for_clip = list(
                                    self.model.trainable_parameters()
                                    if hasattr(self.model, "trainable_parameters")
                                    else (parameter for parameter in self.model.parameters() if parameter.requires_grad)
                                )
                                torch.nn.utils.clip_grad_norm_(parameters_for_clip, grad_clip)
                            if self.scaler.is_enabled():
                                self.scaler.step(optimizer)
                                self.scaler.update()
                            else:
                                optimizer.step()
                            self.model.on_optimizer_step(epoch) if hasattr(self.model, "on_optimizer_step") else None
                            optimizer.zero_grad(set_to_none=True)
                        batch_size = int(batch["observed_data"].shape[0])
                        samples_seen += batch_size
                        for key, value in losses.items():
                            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu()) * batch_size
                        progress.set_postfix(loss=sums["loss"] / max(samples_seen, 1))
                        progress.update(1)

                    # 不在这里 close；验证和日志期间仍保留这一行，直到下一 epoch 开始。
                    progress.refresh()
                    if samples_seen == 0:
                        raise RuntimeError("训练循环没有处理任何 batch")
                    train_metrics = {f"train_{key}": value / samples_seen for key, value in sums.items()}
                    if not should_validate(epoch, epochs, validation_interval):
                        history.append({
                            "phase": phase_name,
                            "epoch": epoch,
                            "global_epoch": total_epochs,
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            **train_metrics,
                        })
                        log(
                            f"{phase_name} Epoch {epoch}: train={train_metrics['train_loss']:.6f}, "
                            f"跳过验证（间隔={validation_interval}）"
                        )
                        continue
                    valid_losses, valid_summary, validation_by_scenario = self._validation(phase_cfg)
                    selection_score = float(valid_summary["SELECTION_SCORE"])
                    scheduler.step(selection_score)
                    record = {
                        "phase": phase_name,
                        "epoch": epoch,
                        "global_epoch": total_epochs,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        **train_metrics,
                        **{f"valid_{key}": value for key, value in valid_losses.items()},
                        **{f"valid_{key}": value for key, value in valid_summary.items()},
                        **self._history_scenario_fields(validation_by_scenario),
                    }
                    history.append(record)
                    mae_text = (
                        ""
                        if metric_name == "MAE"
                        else f", macro-MAE={float(valid_summary.get('MAE', np.nan)):.6f}"
                    )
                    log(
                        f"{phase_name} Epoch {epoch}: train={record['train_loss']:.6f}, "
                        f"macro-{metric_name}={selection_score:.6f}{mae_text}, lr={record['lr']:.3e}"
                    )
                    self._save_checkpoint(
                        phase_last,
                        epoch,
                        phase_name,
                        valid_summary,
                        validation_by_scenario,
                        optimizer,
                        scheduler,
                    )
                    if np.isfinite(selection_score) and selection_score < best_score:
                        best_score = selection_score
                        best_summary = dict(valid_summary)
                        best_by_scenario = validation_by_scenario
                        bad_epochs = 0
                        self._save_checkpoint(
                            phase_best,
                            epoch,
                            phase_name,
                            best_summary,
                            best_by_scenario,
                            optimizer,
                            scheduler,
                        )
                    else:
                        bad_epochs += 1
                        if bad_epochs >= patience:
                            log(
                                f"阶段 {phase_name} 早停：连续 {patience} 次验证 "
                                f"macro-{metric_name} 未改善"
                            )
                            break

                if not phase_best.exists():
                    raise RuntimeError(f"阶段 {phase_name} 没有产生有效验证 checkpoint")
                checkpoint = torch.load(phase_best, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint["model"])
                final_summary = dict(checkpoint.get("validation_summary", best_summary))
                final_by_scenario = dict(checkpoint.get("validation_by_scenario", best_by_scenario))
                shutil.copy2(phase_last, self.last_path)
                shutil.copy2(phase_best, self.best_path)
        finally:
            if active_progress is not None:
                # 已没有下一 epoch 时保留最后一条完成进度；异常时也确保终端状态恢复。
                active_progress.leave = True
                active_progress.close()

        history_frame = pd.DataFrame(history)
        write_dataframe(history_frame, self.run_dir / "training_history.csv")
        write_json(final_by_scenario, self.run_dir / "best_validation_by_scenario.json")
        runtime = {
            "training_seconds": float(perf_counter() - started),
            "epochs_completed": int(total_epochs),
            "best_validation_score": float(final_summary["SELECTION_SCORE"]),
            "best_validation_mae": float(final_summary.get("MAE", np.nan)),
            "best_validation_wmape": float(final_summary.get("WMAPE", np.nan)),
            "validation_metric": metric_name,
            "best_validation_by_scenario": final_by_scenario,
        }
        return history_frame, runtime

    def test(
        self,
        test_set=None,
        output_dir: str | Path | None = None,
        random_seed: int | None = None,
    ) -> tuple[dict[str, float], dict[str, np.ndarray], pd.DataFrame, dict[str, float]]:
        """对一个固定测试场景评价一次，并把结果写入独立测试运行目录。"""
        dataset = test_set if test_set is not None else self.test_set
        if dataset is None:
            raise RuntimeError("测试阶段缺少测试数据集")
        destination = ensure_dir(output_dir or self.run_dir)
        loader = self._loader(dataset, False)
        evaluation = self.cfg["evaluation"]
        n_samples = (
            int(evaluation.get("n_samples", 10))
            if bool(getattr(self.model, "is_probabilistic", False))
            else 1
        )
        max_test_batches = evaluation.get("max_test_batches")
        scenario_name = str(getattr(dataset, "scenario_name", self.cfg.get("search", {}).get("scenario", "test")))
        seed = random_seed
        if seed is None and evaluation.get("test_seed") is not None:
            seed = _scenario_seed(int(evaluation["test_seed"]), scenario_name)
        started = perf_counter()
        metrics, arrays, diagnostic = evaluate_model(
            self.model,
            loader,
            self.region_data,
            self.device,
            n_samples=n_samples,
            amp_enabled=self.amp_enabled,
            amp_dtype=self.amp_dtype,
            max_batches=int(max_test_batches) if max_test_batches is not None else None,
            random_seed=seed,
        )
        runtime = {
            "test_seconds": float(perf_counter() - started),
            "n_samples": n_samples,
            "test_seed": seed,
        }
        runtime.update(self._benchmark_inference(dataset, n_samples))
        write_json(metrics, destination / "metrics.json")
        prediction_arrays = dict(arrays)
        if not bool(evaluation.get("save_full_samples", True)):
            prediction_arrays.pop("samples", None)
        np.savez_compressed(destination / "predictions.npz", **prediction_arrays)
        write_dataframe(diagnostic, destination / "target_diagnostics.csv")
        subgroup_table, failure_definition = failure_subgroup_metrics(diagnostic)
        write_dataframe(subgroup_table, destination / "subgroup_metrics.csv")
        write_json(failure_definition, destination / "failure_definition.json")
        case_cfg = evaluation.get("case_study", {})
        if bool(case_cfg.get("enabled", False)):
            pair_table, city_table, case_manifest = network_case_snapshot(
                self.region_data,
                diagnostic,
                str(case_cfg.get("selection_rule", "max_target_count_then_earliest")),
            )
            pair_table.to_csv(destination / "network_case_pairs.tsv", sep="\t", index=False, encoding="utf-8")
            city_table.to_csv(destination / "network_case_cities.tsv", sep="\t", index=False, encoding="utf-8")
            write_json(case_manifest, destination / "network_case_manifest.json")
        if "gate" in arrays:
            gate_arrays = {
                key: arrays[key]
                for key in (
                    "gate",
                    "router_features_0",
                    "router_features_1",
                    "router_features_2",
                    "router_features_3",
                    "router_features_4",
                    "router_features_5",
                    "gap_length",
                    "pair_index",
                )
                if key in arrays
            }
            np.savez_compressed(destination / "gate_statistics.npz", **gate_arrays)
        if hasattr(self.model, "relation_weights"):
            write_json(self.model.relation_weights(), destination / "relation_weights.json")
        return metrics, arrays, diagnostic, runtime

    @torch.no_grad()
    def _benchmark_inference(self, dataset, evaluation_samples: int) -> dict[str, Any]:
        """在固定 batch 和样本数下执行预热与重复计时，供公平效率比较使用。"""
        settings = self.cfg["evaluation"].get("inference_benchmark", {})
        if not bool(settings.get("enabled", False)):
            return {"inference_benchmark_enabled": False}
        warmup_runs = max(0, int(settings.get("warmup_runs", 1)))
        repeats = max(1, int(settings.get("repeats", 3)))
        batch_size = max(1, int(settings.get("batch_size", 1)))
        requested_samples = max(1, int(settings.get("n_samples", 100)))
        actual_samples = requested_samples if bool(getattr(self.model, "is_probabilistic", False)) else 1
        impute_kwargs: dict[str, Any] = {"n_samples": actual_samples}
        if bool(getattr(self.model, "supports_sample_cpu_offload", False)):
            impute_kwargs["sample_output_device"] = "cpu"
        loader = self._loader(dataset, False, batch_size=batch_size)
        batch = move_batch(next(iter(loader)), self.device)
        self.model.eval()
        self.model.use_evaluation_weights() if hasattr(self.model, "use_evaluation_weights") else None
        evaluation_peak_mb = (
            float(torch.cuda.max_memory_allocated(self.device) / 1024**2)
            if self.device.type == "cuda" else 0.0
        )
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            for _ in range(warmup_runs):
                with _autocast_context(self.device, self.amp_enabled, self.amp_dtype):
                    self.model.impute(batch, **impute_kwargs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            durations: list[float] = []
            for _ in range(repeats):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                started = perf_counter()
                with _autocast_context(self.device, self.amp_enabled, self.amp_dtype):
                    self.model.impute(batch, **impute_kwargs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                durations.append(float(perf_counter() - started))
        finally:
            self.model.restore_training_weights() if hasattr(self.model, "restore_training_weights") else None
        windows = int(batch["observed_data"].shape[0])
        mean_seconds = float(np.mean(durations))
        benchmark_peak_mb = (
            float(torch.cuda.max_memory_allocated(self.device) / 1024**2)
            if self.device.type == "cuda" else 0.0
        )
        return {
            "inference_benchmark_enabled": True,
            "inference_benchmark_seconds_mean": mean_seconds,
            "inference_benchmark_seconds_sample_sd": float(np.std(durations, ddof=1)) if repeats > 1 else 0.0,
            "inference_benchmark_seconds_min": float(np.min(durations)),
            "inference_benchmark_seconds_per_window": mean_seconds / max(windows, 1),
            "inference_benchmark_windows_per_second": float(windows / max(mean_seconds, 1e-12)),
            "inference_benchmark_requested_samples": requested_samples,
            "inference_benchmark_actual_samples": actual_samples,
            "inference_benchmark_evaluation_samples": int(evaluation_samples),
            "inference_benchmark_batch_size": batch_size,
            "inference_benchmark_warmup_runs": warmup_runs,
            "inference_benchmark_repeats": repeats,
            "inference_benchmark_peak_gpu_memory_mb": benchmark_peak_mb,
            "pre_benchmark_evaluation_peak_gpu_memory_mb": evaluation_peak_mb,
        }
