"""统一实验生命周期：训练/验证与冻结 checkpoint 测试严格分离。"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
import json
import traceback

import numpy as np
import torch

from data.dataset import ODWindowDataset
from data.reader import read_region
from data.tensor_builder import RegionTensorData, build_region_tensor, data_manifest, scaler_metadata
from models.factory import build_model
from utils.config import config_hash, load_json, save_json
from utils.io import append_csv_row, environment_snapshot, git_commit, write_json
from utils.logger import log, set_log_file
from utils.model_names import canonical_model_name
from utils.paths import (
    allocate_run_dir,
    code_root,
    experiment_signature,
    find_completed_run,
    model_results_root,
    result_domain,
    resolve_code_relative,
    run_number,
)
from utils.seed import set_seed
from .trainer import Trainer


TRAINING_ROLES = {"tuning", "final_train", "ablation_train"}
TEST_ROLES = {"final", "ablation", "calibration"}


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 未检测到可用 CUDA")
    return device


def resolve_project_path(path: str | Path) -> Path:
    return resolve_code_relative(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_existing_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def classify_error(exc: BaseException) -> str:
    """将 CUDA OOM 显式标记，其他异常保留异常类名。"""
    message = str(exc).lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in message:
        return "CUDA_OUT_OF_MEMORY"
    return type(exc).__name__


def _append_registries(
    row: dict[str, Any],
    results_root: Path,
    model_name: str,
    result_folder: str | None = None,
) -> None:
    append_csv_row(
        row,
        model_results_root(results_root, model_name, result_folder=result_folder)
        / "experiment_registry.csv",
    )
    append_csv_row(row, results_root / "experiment_registry_all.csv")


def _validation_sets(region_data: RegionTensorData, cfg: dict[str, Any]) -> dict[str, ODWindowDataset]:
    """构造固定的多场景验证集合；旧配置回退为单场景验证。"""
    raw_scenarios = cfg.get("evaluation", {}).get("validation_scenarios") or []
    if not raw_scenarios:
        name = str(cfg.get("search", {}).get("scenario") or "validation")
        return {name: ODWindowDataset(region_data, "valid", cfg, scenario_name=name)}
    result: dict[str, ODWindowDataset] = {}
    for index, item in enumerate(raw_scenarios, 1):
        name = str(item.get("name") or f"validation_{index}")
        if name in result:
            raise ValueError(f"验证场景名称重复：{name}")
        result[name] = ODWindowDataset(
            region_data,
            "valid",
            cfg,
            mask_spec=item,
            scenario_name=name,
        )
    return result


def _resolve_training_checkpoint(
    cfg: dict[str, Any],
    model_name: str,
    region: str,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """由冻结的训练配置哈希定位唯一已完成 checkpoint，并进行一致性校验。"""
    checkpoint_cfg = cfg.get("checkpoint", {})
    source_value = checkpoint_cfg.get("training_config")
    expected_hash = str(checkpoint_cfg.get("training_config_hash", ""))
    if not source_value or not expected_hash:
        raise ValueError("测试配置缺少 checkpoint.training_config 或 training_config_hash")
    source_path = resolve_code_relative(source_value)
    training_cfg = load_json(source_path)
    actual_hash = config_hash(training_cfg)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"训练配置已被修改：期望 hash={expected_hash}，当前 hash={actual_hash}。"
            "请重新生成正式测试配置。"
        )
    source_model = canonical_model_name(training_cfg["model"]["name"])
    source_role = str(training_cfg.get("search", {}).get("role", ""))
    requested_role = str(cfg.get("search", {}).get("role"))
    if requested_role in {"final", "calibration"}:
        expected_source_role = "final_train"
    else:
        expected_source_role = "ablation_train"
    if source_model != model_name or source_role != expected_source_role:
        raise RuntimeError(
            f"测试配置与训练配置不匹配：model={source_model}, role={source_role}, "
            f"期望 model={model_name}, role={expected_source_role}"
        )
    if int(training_cfg["project"]["seed"]) != int(cfg["project"]["seed"]):
        raise RuntimeError("测试配置与训练 checkpoint 的 model seed 不一致")
    if str(training_cfg.get("search", {}).get("grid_id")) != str(cfg.get("search", {}).get("grid_id")):
        raise RuntimeError("测试配置与训练 checkpoint 的 grid_id 不一致")
    if str(training_cfg.get("search", {}).get("search_space_hash")) != str(
        cfg.get("search", {}).get("search_space_hash")
    ):
        raise RuntimeError("测试配置与训练 checkpoint 的 search_space_hash 不一致")

    training_results_root = resolve_project_path(training_cfg["paths"]["results_dir"])
    training_folder = training_cfg.get("paths", {}).get("result_folder")
    training_domain = result_domain(training_cfg, region)
    training_run = find_completed_run(
        training_results_root,
        model_name,
        training_domain,
        actual_hash,
        result_folder=training_folder,
        run_subdir=training_cfg.get("paths", {}).get("run_subdir"),
    )
    if training_run is None:
        raise FileNotFoundError(
            f"尚未找到已完成的训练运行：{source_path}。请先执行 final_train/ablation_train。"
        )
    checkpoint_path = training_run / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"训练运行缺少 checkpoint_best.pt：{training_run}")
    runtime = _read_json_if_exists(training_run / "runtime.json")
    metadata = _read_json_if_exists(training_run / "run_metadata.json")
    return checkpoint_path, training_run, training_cfg, runtime, metadata


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _base_metadata(
    cfg: dict[str, Any],
    signature: dict[str, Any],
    hash_value: str,
    run_dir: Path,
    model,
    parameter_breakdown: dict[str, int],
    created_at: str,
) -> dict[str, Any]:
    return {
        "project": cfg["project"]["name"],
        "run_number": run_number(run_dir),
        "config_hash": hash_value,
        "protocol_version": cfg.get("protocol_version", "unspecified"),
        **signature,
        "region": cfg["data"]["region"],
        "model_name": canonical_model_name(cfg["model"]["name"]),
        "is_probabilistic": bool(getattr(model, "is_probabilistic", False)),
        "implementation_type": str(getattr(model, "implementation_type", "unknown")),
        "upstream_repository": getattr(model, "upstream_repository", None),
        "parameter_count": parameter_breakdown["total"],
        "window_len": cfg["data"]["window_len"],
        "city_limit": cfg["data"].get("city_limit"),
        "diffusion_steps": cfg.get("diffusion", {}).get("num_steps"),
        "evaluation_samples": cfg["evaluation"]["n_samples"],
        "created_at": created_at,
    }


def _validation_metrics(train_runtime: dict[str, Any]) -> dict[str, float]:
    return {
        "VALIDATION_SCORE": float(train_runtime["best_validation_score"]),
        "VALIDATION_MAE": float(train_runtime.get("best_validation_mae", np.nan)),
        "VALIDATION_WMAPE": float(train_runtime.get("best_validation_wmape", np.nan)),
    }


def run_single_experiment(
    cfg: dict[str, Any],
    region_data: RegionTensorData | None = None,
) -> dict[str, Any]:
    """执行一个训练配置或一个冻结 checkpoint 测试配置。"""
    cfg = deepcopy(cfg)
    model_name = canonical_model_name(cfg["model"].get("name", "MRDP-OD"))
    cfg["model"]["name"] = model_name
    region = str(cfg["data"]["region"])
    domain = result_domain(cfg, region)
    role = str(cfg.get("search", {}).get("role", ""))
    if role not in TRAINING_ROLES | TEST_ROLES:
        raise ValueError(
            f"统一流水线不支持 search.role={role!r}；请重新生成实验配置。"
        )
    results_root = resolve_project_path(cfg["paths"]["results_dir"])
    result_folder = cfg.get("paths", {}).get("result_folder")
    run_subdir = cfg.get("paths", {}).get("run_subdir")
    hash_value = config_hash(cfg)
    existing = find_completed_run(
        results_root,
        model_name,
        domain,
        hash_value,
        result_folder=result_folder,
        run_subdir=run_subdir,
    )
    if existing is not None:
        log(f"跳过已完成配置：{existing} / hash={hash_value}")
        metrics = _read_existing_metrics(existing)
        return {
            "run_dir": str(existing),
            "metrics": metrics,
            "status": "skipped",
            "config_hash": hash_value,
            "validation_score": metrics.get("VALIDATION_SCORE"),
            "validation_mae": metrics.get("VALIDATION_MAE"),
        }

    run_dir = allocate_run_dir(
        results_root,
        model_name,
        domain,
        result_folder=result_folder,
        run_subdir=run_subdir,
    )
    set_log_file(run_dir / "run.log")
    signature = experiment_signature(cfg)
    status = {
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "config_hash": hash_value,
        "protocol_version": cfg.get("protocol_version", "unspecified"),
        **signature,
    }
    write_json(status, run_dir / "run_status.json")
    save_json(cfg, run_dir / "config_resolved.json")
    _write_text(run_dir / "config_hash.txt", hash_value + "\n")
    _write_text(run_dir / "seed.txt", str(cfg["project"]["seed"]) + "\n")
    _write_text(run_dir / "git_commit.txt", git_commit(code_root()) + "\n")
    environment = environment_snapshot()
    write_json(environment, run_dir / "environment.json")
    _write_text(
        run_dir / "environment.txt",
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n",
    )

    started = perf_counter()
    try:
        set_seed(int(cfg["project"]["seed"]), bool(cfg["project"].get("deterministic", False)))
        cpu_threads = max(1, int(cfg["project"].get("cpu_threads", 8)))
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(cpu_threads, 4)))
        except RuntimeError:
            pass
        device = select_device(str(cfg["project"].get("device", "auto")))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        log(f"模型：{model_name}")
        log(f"运行目录：{run_dir}")
        log(f"协议/角色：{cfg.get('protocol_version')} / {role}")
        log(f"参数组合：{signature}")
        log(f"设备：{device} / {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

        if region_data is None:
            data_dir = resolve_project_path(cfg["paths"]["dataset_dir"])
            raw = read_region(data_dir, region)
            region_data = build_region_tensor(region, raw, cfg)
        write_json(data_manifest(region_data), run_dir / "data_manifest.json")
        write_json(scaler_metadata(region_data), run_dir / "scaler_metadata.json")

        model = build_model(region_data, cfg)
        parameter_breakdown = model.parameter_breakdown()
        metadata = _base_metadata(
            cfg,
            signature,
            hash_value,
            run_dir,
            model,
            parameter_breakdown,
            status["started_at"],
        )

        if role in TRAINING_ROLES:
            train_set = ODWindowDataset(region_data, "train", cfg, scenario_name="joint_training")
            valid_sets = _validation_sets(region_data, cfg)
            metadata.update(
                {
                    "train_windows": len(train_set),
                    "valid_windows": sum(len(dataset) for dataset in valid_sets.values()),
                    "validation_scenarios": list(valid_sets),
                    "test_windows": 0,
                }
            )
            write_json(metadata, run_dir / "run_metadata.json")
            write_json(parameter_breakdown, run_dir / "model_summary.json")
            trainer = Trainer(
                model,
                train_set,
                valid_sets,
                None,
                region_data,
                cfg,
                run_dir,
                device,
            )
            _, train_runtime = trainer.fit()
            metrics = _validation_metrics(train_runtime)
            runtime = {
                **train_runtime,
                "test_seconds": 0.0,
                "n_samples": 0,
                "total_seconds": float(perf_counter() - started),
                "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda"
                else 0.0,
            }
            write_json(metrics, run_dir / "metrics.json")
        else:
            checkpoint_path, training_run, training_cfg, source_runtime, source_metadata = _resolve_training_checkpoint(
                cfg, model_name, region
            )
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            checkpoint_phase = str(checkpoint.get("phase", "main"))
            if hasattr(model, "set_training_phase") and checkpoint_phase not in {"no_training", ""}:
                model.set_training_phase(checkpoint_phase)
            evaluation_split = str(cfg.get("evaluation", {}).get("split", "test"))
            if evaluation_split not in {"valid", "test"}:
                raise ValueError(f"evaluation.split 只能是 valid 或 test：{evaluation_split}")
            test_set = ODWindowDataset(
                region_data,
                evaluation_split,
                cfg,
                scenario_name=str(cfg.get("search", {}).get("scenario") or "test"),
            )
            metadata.update(
                {
                    "train_windows": source_metadata.get("train_windows"),
                    "valid_windows": source_metadata.get("valid_windows"),
                    "test_windows": len(test_set),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_run_dir": str(training_run),
                    "checkpoint_config_hash": config_hash(training_cfg),
                }
            )
            write_json(metadata, run_dir / "run_metadata.json")
            write_json(parameter_breakdown, run_dir / "model_summary.json")
            trainer = Trainer(
                model,
                None,
                None,
                test_set,
                region_data,
                cfg,
                run_dir,
                device,
            )
            metrics, _, _, test_runtime = trainer.test()
            validation = checkpoint.get("validation_summary", {})
            metrics.update(
                {
                    "VALIDATION_SCORE": float(validation.get("SELECTION_SCORE", np.nan)),
                    "VALIDATION_MAE": float(validation.get("MAE", np.nan)),
                    "VALIDATION_WMAPE": float(validation.get("WMAPE", np.nan)),
                }
            )
            write_json(metrics, run_dir / "metrics.json")
            source_training_seconds = float(source_runtime.get("training_seconds", 0.0))
            runtime = {
                **test_runtime,
                "training_seconds": source_training_seconds,
                "epochs_completed": source_runtime.get("epochs_completed"),
                "best_validation_score": source_runtime.get("best_validation_score"),
                "best_validation_mae": source_runtime.get("best_validation_mae"),
                "best_validation_wmape": source_runtime.get("best_validation_wmape"),
                "evaluation_wall_seconds": float(perf_counter() - started),
                "total_seconds": source_training_seconds + float(test_runtime["test_seconds"]),
                "peak_gpu_memory_mb": max(
                    float(source_runtime.get("peak_gpu_memory_mb", 0.0)),
                    float(test_runtime.get("pre_benchmark_evaluation_peak_gpu_memory_mb", 0.0)),
                    float(torch.cuda.max_memory_allocated(device) / 1024**2)
                    if device.type == "cuda"
                    else 0.0,
                ),
                "checkpoint_run_dir": str(training_run),
            }

        write_json(runtime, run_dir / "runtime.json")
        status.update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "metrics": metrics,
            }
        )
        write_json(status, run_dir / "run_status.json")
        registry_row = {
            **metadata,
            **metrics,
            **runtime,
            "status": "completed",
            "run_dir": str(run_dir),
        }
        _append_registries(
            registry_row,
            results_root,
            model_name,
            result_folder=result_folder,
        )
        report_lines = [
            f"# {model_name} 运行报告",
            "",
            f"- 协议：{cfg.get('protocol_version', 'unspecified')}",
            f"- 角色：{role}",
            f"- 城市群：{region}",
            f"- 结果域：{domain}",
            f"- 运行目录：{run_dir.name}",
            f"- Grid ID：{signature.get('grid_id')}",
            f"- 场景：{signature.get('scenario')}",
            f"- 随机种子：{signature.get('seed')}",
            f"- 配置哈希：{hash_value}",
            f"- 参数量：{parameter_breakdown['total']:,}",
            "",
            "## 指标",
            "",
            *[f"- {key}: {value}" for key, value in metrics.items()],
        ]
        _write_text(run_dir / "run_report.md", "\n".join(report_lines) + "\n")
        log(f"运行完成：{metrics}")
        return {
            "run_dir": str(run_dir),
            "metrics": metrics,
            "status": "completed",
            "config_hash": hash_value,
            "validation_score": metrics.get("VALIDATION_SCORE"),
            "validation_mae": metrics.get("VALIDATION_MAE"),
        }
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "error": repr(exc),
                "error_type": classify_error(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(status, run_dir / "run_status.json")
        _write_text(run_dir / "error.txt", traceback.format_exc())
        failed_row = {
            **signature,
            "protocol_version": cfg.get("protocol_version", "unspecified"),
            "config_hash": hash_value,
            "model_name": model_name,
            "status": "failed",
            "run_dir": str(run_dir),
            "error_type": classify_error(exc),
            "error": repr(exc),
        }
        _append_registries(
            failed_row,
            results_root,
            model_name,
            result_folder=result_folder,
        )
        log(f"运行失败 [{classify_error(exc)}]：{exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        set_log_file(None)
