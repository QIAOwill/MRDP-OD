"""依据多区域、多验证场景选择单个全局超参数组合。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import math

import numpy as np

from training.experiment import resolve_project_path
from utils.config import config_hash, load_json, save_json
from utils.paths import find_completed_run, result_domain
from .configs import config_root, search_hash, selected_grid_path
from .protocol import (
    MODEL_SEEDS,
    MODELS,
    PROTOCOL_VERSION,
    REGIONS,
    TUNING_SEED,
    VALIDATION_METRIC,
    VALIDATION_SCENARIOS,
    model_folder_name,
)


def _tuning_config_paths(model_name: str, smoke: bool) -> list[Path]:
    root = config_root(smoke) / "01_overall" / model_folder_name(model_name)
    return sorted(root.glob("*/tuning/*.json"))


def _completed_metrics(cfg: dict[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    results_root = resolve_project_path(cfg["paths"]["results_dir"])
    run_dir = find_completed_run(
        results_root,
        cfg["model"]["name"],
        result_domain(cfg, cfg["data"]["region"]),
        config_hash(cfg),
        result_folder=cfg["paths"].get("result_folder"),
        run_subdir=cfg["paths"].get("run_subdir"),
    )
    if run_dir is None:
        return None
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    return run_dir, json.loads(metrics_path.read_text(encoding="utf-8"))


def select_global_grid(model_name: str, smoke: bool = False) -> dict[str, Any]:
    """验证所有 grid×region 均完成后，以区域等权 macro-WMAPE 选唯一 grid。"""
    paths = _tuning_config_paths(model_name, smoke)
    if not paths:
        raise FileNotFoundError(f"找不到 {model_name} 的调参配置，请先执行超参数搜索")
    rows: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for path in paths:
        cfg = load_json(path)
        grid_id = str(cfg["search"]["grid_id"])
        region = str(cfg["data"]["region"])
        completed = _completed_metrics(cfg)
        if completed is None:
            missing.append(f"{grid_id}/{region}")
            continue
        run_dir, metrics = completed
        score = float(metrics.get("VALIDATION_SCORE", math.nan))
        if not math.isfinite(score):
            missing.append(f"{grid_id}/{region}(invalid score)")
            continue
        row = rows.setdefault(grid_id, {"scores": {}, "runs": {}, "hyperparameters": cfg["search"]["hyperparameters"]})
        row["scores"][region] = score
        row["runs"][region] = str(run_dir)
    if missing:
        preview = ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else "")
        raise RuntimeError(
            f"{model_name} 的全局调参尚未完整：缺少 {len(missing)} 个 grid-region 结果：{preview}。"
            "为防止 validation selection bias，不会用部分结果提前选参。"
        )
    expected_regions = set(REGIONS)
    candidates: list[dict[str, Any]] = []
    for grid_id, row in rows.items():
        if set(row["scores"]) != expected_regions:
            raise RuntimeError(f"{model_name}/{grid_id} 的区域集合不完整：{sorted(row['scores'])}")
        region_scores = [float(row["scores"][region]) for region in sorted(expected_regions)]
        candidates.append({
            "grid_id": grid_id,
            "macro_validation_wmape": float(np.mean(region_scores)),
            "region_validation_wmape": row["scores"],
            "hyperparameters": row["hyperparameters"],
            "source_run_dirs": row["runs"],
        })
    candidates.sort(key=lambda item: (item["macro_validation_wmape"], item["grid_id"]))
    best = candidates[0]
    payload = {
        "model_name": model_name,
        "protocol_version": PROTOCOL_VERSION,
        "search_space_hash": search_hash(model_name),
        "selection_scope": "equal-weight macro over fixed regions and validation scenarios",
        "selection_metric": f"macro_{VALIDATION_METRIC}",
        "regions": sorted(expected_regions),
        "validation_scenarios": list(VALIDATION_SCENARIOS),
        "tuning_seed": TUNING_SEED,
        "final_model_seeds": list(MODEL_SEEDS),
        "best_grid_id": best["grid_id"],
        "best_validation_score": best["macro_validation_wmape"],
        "hyperparameters": best["hyperparameters"],
        "all_candidates": candidates,
    }
    save_json(payload, selected_grid_path(model_name, smoke))
    return payload


def select_models(models: Iterable[str] = MODELS, smoke: bool = False) -> dict[str, dict[str, Any]]:
    return {model: select_global_grid(model, smoke) for model in models}
