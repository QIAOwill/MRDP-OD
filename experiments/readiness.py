"""核验全部实验是否已产生六个 Results 子模块所需的原始数据。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd

from training.experiment import resolve_project_path
from utils.config import config_hash
from utils.io import write_json
from utils.paths import find_completed_run, result_domain
from .configs import experiment_base_config, generated_configs
from .protocol import (
    ABLATIONS,
    MODELS,
    NETWORK_CASE,
    REGIONS,
    REPRESENTATIVE_SCENARIOS,
    ROBUSTNESS_SCENARIOS,
)


EXPERIMENT_IDS = ("01_overall", "02_robustness", "03_cross_pattern", "04_ablation", "05_calibration")
TRANSFER_TEST_SCENARIOS = (*REPRESENTATIVE_SCENARIOS, "mnar_high_30")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_run(cfg: dict[str, Any]) -> Path | None:
    results_root = resolve_project_path(cfg["paths"]["results_dir"])
    return find_completed_run(
        results_root,
        str(cfg["model"]["name"]),
        result_domain(cfg, cfg["data"]["region"]),
        config_hash(cfg),
        result_folder=cfg["paths"].get("result_folder"),
        run_subdir=cfg["paths"].get("run_subdir"),
    )


def _required_artifacts(cfg: dict[str, Any], run_dir: Path) -> list[Path]:
    role = str(cfg.get("search", {}).get("role"))
    if role in {"tuning", "final_train", "ablation_train"}:
        return [run_dir / "checkpoint_best.pt", run_dir / "metrics.json", run_dir / "runtime.json"]
    if role == "transfer_train":
        required = [run_dir / "checkpoint_best.pt", run_dir / "runtime.json", run_dir / "metrics.json"]
        for scenario_name in TRANSFER_TEST_SCENARIOS:
            destination = run_dir / "test" / scenario_name
            required.extend([
                destination / "metrics.json",
                destination / "predictions.npz",
                destination / "target_diagnostics.csv",
                destination / "subgroup_metrics.csv",
            ])
        return required
    required = [
        run_dir / "metrics.json",
        run_dir / "runtime.json",
        run_dir / "predictions.npz",
        run_dir / "target_diagnostics.csv",
        run_dir / "subgroup_metrics.csv",
        run_dir / "failure_definition.json",
    ]
    case_cfg = cfg.get("evaluation", {}).get("case_study", {})
    if bool(case_cfg.get("enabled", False)):
        required.extend([
            run_dir / "network_case_pairs.tsv",
            run_dir / "network_case_cities.tsv",
            run_dir / "network_case_manifest.json",
        ])
    return required


def _design_checks(configs: list[dict[str, Any]], smoke: bool) -> list[dict[str, Any]]:
    checks: list[tuple[str, bool, str]] = []
    # 11.1：三地区主场景、MAR/MNAR/native-like 均有全模型测试。
    accuracy_scenarios = {
        "random_50", "temporal_50", "persistent_50", "city_1",
        "mar_weather_50", "mnar_high_50", "native_like",
    }
    accuracy_rows = [
        cfg for cfg in configs
        if cfg.get("search", {}).get("role") == "final"
        and cfg.get("search", {}).get("scenario") in accuracy_scenarios
        and cfg.get("experiment", {}).get("experiment") in {"01_overall", "02_robustness"}
    ]
    accuracy_cells = {
        (cfg["model"]["name"], cfg["data"]["region"], cfg["search"]["scenario"])
        for cfg in accuracy_rows
    }
    expected_accuracy = {
        (model, region, scenario_name)
        for model in MODELS for region in REGIONS for scenario_name in accuracy_scenarios
    }
    checks.append(("11.1 multi-region accuracy", expected_accuracy <= accuracy_cells, f"cells={len(accuracy_cells)}/{len(expected_accuracy)}"))

    # smoke 只展开精简矩阵，其目的在于验证数据链；正式运行才要求完整组合。
    if smoke:
        for experiment in EXPERIMENT_IDS:
            checks.append((f"smoke {experiment}", any(
                cfg.get("experiment", {}).get("experiment") == experiment for cfg in configs
            ), "at least one generated configuration"))
    else:
        pattern_variants = {"train_random", "train_temporal", "train_persistent", "train_city"}
        pattern_cells = {
            (cfg.get("experiment", {}).get("variant_name"), cfg.get("search", {}).get("scenario"))
            for cfg in configs if cfg.get("experiment", {}).get("experiment") == "03_cross_pattern"
            and cfg.get("search", {}).get("role") == "final"
        }
        expected_pattern = {(variant, scenario_name) for variant in pattern_variants for scenario_name in REPRESENTATIVE_SCENARIOS}
        checks.append(("11.2 cross-pattern 4x4", expected_pattern <= pattern_cells, f"cells={len(pattern_cells)}/{len(expected_pattern)}"))
        ablation_variants = {
            cfg.get("experiment", {}).get("variant_name") for cfg in configs
            if cfg.get("experiment", {}).get("experiment") == "04_ablation"
        }
        checks.append(("11.3 components and routing", set(ABLATIONS) <= ablation_variants, f"variants={len(ablation_variants)}/{len(ABLATIONS)}"))

        calibration_scenarios = {
            cfg.get("search", {}).get("scenario") for cfg in configs
            if cfg.get("experiment", {}).get("experiment") == "05_calibration"
        }
        required_calibration = {*REPRESENTATIVE_SCENARIOS, "mnar_high_30"}
        checks.append(("11.4 in-domain and shifted calibration", required_calibration <= calibration_scenarios, str(sorted(str(v) for v in calibration_scenarios))))

        case_rows = [
            cfg for cfg in configs
            if bool(cfg.get("evaluation", {}).get("case_study", {}).get("enabled", False))
        ]
        case_models = {cfg["model"]["name"] for cfg in case_rows}
        checks.append(("11.5 preregistered network case", set(MODELS) <= case_models, f"region={NETWORK_CASE['region']}, scenario={NETWORK_CASE['scenario']}, models={len(case_models)}"))

        benchmark_rows = [
            cfg for cfg in configs
            if bool(cfg.get("evaluation", {}).get("inference_benchmark", {}).get("enabled", False))
        ]
        benchmark_cells = {(cfg["model"]["name"], cfg["data"]["region"]) for cfg in benchmark_rows}
        checks.append(("11.6 fixed S=100 inference benchmark", {
            (model, region) for model in MODELS for region in REGIONS
        } <= benchmark_cells, f"cells={len(benchmark_cells)}/{len(MODELS) * len(REGIONS)}"))
    return [
        {"requirement": name, "passed": bool(passed), "details": details}
        for name, passed, details in checks
    ]


def verify_results_readiness(smoke: bool = False, strict: bool = True) -> Path:
    """逐配置核验完成状态与原始文件，并对六部分实验矩阵做结构检查。"""
    config_paths = [path for experiment in EXPERIMENT_IDS for path in generated_configs(experiment, smoke)]
    # 同一路径不会重复；排序后输出便于定位中断点。
    config_paths = sorted(set(config_paths))
    configs = [_read(path) for path in config_paths]
    rows: list[dict[str, Any]] = []
    for path, cfg in zip(config_paths, configs):
        run_dir = _completed_run(cfg)
        missing: list[str] = []
        if run_dir is None:
            missing.append("completed_run")
        else:
            missing.extend(str(item.relative_to(run_dir)) for item in _required_artifacts(cfg, run_dir) if not item.exists())
            benchmark_cfg = cfg.get("evaluation", {}).get("inference_benchmark", {})
            if bool(benchmark_cfg.get("enabled", False)):
                runtime_path = run_dir / "runtime.json"
                runtime = _read(runtime_path) if runtime_path.exists() else {}
                if not bool(runtime.get("inference_benchmark_enabled", False)):
                    missing.append("runtime.inference_benchmark")
        rows.append({
            "config": str(path),
            "experiment": cfg.get("experiment", {}).get("experiment"),
            "model": cfg.get("model", {}).get("name"),
            "region": cfg.get("data", {}).get("region"),
            "role": cfg.get("search", {}).get("role"),
            "scenario": cfg.get("search", {}).get("scenario"),
            "run_dir": str(run_dir) if run_dir else "",
            "ready": not missing,
            "missing": "; ".join(missing),
        })
    design = _design_checks(configs, smoke)
    cfg = experiment_base_config("MRDP-OD", "protocol_audit", smoke)
    root = resolve_project_path(cfg["paths"]["results_dir"])
    output = root / ("_smoke" if smoke else "") / "Analysis"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "raw_result_readiness.tsv", sep="\t", index=False, encoding="utf-8")
    pd.DataFrame(design).to_csv(output / "six_module_design_readiness.tsv", sep="\t", index=False, encoding="utf-8")
    missing_count = sum(not row["ready"] for row in rows)
    failed_design = [row["requirement"] for row in design if not row["passed"]]
    status = {
        "passed": bool(rows) and missing_count == 0 and not failed_design,
        "generated_config_count": len(rows),
        "ready_config_count": len(rows) - missing_count,
        "missing_config_or_artifact_count": missing_count,
        "failed_design_requirements": failed_design,
        "config_report": str(output / "raw_result_readiness.tsv"),
        "design_report": str(output / "six_module_design_readiness.tsv"),
    }
    network_status = root / ("_smoke" if smoke else "") / "06_network" / "run_status.json"
    if not network_status.exists() or _read(network_status).get("status") != "completed":
        status["passed"] = False
        status["failed_design_requirements"].append("11.5 explicit 06_network network/downstream experiment")
    status_path = output / "results_readiness.json"
    write_json(status, status_path)
    if strict and not status["passed"]:
        raise RuntimeError(
            f"结果完整性检查未通过：缺失配置/文件 {missing_count} 项，"
            f"设计矩阵失败 {status['failed_design_requirements']}。详见 {status_path}"
        )
    return status_path
