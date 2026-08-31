"""生成可复现的原子实验配置，不运行模型或读取测试指标。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable
import json
import shutil

from .search_space import (
    base_config,
    expand_search_space,
    read_search_spaces,
    set_nested,
)
from utils.config import config_hash, save_json, validate_config
from .protocol import (
    ABLATIONS,
    CORE_SCENARIOS,
    EXPERIMENTS,
    MASK_SEED,
    MODEL_SEEDS,
    MODELS,
    PATTERN_SHIFT_MODELS,
    PROBABILISTIC_MODELS,
    PROTOCOL_VERSION,
    NETWORK_CASE,
    REGIONS,
    REPRESENTATIVE_SCENARIOS,
    ROBUSTNESS_SCENARIOS,
    TEST_SEED,
    TUNING_SEED,
    VALIDATION_METRIC,
    VALIDATION_SCENARIOS,
    VALIDATION_SEED,
    mixed_training_scenarios,
    model_folder_name,
    scenario,
    scenario_list,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT
RESULTS_ROOT = CODE_ROOT / "outputs"

TUNING_VALIDATION_POLICIES: dict[str, dict[str, Any]] = {}


def config_root(smoke: bool = False) -> Path:
    """返回运行时配置目录；配置与模型输出统一保存在 Results 下。"""
    return RESULTS_ROOT / "_configs" / ("smoke" if smoke else "experiments")


def config_reference(path: Path) -> str:
    """将输出目录中的配置转换为相对代码根目录的可移植引用。"""
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    return relative.as_posix()


def search_space(model_name: str) -> dict[str, Any]:
    return read_search_spaces()[model_name]


def search_hash(model_name: str) -> str:
    return config_hash({
        "protocol_version": PROTOCOL_VERSION,
        "model_name": model_name,
        "search_space": search_space(model_name),
        "regions": list(REGIONS),
        "validation_scenarios": list(VALIDATION_SCENARIOS),
        "selection_metric": f"macro_{VALIDATION_METRIC}",
    })


def experiment_base_config(model_name: str, experiment: str, smoke: bool = False) -> dict[str, Any]:
    if model_name not in MODELS:
        raise ValueError(f"未知模型：{model_name}")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"未知实验：{experiment}")
    cfg = base_config(model_name)
    cfg["protocol_version"] = PROTOCOL_VERSION
    cfg["project"]["name"] = model_name
    cfg["project"]["deterministic"] = True
    prefix = "_smoke/" if smoke else ""
    cfg["paths"]["result_folder"] = (
        f"{prefix}{experiment}/{model_folder_name(model_name)}"
    )
    cfg["data"]["region"] = REGIONS[0]
    cfg["data"]["entity_split"] = None
    cfg["data"]["target_scaler"] = "standard"
    cfg["data"]["train_fraction"] = 1.0
    cfg["data"]["feature_dropout"] = {
        "context": 0.0, "pair_static": 0.0, "city_static": 0.0, "seed": 52024,
    }
    cfg["mask"]["seed"] = MASK_SEED
    cfg["mask"]["driver_strength"] = 3.0
    cfg["mask"]["train_mix"] = None
    cfg["mask"]["train_scenarios"] = mixed_training_scenarios()
    cfg["evaluation"].update({
        "n_samples": 100,
        "validation_n_samples": 5,
        "validation_metric": VALIDATION_METRIC,
        "validation_seed": VALIDATION_SEED,
        "test_seed": TEST_SEED,
        "validation_scenarios": scenario_list(VALIDATION_SCENARIOS),
        "split": "test",
        "entity_target": False,
        "include_native_queries": False,
        "save_full_samples": True,
        "inference_benchmark": {
            "enabled": False,
            "n_samples": 100,
            "batch_size": 1,
            "warmup_runs": 1,
            "repeats": 3,
        },
        "case_study": {
            "enabled": False,
            **deepcopy(NETWORK_CASE),
        },
    })
    if model_name == "MRDP-OD":
        cfg["model"].setdefault("city_token_mode", "transductive")
        cfg["model"].setdefault("use_residual_diffusion", True)
        cfg["model"].setdefault("router_feature_mask", [1.0] * 6)
    cfg["experiment"] = {
        "experiment": experiment,
        "experiment_name": experiment,
        "stage": None,
        "smoke": bool(smoke),
        "source_regions": None,
        "target_region": None,
    }
    if smoke:
        cfg["training"].update({
            "epochs": 1, "patience": 1, "max_train_batches": 1, "max_valid_batches": 1,
        })
        cfg["evaluation"].update({"n_samples": 2, "validation_n_samples": 1, "max_test_batches": 1})
        cfg["evaluation"]["inference_benchmark"].update({"n_samples": 2, "warmup_runs": 0, "repeats": 1})
    return cfg


def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        set_nested(cfg, key, value)


def _apply_experiment_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> None:
    """应用实验专用覆盖项；父级必须存在，但允许新增叶子配置键。"""
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target: dict[str, Any] = cfg
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                raise KeyError(f"实验覆盖项的父级配置不存在：{dotted_key}")
            target = child
        target[parts[-1]] = deepcopy(value)


def _merge_existing_sections(target: dict[str, Any], updates: dict[str, Any]) -> None:
    """递归合并运行策略；允许新增策略键，但不替换未提及的模型参数。"""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_existing_sections(target[key], value)
        else:
            target[key] = deepcopy(value)


def _apply_tuning_validation_policy(cfg: dict[str, Any], model_name: str, smoke: bool) -> None:
    """只给尚未运行的高开销基线添加调参验证加速策略。"""
    if smoke:
        return
    policy = TUNING_VALIDATION_POLICIES.get(model_name)
    if policy:
        _merge_existing_sections(cfg, policy)


def _config_path(
    experiment: str,
    model_name: str,
    stage: str,
    domain: str,
    filename: str,
    smoke: bool,
) -> Path:
    root = config_root(smoke)
    return root / experiment / model_folder_name(model_name) / domain / stage / filename


def _save(cfg: dict[str, Any], path: Path) -> Path:
    validate_config(cfg)
    save_json(cfg, path)
    return path


def _prepare_run_location(
    cfg: dict[str, Any], region: str, stage: str, scenario_name: str, seed: int, grid_id: str | None = None,
) -> None:
    cfg["data"]["region"] = region
    cfg["paths"]["result_domain"] = region
    parts = [stage]
    if grid_id:
        parts.append(grid_id)
    if scenario_name:
        parts.append(scenario_name)
    parts.append(f"seed_{int(seed)}")
    cfg["paths"]["run_subdir"] = "/".join(parts)
    cfg["experiment"]["stage"] = stage
    cfg["experiment"]["target_region"] = region


def _grid_candidates(model_name: str, smoke: bool) -> list[dict[str, Any]]:
    candidates = expand_search_space(model_name, search_space(model_name))
    return candidates[:1] if smoke else candidates


def generate_overall_tuning(
    models: Iterable[str] = MODELS,
    regions: Iterable[str] = REGIONS,
    smoke: bool = False,
    clean: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for model_name in models:
        candidates = _grid_candidates(model_name, smoke)
        if clean:
            root = _config_path("01_overall", model_name, "tuning", "_", "_", smoke).parents[2]
            if root.exists():
                shutil.rmtree(root)
        for index, overrides in enumerate(candidates, 1):
            grid_id = f"grid_{index:04d}"
            for region in regions:
                cfg = experiment_base_config(model_name, "01_overall", smoke)
                _apply_overrides(cfg, overrides)
                _apply_tuning_validation_policy(cfg, model_name, smoke)
                cfg["project"]["seed"] = TUNING_SEED
                _prepare_run_location(cfg, region, "tuning", "multi_validation", TUNING_SEED, grid_id)
                cfg["search"] = {
                    "role": "tuning", "grid_id": grid_id, "scenario": "multi_validation",
                    "search_space_hash": search_hash(model_name),
                    "selection_metric": f"macro_{VALIDATION_METRIC}",
                    "hyperparameters": deepcopy(overrides),
                }
                path = _config_path(
                    "01_overall", model_name, "tuning", region,
                    f"{grid_id}__seed_{TUNING_SEED}.json", smoke,
                )
                paths.append(_save(cfg, path))
    return paths


def selected_grid_path(model_name: str, smoke: bool = False) -> Path:
    root = config_root(smoke)
    return root / "_selected" / f"{model_folder_name(model_name)}.json"


def load_selected(model_name: str, smoke: bool = False) -> dict[str, Any]:
    path = selected_grid_path(model_name, smoke)
    if not path.exists():
        raise FileNotFoundError(f"尚未选择 {model_name} 的全局最佳 grid：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION or payload.get("search_space_hash") != search_hash(model_name):
        raise RuntimeError(f"{model_name} 的 selected grid 已过期，请重新 tuning")
    return payload


def _seeds(model_name: str, smoke: bool) -> tuple[int, ...]:
    if smoke:
        return (TUNING_SEED,)
    return MODEL_SEEDS


def _training_and_tests(
    experiment: str,
    model_name: str,
    region: str,
    seed: int,
    grid_id: str,
    overrides: dict[str, Any],
    train_scenarios: list[dict[str, Any]],
    test_scenarios: Iterable[str],
    smoke: bool,
    variant_name: str = "joint_training",
    model_overrides: dict[str, Any] | None = None,
    entity_split: dict[str, Any] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> list[Path]:
    cfg = experiment_base_config(model_name, experiment, smoke)
    _apply_overrides(cfg, overrides)
    if model_overrides:
        cfg["model"].update(deepcopy(model_overrides))
    if config_overrides:
        _apply_experiment_overrides(cfg, config_overrides)
    cfg["data"]["entity_split"] = deepcopy(entity_split)
    cfg["experiment"]["variant_name"] = variant_name
    cfg["mask"]["train_scenarios"] = deepcopy(train_scenarios)
    cfg["project"]["seed"] = int(seed)
    train_role = "ablation_train" if experiment == "04_ablation" else "final_train"
    _prepare_run_location(cfg, region, "final_train", variant_name, seed, grid_id)
    cfg["search"] = {
        "role": train_role, "grid_id": grid_id, "scenario": variant_name,
        "search_space_hash": search_hash(model_name), "hyperparameters": deepcopy(overrides),
    }
    train_path = _config_path(
        experiment, model_name, "final_train", region,
        f"{variant_name}__seed_{seed}.json", smoke,
    )
    _save(cfg, train_path)
    paths = [train_path]
    training_hash = config_hash(cfg)
    for scenario_name in test_scenarios:
        test_cfg = deepcopy(cfg)
        test_cfg["mask"].update(scenario(scenario_name))
        test_cfg["mask"].pop("name", None)
        test_cfg["mask"]["train_scenarios"] = None
        test_cfg["evaluation"]["entity_target"] = bool(entity_split)
        test_cfg["evaluation"]["inference_benchmark"]["enabled"] = bool(
            experiment == "01_overall" and scenario_name == "random_50"
        )
        test_cfg["evaluation"]["case_study"]["enabled"] = bool(
            experiment == "01_overall"
            and region == NETWORK_CASE["region"]
            and scenario_name == NETWORK_CASE["scenario"]
        )
        role = "ablation" if experiment == "04_ablation" else "final"
        test_cfg["search"].update({"role": role, "scenario": scenario_name})
        _prepare_run_location(test_cfg, region, "test", f"{variant_name}/{scenario_name}", seed)
        test_cfg["checkpoint"] = {
            "training_config": config_reference(train_path),
            "training_config_hash": training_hash,
        }
        test_path = _config_path(
            experiment, model_name, "test", region,
            f"{variant_name}__{scenario_name}__seed_{seed}.json", smoke,
        )
        _save(test_cfg, test_path)
        paths.append(test_path)
    return paths


def generate_overall(
    models: Iterable[str] = MODELS,
    regions: Iterable[str] = REGIONS,
    smoke: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for model_name in models:
        selected = load_selected(model_name, smoke)
        for region in regions:
            for seed in _seeds(model_name, smoke):
                paths.extend(_training_and_tests(
                    "01_overall", model_name, region, seed,
                    selected["best_grid_id"], selected["hyperparameters"],
                    mixed_training_scenarios(), CORE_SCENARIOS, smoke,
                ))
    return paths


def _find_overall_training_config(model_name: str, region: str, seed: int, smoke: bool) -> Path:
    root = _config_path("01_overall", model_name, "final_train", region, "_", smoke).parent
    matches = sorted(root.glob(f"*__seed_{seed}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{model_name}/{region}/seed={seed} 应有唯一 01_overall training config，实际 {len(matches)}")
    return matches[0]


def _find_variant_training_config(
    experiment: str,
    model_name: str,
    region: str,
    seed: int,
    variant_name: str,
    smoke: bool,
) -> Path:
    """定位指定实验变体唯一的训练配置，供分布外校准复用冻结 checkpoint。"""
    root = _config_path(experiment, model_name, "final_train", region, "_", smoke).parent
    matches = sorted(root.glob(f"{variant_name}__seed_{seed}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{experiment}/{model_name}/{region}/{variant_name}/seed={seed} "
            f"应有唯一 training config，实际 {len(matches)}"
        )
    return matches[0]


def generate_robustness(
    models: Iterable[str] = MODELS,
    regions: Iterable[str] = REGIONS,
    smoke: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    for model_name in models:
        selected = load_selected(model_name, smoke)
        for region in regions:
            for seed in _seeds(model_name, smoke):
                train_path = _find_overall_training_config(model_name, region, seed, smoke)
                train_cfg = json.loads(train_path.read_text(encoding="utf-8"))
                training_hash = config_hash(train_cfg)
                for scenario_name in ROBUSTNESS_SCENARIOS:
                    cfg = deepcopy(train_cfg)
                    cfg["paths"]["result_folder"] = experiment_base_config(model_name, "02_robustness", smoke)["paths"]["result_folder"]
                    cfg["experiment"].update({"experiment": "02_robustness", "experiment_name": "02_robustness"})
                    cfg["mask"].update(scenario(scenario_name))
                    cfg["mask"].pop("name", None)
                    cfg["mask"]["train_scenarios"] = None
                    cfg["search"].update({"role": "final", "scenario": scenario_name})
                    _prepare_run_location(cfg, region, "test", scenario_name, seed)
                    cfg["checkpoint"] = {
                        "training_config": config_reference(train_path),
                        "training_config_hash": training_hash,
                    }
                    path = _config_path(
                        "02_robustness", model_name, "test", region,
                        f"{scenario_name}__seed_{seed}.json", smoke,
                    )
                    paths.append(_save(cfg, path))
                if model_name in PROBABILISTIC_MODELS:
                    cfg = deepcopy(train_cfg)
                    cfg["paths"]["result_folder"] = experiment_base_config(model_name, "02_robustness", smoke)["paths"]["result_folder"]
                    cfg["experiment"].update({"experiment": "02_robustness", "experiment_name": "02_robustness"})
                    cfg["mask"].update(scenario("random_30"))
                    cfg["mask"].pop("name", None)
                    cfg["mask"]["train_scenarios"] = None
                    cfg["evaluation"]["include_native_queries"] = True
                    cfg["search"].update({"role": "final", "scenario": "native_operational"})
                    _prepare_run_location(cfg, region, "test", "native_operational", seed)
                    cfg["checkpoint"] = {
                        "training_config": config_reference(train_path),
                        "training_config_hash": training_hash,
                    }
                    path = _config_path(
                        "02_robustness", model_name, "test", region,
                        f"native_operational__seed_{seed}.json", smoke,
                    )
                    paths.append(_save(cfg, path))
        # 扩展机制训练只给主模型和一个强概率基线，避免为每个机制单独重训。
        if model_name == "MRDP-OD":
            extended_names = list(CORE_SCENARIOS) + [
                name for name in ROBUSTNESS_SCENARIOS if not name.startswith("random_")
            ]
            for region in regions:
                for seed in _seeds(model_name, smoke):
                    paths.extend(_training_and_tests(
                        "02_robustness", model_name, region, seed,
                        selected["best_grid_id"], selected["hyperparameters"],
                        mixed_training_scenarios(extended_names), ROBUSTNESS_SCENARIOS, smoke,
                        variant_name="extended_mechanism_training",
                    ))
    return paths


def generate_ablation(regions: Iterable[str] = REGIONS, smoke: bool = False) -> list[Path]:
    selected = load_selected("MRDP-OD", smoke)
    paths: list[Path] = []
    variants = list(ABLATIONS.items())[:2] if smoke else list(ABLATIONS.items())
    for region in regions:
        for seed in _seeds("MRDP-OD", smoke):
            for name, model_overrides in variants:
                paths.extend(_training_and_tests(
                    "04_ablation", "MRDP-OD", region, seed,
                    selected["best_grid_id"], selected["hyperparameters"],
                    mixed_training_scenarios(), REPRESENTATIVE_SCENARIOS, smoke,
                    variant_name=name, model_overrides=model_overrides,
                ))
    return paths


def generate_cross_pattern(
    models: Iterable[str] = PATTERN_SHIFT_MODELS,
    regions: Iterable[str] = REGIONS,
    smoke: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    patterns = {
        "train_random": ("random_30", "random_50"),
        "train_temporal": ("temporal_30", "temporal_50"),
        "train_persistent": ("persistent_30", "persistent_50"),
        "train_city": ("city_1", "city_2"),
    }
    if smoke:
        patterns = {"train_random": patterns["train_random"]}
    for model_name in models:
        selected = load_selected(model_name, smoke)
        for region in regions:
            for seed in _seeds(model_name, smoke):
                for train_name, scenario_names in patterns.items():
                    paths.extend(_training_and_tests(
                        "03_cross_pattern", model_name, region, seed,
                        selected["best_grid_id"], selected["hyperparameters"],
                        mixed_training_scenarios(scenario_names), REPRESENTATIVE_SCENARIOS, smoke,
                        variant_name=train_name,
                    ))
    return paths


def generate_calibration(
    models: Iterable[str] = PROBABILISTIC_MODELS,
    regions: Iterable[str] = REGIONS,
    smoke: bool = False,
) -> list[Path]:
    """生成 validation 样本配置；正式 test 样本直接复用 01_overall/02_robustness。"""
    paths: list[Path] = []
    for model_name in models:
        for region in regions:
            for seed in _seeds(model_name, smoke):
                train_path = _find_overall_training_config(model_name, region, seed, smoke)
                train_cfg = json.loads(train_path.read_text(encoding="utf-8"))
                calibration_scenarios = (*REPRESENTATIVE_SCENARIOS, "mnar_high_30")
                for scenario_name in calibration_scenarios:
                    cfg = deepcopy(train_cfg)
                    cfg["paths"]["result_folder"] = experiment_base_config(model_name, "05_calibration", smoke)["paths"]["result_folder"]
                    cfg["experiment"].update({"experiment": "05_calibration", "experiment_name": "05_calibration"})
                    cfg["mask"].update(scenario(scenario_name))
                    cfg["mask"].pop("name", None)
                    cfg["mask"]["train_scenarios"] = None
                    cfg["evaluation"]["split"] = "valid"
                    cfg["search"].update({"role": "calibration", "scenario": scenario_name})
                    _prepare_run_location(cfg, region, "validation", scenario_name, seed)
                    cfg["checkpoint"] = {
                        "training_config": config_reference(train_path),
                        "training_config_hash": config_hash(train_cfg),
                    }
                    path = _config_path(
                        "05_calibration", model_name, "validation", region,
                        f"{scenario_name}__seed_{seed}.json", smoke,
                    )
                    paths.append(_save(cfg, path))

    return paths


GENERATORS = {
    "overall_tuning": generate_overall_tuning,
    "overall_final": generate_overall,
    "04_ablation": generate_ablation,
    "02_robustness": generate_robustness,
    "03_cross_pattern": generate_cross_pattern,
    "05_calibration": generate_calibration,
}


def generated_configs(experiment: str, smoke: bool = False, stage: str | None = None) -> list[Path]:
    root = config_root(smoke)
    if experiment == "01_overall":
        experiment_root = root / "01_overall"
    else:
        experiment_root = root / experiment
    if not experiment_root.exists():
        return []
    paths = sorted(experiment_root.rglob("*.json"))
    if stage:
        paths = [path for path in paths if stage in path.parts]
    return paths
