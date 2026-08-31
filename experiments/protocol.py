"""集中定义实验命名、数据区域、随机种子和评估场景。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


PROTOCOL_VERSION = "multi_region_probabilistic_imputation"
REGIONS = ("Beijing_Group", "Chang_Delta", "Zhu_Delta")
MODEL_SEEDS = (2022, 2023, 2024, 2025, 2026)
TUNING_SEED = 2024
MASK_SEED = 12024
VALIDATION_SEED = 22024
TEST_SEED = 32024
VALIDATION_METRIC = "WMAPE"

# 网络案例在运行前固定；日期由只依赖测试掩码的规则确定，不依据模型误差挑选。
NETWORK_CASE = {
    "region": "Chang_Delta",
    "scenario": "persistent_50",
    "selection_rule": "max_target_count_then_earliest",
}

MODELS = ("MRDP-OD",)
PROBABILISTIC_MODELS = ("MRDP-OD",)
PATTERN_SHIFT_MODELS = ("MRDP-OD",)
INDUCTIVE_MODELS = ("MRDP-OD",)

CORE_SCENARIOS: dict[str, dict[str, Any]] = {
    "random_30": {"missing_type": "random", "missing_rate": 0.30},
    "random_50": {"missing_type": "random", "missing_rate": 0.50},
    "temporal_30": {"missing_type": "independent_temporal_block", "missing_rate": 0.30},
    "temporal_50": {"missing_type": "independent_temporal_block", "missing_rate": 0.50},
    "persistent_30": {"missing_type": "persistent_od", "missing_rate": 0.30},
    "persistent_50": {"missing_type": "persistent_od", "missing_rate": 0.50},
    "city_1": {"missing_type": "city_level", "missing_rate": 0.50, "city_count": 1},
    "city_2": {"missing_type": "city_level", "missing_rate": 0.50, "city_count": 2},
}

ROBUSTNESS_SCENARIOS: dict[str, dict[str, Any]] = {
    "mar_weather_30": {"missing_type": "mar_weather", "missing_rate": 0.30},
    "mar_weather_50": {"missing_type": "mar_weather", "missing_rate": 0.50},
    "mar_calendar_30": {"missing_type": "mar_calendar", "missing_rate": 0.30},
    "mar_calendar_50": {"missing_type": "mar_calendar", "missing_rate": 0.50},
    "mar_distance_30": {"missing_type": "mar_distance", "missing_rate": 0.30},
    "mar_distance_50": {"missing_type": "mar_distance", "missing_rate": 0.50},
    "mnar_high_30": {"missing_type": "mnar_high_flow", "missing_rate": 0.30},
    "mnar_high_50": {"missing_type": "mnar_high_flow", "missing_rate": 0.50},
    "mnar_low_30": {"missing_type": "mnar_low_flow", "missing_rate": 0.30},
    "mnar_low_50": {"missing_type": "mnar_low_flow", "missing_rate": 0.50},
    "native_like": {"missing_type": "native_like", "missing_rate": 0.30},
    "random_70": {"missing_type": "random", "missing_rate": 0.70},
    "random_80": {"missing_type": "random", "missing_rate": 0.80},
}

REPRESENTATIVE_SCENARIOS = ("random_50", "temporal_50", "persistent_50", "city_1")
VALIDATION_SCENARIOS = REPRESENTATIVE_SCENARIOS

ABLATIONS: dict[str, dict[str, Any]] = {
    "direct_diffusion": {"variant": "direct", "use_residual_diffusion": True},
    "temporal_only": {"variant": "temporal_only", "use_residual_diffusion": True},
    "relational_only": {"variant": "relational_only", "use_residual_diffusion": True},
    "fixed_fusion": {"variant": "fixed_fusion", "use_residual_diffusion": True},
    "routed_prior": {"variant": "full", "use_residual_diffusion": False},
    "full": {"variant": "full", "use_residual_diffusion": True},
    "router_no_temporal_support": {
        "variant": "full", "use_residual_diffusion": True,
        "router_feature_mask": [0, 0, 0, 1, 1, 1],
    },
    "router_no_pair_support": {
        "variant": "full", "use_residual_diffusion": True,
        "router_feature_mask": [1, 1, 1, 0, 1, 1],
    },
    "router_no_city_support": {
        "variant": "full", "use_residual_diffusion": True,
        "router_feature_mask": [1, 1, 1, 1, 0, 0],
    },
}

EXPERIMENTS = (
    "protocol_audit",
    "01_overall",
    "02_robustness",
    "03_cross_pattern",
    "04_ablation",
    "05_calibration",
    "06_network",
)


def scenario(name: str) -> dict[str, Any]:
    source = {**CORE_SCENARIOS, **ROBUSTNESS_SCENARIOS}
    if name not in source:
        raise KeyError(f"未知评估场景：{name}")
    return {"name": name, "city_count": None, **deepcopy(source[name])}


def scenario_list(names: Iterable[str]) -> list[dict[str, Any]]:
    return [scenario(name) for name in names]


def mixed_training_scenarios(names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    selected = list(names or CORE_SCENARIOS)
    weight = 1.0 / len(selected)
    return [{**scenario(name), "weight": weight} for name in selected]


def model_folder_name(model_name: str) -> str:
    return str(model_name).replace(" ", "_")
