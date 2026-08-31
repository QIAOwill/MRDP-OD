"""读取 JSON 搜索空间、展开参数组合并构造模型基础配置。"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any
import json

from utils.config import stable_json
from .protocol import (
    MASK_SEED,
    PROTOCOL_VERSION,
    TUNING_SEED,
    VALIDATION_METRIC,
    VALIDATION_SCENARIOS,
    VALIDATION_SEED,
    TEST_SEED,
    mixed_training_scenarios,
    scenario_list,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
SEARCH_SPACE_PATH = CODE_ROOT / "configs" / "search_spaces.json"
SEARCH_MODELS = ("MRDP-OD",)
MEMORY_HEAVY_MODELS: set[str] = set()
MEMORY_SAFE_BATCH_SIZE = 4


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """根据 ``a.b.c`` 路径写入配置，并拒绝不存在的参数路径。"""
    node: Any = config
    parts = dotted_key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"非法参数路径：{dotted_key!r}")
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"搜索参数路径不存在：{dotted_key}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise KeyError(f"搜索参数路径不存在：{dotted_key}")
    node[parts[-1]] = deepcopy(value)


def _candidate_values(value: Any, dotted_key: str) -> list[Any]:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"搜索参数 {dotted_key} 的候选列表不能为空")
        return deepcopy(value)
    return [deepcopy(value)]


def cartesian_grid(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(spec, dict):
        raise TypeError("搜索参数必须是 JSON object")
    if not spec:
        return [{}]
    keys = list(spec)
    value_lists = [_candidate_values(spec[key], key) for key in keys]
    return [dict(zip(keys, values)) for values in product(*value_lists)]


def _normalise_batch_policy(model_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """将有效批量换算为物理批量和梯度累积次数。"""
    candidate = deepcopy(raw)
    effective_key = "training.effective_batch_size"
    batch_key = "training.batch_size"
    accumulation_key = "training.grad_accumulation_steps"
    if effective_key in candidate:
        if batch_key in candidate or accumulation_key in candidate:
            raise ValueError(
                f"{model_name} 不能同时搜索有效批量与物理批量/梯度累积次数"
            )
        effective = int(candidate[effective_key])
        if effective <= 0:
            raise ValueError("training.effective_batch_size 必须大于 0")
        physical = MEMORY_SAFE_BATCH_SIZE if model_name in MEMORY_HEAVY_MODELS else effective
        if effective % physical != 0:
            raise ValueError(
                f"{model_name} 的有效批量 {effective} 必须能被物理批量 {physical} 整除"
            )
        candidate[effective_key] = effective
        candidate[batch_key] = physical
        candidate[accumulation_key] = effective // physical
    elif batch_key in candidate:
        physical = int(candidate[batch_key])
        accumulation = int(candidate.get(accumulation_key, 1))
        if physical <= 0 or accumulation <= 0:
            raise ValueError("batch_size 和 grad_accumulation_steps 必须大于 0")
        if model_name in MEMORY_HEAVY_MODELS and physical > MEMORY_SAFE_BATCH_SIZE:
            raise ValueError(
                f"{model_name} 是显存敏感模型，请搜索 training.effective_batch_size"
            )
        candidate[batch_key] = physical
        candidate[accumulation_key] = accumulation
        candidate[effective_key] = physical * accumulation
    return candidate


def expand_search_space(model_name: str, model_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """展开共享参数与成组参数的笛卡尔积，并删除重复组合。"""
    if not isinstance(model_spec, dict):
        raise TypeError(f"{model_name} 的搜索空间必须是 JSON object")
    unknown_sections = set(model_spec) - {"parameters", "candidate_groups"}
    if unknown_sections:
        raise ValueError(f"{model_name} 搜索空间含未知字段：{sorted(unknown_sections)}")
    shared_candidates = cartesian_grid(model_spec.get("parameters", {}))
    group_specs = model_spec.get("candidate_groups") or [{}]
    if not isinstance(group_specs, list) or not group_specs:
        raise ValueError(f"{model_name}.candidate_groups 必须是非空 JSON list")
    group_candidates: list[dict[str, Any]] = []
    for index, group_spec in enumerate(group_specs, 1):
        if not isinstance(group_spec, dict):
            raise TypeError(f"{model_name}.candidate_groups[{index}] 必须是 JSON object")
        group_candidates.extend(cartesian_grid(group_spec))
    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group_values in group_candidates:
        for shared_values in shared_candidates:
            overlap = set(group_values) & set(shared_values)
            if overlap:
                raise ValueError(
                    f"{model_name} 的 parameters 与 candidate_groups 重复定义：{sorted(overlap)}"
                )
            candidate = _normalise_batch_policy(
                model_name, {**deepcopy(group_values), **deepcopy(shared_values)}
            )
            fingerprint = stable_json(candidate)
            if fingerprint not in seen:
                seen.add(fingerprint)
                expanded.append(candidate)
    if not expanded:
        raise ValueError(f"{model_name} 的搜索空间没有产生任何组合")
    return expanded


def read_search_spaces(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """读取搜索空间并检查模型覆盖是否完整。"""
    source = SEARCH_SPACE_PATH if path is None else Path(path)
    if not source.exists():
        raise FileNotFoundError(f"搜索空间 JSON 不存在：{source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"{source} 必须包含 JSON object：models")
    missing = set(SEARCH_MODELS) - set(models)
    unknown = set(models) - set(SEARCH_MODELS)
    if missing or unknown:
        raise ValueError(f"搜索空间模型不完整；缺少={sorted(missing)}，未知={sorted(unknown)}")
    return deepcopy(models)


def base_config(model_name: str) -> dict[str, Any]:
    """构造模型单次运行的基础配置，搜索参数随后覆盖其中的标量。"""
    cfg: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "paths": {
            "dataset_dir": "sample_data",
            "results_dir": "outputs",
            "result_folder": "_unassigned",
        },
        "data": {
            "region": "Chang_Delta",
            "target_transform": "log1p",
            "split": {"mode": "ratio", "ratio": [0.7, 0.1, 0.2]},
            "window_len": 30,
            "stride": 7,
            "split_stride": {"train": 7, "valid": 14, "test": 30},
            "max_windows": {"train": None, "valid": None, "test": None},
            "city_limit": None,
        },
        "project": {
            "name": model_name,
            "seed": TUNING_SEED,
            "device": "auto",
            "num_workers": 0,
            "cpu_threads": 16,
            "deterministic": True,
        },
        "mask": {
            "missing_type": "random",
            "missing_rate": 0.5,
            "temporal_block_len": 7,
            "city_count": None,
            "seed": MASK_SEED,
            "train_mix": None,
            "train_scenarios": mixed_training_scenarios(),
        },
        "graph": {
            "relations": ["shared_origin", "shared_destination", "reverse", "geo", "hsr", "socio"],
            "baseline_relations": ["shared_origin", "shared_destination", "reverse", "geo"],
            "topk": 24,
            "self_loop": True,
            "similarity_sigma": 1.0,
        },
        "model": {"name": model_name},
        "diffusion": {
            "num_steps": 50,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule": "quadratic",
        },
        "loss": {},
        "training": {
            "batch_size": 8,
            "effective_batch_size": 8,
            "epochs": 100,
            "lr": 0.001,
            "min_lr": 1e-5,
            "lr_factor": 0.5,
            "lr_patience": 5,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "grad_accumulation_steps": 1,
            "patience": 15,
            "max_train_batches": None,
            "max_valid_batches": None,
            "amp": {"enabled": True, "dtype": "bfloat16"},
        },
        "evaluation": {
            "n_samples": 20,
            "validation_n_samples": 1,
            "validation_metric": VALIDATION_METRIC,
            "validation_seed": VALIDATION_SEED,
            "test_seed": TEST_SEED,
            "validation_scenarios": scenario_list(VALIDATION_SCENARIOS),
            "max_test_batches": None,
            "save_full_samples": True,
        },
        "search": {},
    }
    defaults = {
        "model": {
            "variant": "full", "hidden_dim": 96, "prior_hidden_dim": 64,
            "time_emb_dim": 96, "pair_emb_dim": 32, "num_layers": 2,
            "prior_graph_layers": 1, "dropout": 0.1, "use_direction": True,
            "use_pair_id": False, "temporal_scales": [3, 7, 15],
            "router_hidden_dim": 16,
        },
        "loss": {"rel_prior_weight": 0.2, "fused_prior_weight": 0.2},
    }
    for section, values in defaults.items():
        cfg.setdefault(section, {}).update(deepcopy(values))
    if model_name in MEMORY_HEAVY_MODELS:
        cfg["training"].update({
            "batch_size": MEMORY_SAFE_BATCH_SIZE,
            "grad_accumulation_steps": 2,
            "effective_batch_size": 8,
        })
    return cfg
