"""完整 JSON 实验配置的读取、保存和模型感知校验。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

from utils.model_names import canonical_model_name

COMMON_REQUIRED_SECTIONS = (
    "paths", "data", "project", "mask", "graph", "model", "training", "evaluation",
)
VALID_VARIANTS = {"direct", "temporal_only", "relational_only", "fixed_fusion", "full"}
VALID_MISSING_TYPES = {
    "random", "independent_temporal_block", "temporal_block", "system_temporal_block",
    "persistent_od", "od_pair_block", "city_level", "city_level_block",
    "mar_weather", "mar_calendar", "mar_distance",
    "mnar_high_flow", "mnar_low_flow", "native_like",
}
FROZEN_RELATIONS = ["shared_origin", "shared_destination", "reverse", "geo", "hsr", "socio"]
FROZEN_TEMPORAL_SCALES = [3, 7, 15]


def _positive_int(section: dict[str, Any], key: str, location: str, allow_missing: bool = False) -> None:
    if allow_missing and key not in section:
        return
    if int(section.get(key, 0)) <= 0:
        raise ValueError(f"{key} 必须大于 0：{location}")


def validate_config(data: dict[str, Any], source: str | Path = "<memory>") -> None:
    """按模型类型校验配置，避免 baseline 被 MRDP-OD 专属规则拒绝。"""
    location = str(source)
    missing = [section for section in COMMON_REQUIRED_SECTIONS if section not in data]
    if missing:
        raise ValueError(f"配置缺少必要分区 {missing}：{location}")

    result_folder = data["paths"].get("result_folder")
    if result_folder is not None:
        folder_text = str(result_folder).strip()
        folder_path = Path(folder_text)
        if (
            not folder_text
            or folder_text == "."
            or folder_path.is_absolute()
            or ".." in folder_path.parts
        ):
            raise ValueError(
                "paths.result_folder 必须是 Results 下的安全相对路径："
                f"{location}"
            )
    for path_key in ("result_domain", "run_subdir"):
        path_value = data["paths"].get(path_key)
        if path_value is None:
            continue
        path_text = str(path_value).strip()
        path_object = Path(path_text)
        if not path_text or path_text == "." or path_object.is_absolute() or ".." in path_object.parts:
            raise ValueError(f"paths.{path_key} 必须是安全相对路径：{location}")

    model_name = canonical_model_name(data["model"].get("name", "MRDP-OD"))
    data["model"]["name"] = model_name
    if data["data"].get("target_transform", "log1p") != "log1p":
        raise ValueError(f"当前统一流水线只支持 log1p target transform：{location}")
    if str(data["data"].get("target_scaler", "standard")) not in {"standard", "robust"}:
        raise ValueError(f"data.target_scaler 只能是 standard 或 robust：{location}")
    train_fraction = float(data["data"].get("train_fraction", 1.0))
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"data.train_fraction 必须位于 (0,1]：{location}")
    feature_dropout = data["data"].get("feature_dropout") or {}
    for feature_group in ("context", "pair_static", "city_static"):
        fraction = float(feature_dropout.get(feature_group, 0.0))
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"data.feature_dropout.{feature_group} 必须位于 [0,1)：{location}")

    missing_type = str(data["mask"].get("missing_type", ""))
    if missing_type not in VALID_MISSING_TYPES:
        raise ValueError(f"未知缺失机制 {missing_type!r}：{location}")
    missing_rate = float(data["mask"].get("missing_rate", 0.5))
    if not 0.0 < missing_rate < 1.0:
        raise ValueError(f"mask.missing_rate 必须位于 (0,1)：{location}")
    train_mix = data["mask"].get("train_mix")
    if train_mix:
        unknown = set(train_mix) - VALID_MISSING_TYPES
        if unknown:
            raise ValueError(f"train_mix 含未知缺失机制 {sorted(unknown)}：{location}")
        if sum(float(value) for value in train_mix.values()) <= 0:
            raise ValueError(f"train_mix 权重和必须大于 0：{location}")
    train_scenarios = data["mask"].get("train_scenarios")
    if train_scenarios:
        if not isinstance(train_scenarios, list):
            raise ValueError(f"mask.train_scenarios 必须是列表：{location}")
        total_weight = 0.0
        names: set[str] = set()
        for item in train_scenarios:
            if not isinstance(item, dict):
                raise ValueError(f"train_scenarios 的每一项必须是对象：{location}")
            name = str(item.get("name", "")).strip()
            if not name or name in names:
                raise ValueError(f"train_scenarios 场景名缺失或重复：{location}")
            names.add(name)
            kind = str(item.get("missing_type", ""))
            rate = float(item.get("missing_rate", 0.0))
            weight = float(item.get("weight", 0.0))
            if kind not in VALID_MISSING_TYPES or not 0.0 < rate < 1.0 or weight <= 0.0:
                raise ValueError(f"train_scenarios 中存在无效机制、缺失率或权重：{location}")
            total_weight += weight
        if total_weight <= 0.0:
            raise ValueError(f"train_scenarios 权重和必须大于 0：{location}")

    graph_relations = list(data["graph"].get("relations", []))
    if graph_relations and any(name not in FROZEN_RELATIONS for name in graph_relations):
        raise ValueError(f"graph.relations 含未知关系：{location}")

    training = data["training"]
    _positive_int(training, "batch_size", location)
    _positive_int(training, "epochs", location)
    _positive_int(training, "validation_interval", location, allow_missing=True)
    if int(training.get("grad_accumulation_steps", 1)) <= 0:
        raise ValueError(f"grad_accumulation_steps 必须大于 0：{location}")
    evaluation = data["evaluation"]
    _positive_int(evaluation, "n_samples", location)
    _positive_int(evaluation, "validation_n_samples", location, allow_missing=True)
    if not isinstance(evaluation.get("save_full_samples", True), bool):
        raise ValueError(f"evaluation.save_full_samples 必须是布尔值：{location}")
    validation_scenarios = evaluation.get("validation_scenarios")
    if validation_scenarios:
        if not isinstance(validation_scenarios, list):
            raise ValueError(f"evaluation.validation_scenarios 必须是列表：{location}")
        for item in validation_scenarios:
            if not isinstance(item, dict):
                raise ValueError(f"validation_scenarios 的每一项必须是对象：{location}")
            if str(item.get("missing_type", "")) not in VALID_MISSING_TYPES:
                raise ValueError(f"validation_scenarios 包含未知缺失机制：{location}")
            if not 0.0 < float(item.get("missing_rate", 0.0)) < 1.0:
                raise ValueError(f"validation_scenarios 包含无效缺失率：{location}")
    validation_metric = str(evaluation.get("validation_metric", "MAE")).upper()
    if validation_metric not in {"MAE", "RMSE", "WMAPE"}:
        raise ValueError(f"evaluation.validation_metric 仅支持 MAE、RMSE 或 WMAPE：{location}")

    protocol_version = str(data.get("protocol_version", ""))
    role = str(data.get("search", {}).get("role", ""))
    unified_roles = {
        "tuning", "final_train", "final", "ablation_train", "ablation",
        "calibration", "transfer_train", "transfer_test", "analysis",
    }
    if protocol_version == "multi_region_probabilistic_imputation" and role in unified_roles:
        if not str(data.get("search", {}).get("search_space_hash", "")).strip():
            raise ValueError(f"统一协议配置必须声明 search.search_space_hash：{location}")
    if protocol_version == "multi_region_probabilistic_imputation" and role in {
        "final", "ablation", "calibration", "transfer_test"
    }:
        checkpoint = data.get("checkpoint", {})
        if not checkpoint.get("training_config") or not checkpoint.get("training_config_hash"):
            raise ValueError(f"统一协议的测试配置必须声明训练配置和哈希：{location}")

    if model_name == "MRDP-OD":
        diffusion = data.get("diffusion")
        if not isinstance(diffusion, dict):
            raise ValueError(f"{model_name} 配置必须包含 diffusion 分区：{location}")
        _positive_int(diffusion, "num_steps", location)
        beta_start = float(diffusion.get("beta_start", 0.0))
        beta_end = float(diffusion.get("beta_end", 0.0))
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError(f"beta 必须满足 0 < beta_start < beta_end < 1：{location}")

    if model_name == "MRDP-OD":
        model = data["model"]
        variant = str(model.get("variant", "full"))
        if variant not in VALID_VARIANTS:
            raise ValueError(f"未知模型变体 {variant!r}：{location}")
        scales = [int(value) for value in model.get("temporal_scales", FROZEN_TEMPORAL_SCALES)]
        if len(scales) != 3 or any(value <= 0 for value in scales):
            raise ValueError(f"model.temporal_scales 必须包含三个正整数：{location}")
        scale_sensitivity = model.get("allow_temporal_scale_sensitivity", False)
        if not isinstance(scale_sensitivity, bool):
            raise ValueError(
                f"model.allow_temporal_scale_sensitivity 必须是布尔值：{location}"
            )
        if scale_sensitivity:
            raise ValueError(
                "六个论文实验不允许修改冻结的时间先验尺度："
                f"{location}"
            )
        elif scales != FROZEN_TEMPORAL_SCALES:
            raise ValueError(
                f"非时间尺度敏感性实验必须冻结 model.temporal_scales="
                f"{FROZEN_TEMPORAL_SCALES}：{location}"
            )
        if int(model.get("router_hidden_dim", 16)) <= 0:
            raise ValueError(f"model.router_hidden_dim 必须大于 0：{location}")
        if not isinstance(model.get("use_pair_id", False), bool):
            raise ValueError(f"model.use_pair_id 必须是布尔值：{location}")
        token_mode = str(model.get("city_token_mode", "transductive"))
        if token_mode not in {"transductive", "inductive"}:
            raise ValueError(f"model.city_token_mode 只能是 transductive 或 inductive：{location}")
        if token_mode == "inductive" and bool(model.get("use_pair_id", False)):
            raise ValueError(f"inductive tokenizer 禁止 use_pair_id=true：{location}")
        relations = list(data["graph"].get("relations", []))
        if relations != FROZEN_RELATIONS:
            raise ValueError(f"MRDP-OD 的 graph.relations 必须按固定顺序包含六类关系：{location}")
        loss = data.get("loss", {})
        for name in ("rel_prior_weight", "fused_prior_weight"):
            if float(loss.get(name, 0.0)) < 0:
                raise ValueError(f"loss.{name} 不能为负：{location}")

def load_json(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"配置根节点必须是 JSON 对象：{config_path}")
    validate_config(data, config_path)
    data.setdefault("_meta", {})["source_config"] = str(config_path)
    return data


def save_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(data: dict[str, Any]) -> str:
    """忽略来源路径后计算配置哈希，用于自动跳过已完成组合。"""
    clean = json.loads(json.dumps(data, default=str))
    clean.pop("_meta", None)
    return hashlib.sha256(stable_json(clean).encode("utf-8")).hexdigest()[:20]
