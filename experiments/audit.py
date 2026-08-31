"""数据、掩码、预处理泄漏边界和模型能力审计。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from data.dataset import ODWindowDataset
from data.reader import read_region
from data.tensor_builder import build_region_tensor, data_manifest, scaler_metadata
from models.factory import MODEL_CLASSES
from training.experiment import resolve_project_path
from utils.io import write_json
from utils.model_names import canonical_model_name
from .configs import experiment_base_config
from .protocol import CORE_SCENARIOS, MODELS, REGIONS, ROBUSTNESS_SCENARIOS, scenario


PRIMARY_KEYS = {
    "od": ["date", "origin_id", "destination_id"],
    "city_dynamic": ["date", "city_id"],
    "pair_weather": ["date", "origin_id", "destination_id"],
    "city_static": ["city_id"],
    "pair_static": ["origin_id", "destination_id"],
}


def _native_run_lengths(mask: np.ndarray) -> np.ndarray:
    lengths: list[int] = []
    missing = np.asarray(mask) <= 0
    for pair in range(missing.shape[1]):
        start = None
        for index, value in enumerate(np.r_[missing[:, pair], False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                lengths.append(index - start)
                start = None
    return np.asarray(lengths, dtype=np.int64)


def _mask_audit(region_data, cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenarios = {**CORE_SCENARIOS, **ROBUSTNESS_SCENARIOS}
    for name in scenarios:
        spec = scenario(name)
        dataset = ODWindowDataset(region_data, "test", cfg, mask_spec=spec, scenario_name=name)
        rates: list[float] = []
        target_count = 0
        conditional_count = 0
        overlap_count = 0
        for index in range(len(dataset)):
            item = dataset[index]
            target = item["target_mask"].numpy()
            cond = item["cond_mask"].numpy()
            native = item["native_mask"].numpy()
            rates.append(float(item["actual_missing_rate"]))
            target_count += int(target.sum())
            conditional_count += int(cond.sum())
            overlap_count += int((target * cond).sum())
            if np.any(target > native) or np.any(cond > native):
                raise RuntimeError(f"{region_data.region}/{name} mask 超出 native observation")
        rows.append({
            "region": region_data.region,
            "scenario": name,
            "missing_type": spec["missing_type"],
            "configured_rate": spec["missing_rate"],
            "actual_rate_mean": float(np.mean(rates)),
            "actual_rate_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
            "windows": len(dataset),
            "target_count_across_windows": target_count,
            "conditional_count_across_windows": conditional_count,
            "target_condition_overlap": overlap_count,
            "passed": overlap_count == 0,
        })
    return pd.DataFrame(rows)


def _capability_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in MODELS:
        cls = MODEL_CLASSES[canonical_model_name(name)]
        rows.append({
            "model": name,
            "probabilistic": bool(getattr(cls, "is_probabilistic", False)),
            "requires_training": bool(getattr(cls, "requires_training", True)),
            "supports_unseen_od": bool(getattr(cls, "supports_unseen_od", False)),
            "supports_unseen_city": bool(getattr(cls, "supports_unseen_city", False)),
            "supports_cross_region": bool(getattr(cls, "supports_cross_region", False)),
            "cross_region_reporting": (
                "eligible" if bool(getattr(cls, "supports_cross_region", False))
                else "N.A. (transductive identity or fixed graph size)"
            ),
        })
    return pd.DataFrame(rows)


def run_protocol_audit(smoke: bool = False) -> Path:
    cfg = experiment_base_config("MRDP-OD", "protocol_audit", smoke)
    dataset_dir = resolve_project_path(cfg["paths"]["dataset_dir"])
    results_root = resolve_project_path(cfg["paths"]["results_dir"])
    output = results_root / ("_smoke" if smoke else "") / "protocol_audit"
    output.mkdir(parents=True, exist_ok=True)
    region_rows: list[dict[str, Any]] = []
    mask_frames: list[pd.DataFrame] = []
    schemas: dict[str, dict[str, list[str]]] = {}
    for region in REGIONS:
        raw = read_region(dataset_dir, region)
        duplicates = {
            table: int(raw[table].duplicated(keys).sum())
            for table, keys in PRIMARY_KEYS.items()
        }
        local_cfg = json.loads(json.dumps(cfg))
        local_cfg["data"]["region"] = region
        region_data = build_region_tensor(region, raw, local_cfg)
        manifest = data_manifest(region_data)
        runs = _native_run_lengths(region_data.native_mask)
        row = {
            **manifest,
            "native_observed": int(region_data.native_mask.sum()),
            "native_missing": int(region_data.native_mask.size - region_data.native_mask.sum()),
            "native_observation_rate": float(region_data.native_mask.mean()),
            "native_missing_run_mean": float(runs.mean()) if runs.size else 0.0,
            "native_missing_run_p95": float(np.quantile(runs, 0.95)) if runs.size else 0.0,
            "duplicate_primary_keys": int(sum(duplicates.values())),
            "split_overlap": int(
                len(set(region_data.train_idx) & set(region_data.valid_idx))
                + len(set(region_data.train_idx) & set(region_data.test_idx))
                + len(set(region_data.valid_idx) & set(region_data.test_idx))
            ),
        }
        region_rows.append(row)
        schemas[region] = {
            "context": region_data.context_feature_names,
            "pair_static": region_data.pair_feature_names,
            "city_static": region_data.city_feature_names,
        }
        region_dir = output / region
        region_dir.mkdir(parents=True, exist_ok=True)
        write_json({"manifest": manifest, "duplicates": duplicates}, region_dir / "dataset_manifest.json")
        write_json(scaler_metadata(region_data), region_dir / "scaler_metadata.json")
        mask_frame = _mask_audit(region_data, local_cfg)
        mask_frame.to_csv(region_dir / "mask_validation.tsv", sep="\t", index=False, encoding="utf-8")
        mask_frames.append(mask_frame)
    schema_reference = schemas[REGIONS[0]]
    schema_consistent = all(value == schema_reference for value in schemas.values())
    write_json({"schema_consistent": schema_consistent, "schemas": schemas}, output / "feature_schema_audit.json")
    pd.DataFrame(region_rows).to_csv(output / "dataset_summary.tsv", sep="\t", index=False, encoding="utf-8")
    pd.concat(mask_frames, ignore_index=True).to_csv(
        output / "mask_validation_all.tsv", sep="\t", index=False, encoding="utf-8"
    )
    capabilities = _capability_table()
    capabilities.to_csv(output / "baseline_capabilities.tsv", sep="\t", index=False, encoding="utf-8")
    passed = bool(
        schema_consistent
        and all(row["duplicate_primary_keys"] == 0 and row["split_overlap"] == 0 for row in region_rows)
        and all(bool(frame["passed"].all()) for frame in mask_frames)
    )
    report = [
        "# 数据与实验协议审计",
        "",
        f"- 总体状态：{'PASS' if passed else 'FAIL'}",
        f"- 三区域特征 schema 一致：{schema_consistent}",
        "- scaler 由各训练日期拟合；cross-region 另使用 source-only scaler。",
        "- evaluation target 与 native query 已分离，原生未知位置不参与监督精度。",
        "- baseline 的 unseen/cross-region 不支持项按能力表报告 N.A.，不伪造迁移结果。",
        "",
        "详细数值见 dataset_summary.tsv、mask_validation_all.tsv、feature_schema_audit.json 和 baseline_capabilities.tsv。",
    ]
    (output / "AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json({"passed": passed, "output": str(output)}, output / "audit_status.json")
    if not passed:
        raise RuntimeError(f"protocol_audit protocol audit 未通过，请查看 {output}")
    return output
