"""从冻结测试预测执行 06_network 网络恢复与下游效用实验。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from analysis.results import collect_standard_runs, experiment_results_root
from data.reader import read_region
from data.tensor_builder import build_region_tensor
from evaluation.downstream import downstream_forecast_metrics
from evaluation.network_metrics import network_recovery_metrics
from training.experiment import resolve_project_path
from utils.config import stable_json
from utils.io import write_json


NETWORK_SCENARIOS = ("persistent_50", "city_1", "mnar_high_30")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _arrays(run_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(run_dir) / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as values:
        return {key: values[key] for key in values.files}


def _region_data(row: pd.Series, cache: dict[str, Any]):
    cfg = _json(Path(row["run_dir"]) / "config_resolved.json")
    key = stable_json({"region": cfg["data"]["region"], "data": cfg["data"], "graph": cfg["graph"]})
    if key not in cache:
        raw = read_region(resolve_project_path(cfg["paths"]["dataset_dir"]), cfg["data"]["region"])
        cache[key] = build_region_tensor(cfg["data"]["region"], raw, cfg)
    return cache[key]


def _gap_closed(frame: pd.DataFrame) -> pd.DataFrame:
    """相对 raw-incomplete 到 oracle 的可关闭差距，计算下游收益比例。"""
    if frame.empty:
        return frame
    result = frame.copy()
    keys = ["region", "seed", "scenario"]
    for metric in ("FORECAST_WMAPE", "FORECAST_RMSE"):
        result[f"{metric}_GAP_CLOSED"] = np.nan
        for _, indices in result.groupby(keys, dropna=False).groups.items():
            part = result.loc[indices]
            raw = part.loc[part["model"].eq("Raw incomplete"), metric]
            oracle = part.loc[part["model"].eq("Oracle"), metric]
            if raw.empty or oracle.empty:
                continue
            denominator = float(raw.iloc[0] - oracle.iloc[0])
            if abs(denominator) <= 1e-12:
                continue
            result.loc[indices, f"{metric}_GAP_CLOSED"] = (
                float(raw.iloc[0]) - result.loc[indices, metric]
            ) / denominator
    return result


def run_network_downstream_experiment(smoke: bool = False) -> Path:
    """执行 06_network；只读取已经完成的 01_overall/02_robustness 测试结果，不接触训练或测试集选参。"""
    runs = collect_standard_runs(smoke)
    selected = runs[
        runs.get("scenario", pd.Series(dtype=str)).isin(NETWORK_SCENARIOS)
        & runs.get("role", pd.Series(dtype=str)).eq("final")
        & runs.get("experiment", pd.Series(dtype=str)).isin(["01_overall", "02_robustness"])
    ].copy()
    if not selected.empty:
        selected = selected[
            (selected["experiment"] == "01_overall")
            | selected["checkpoint_training_config"].astype(str).str.contains("01_overall_main")
        ]
    if selected.empty:
        raise RuntimeError("06_network 未找到 01_overall/02_robustness 的 persistent_50、city_1 或 mnar_high_30 冻结测试输出")

    rows: list[dict[str, Any]] = []
    cache: dict[str, Any] = {}
    reference_keys: set[tuple[str, int, str]] = set()
    for _, row in selected.iterrows():
        arrays = _arrays(row["run_dir"])
        region_data = _region_data(row, cache)
        base = {
            "model": row["model"], "region": row["region"], "seed": int(row["seed"]),
            "scenario": row["scenario"], "source_run": row["run_dir"],
        }
        rows.append({
            **base,
            **network_recovery_metrics(region_data, arrays),
            **downstream_forecast_metrics(region_data, arrays),
        })
        key = (str(row["region"]), int(row["seed"]), str(row["scenario"]))
        if key in reference_keys:
            continue
        reference_keys.add(key)
        for name, prediction in (
            ("Raw incomplete", np.zeros_like(arrays["y_true"])),
            ("Oracle", np.asarray(arrays["y_true"]).copy()),
        ):
            reference_arrays = dict(arrays)
            reference_arrays["y_pred"] = prediction
            rows.append({
                "model": name, "region": row["region"], "seed": int(row["seed"]),
                "scenario": row["scenario"], "source_run": row["run_dir"],
                **network_recovery_metrics(region_data, reference_arrays),
                **downstream_forecast_metrics(region_data, reference_arrays),
            })

    result = _gap_closed(pd.DataFrame(rows))
    output = experiment_results_root(smoke) / "06_network"
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "network_downstream_by_seed.tsv", sep="\t", index=False, encoding="utf-8")
    metrics = [
        column for column in result
        if column.startswith("NETWORK_") or column.startswith("TOP_") or column.startswith("FORECAST_")
    ]
    summary = result.groupby(["model", "region", "scenario"], dropna=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "network_downstream_summary.tsv", sep="\t", index=False, encoding="utf-8")

    case_rows: list[dict[str, Any]] = []
    for manifest_path in experiment_results_root(smoke).rglob("network_case_manifest.json"):
        manifest = _json(manifest_path)
        run_dir = manifest_path.parent
        cfg = _json(run_dir / "config_resolved.json")
        case_rows.append({
            "model": cfg.get("model", {}).get("name"),
            "seed": cfg.get("project", {}).get("seed"),
            "scenario": cfg.get("search", {}).get("scenario"),
            **manifest,
            "run_dir": str(run_dir),
        })
    pd.DataFrame(case_rows).to_csv(output / "network_case_inventory.tsv", sep="\t", index=False, encoding="utf-8")
    status = {
        "status": "completed",
        "source_completed_runs": int(len(selected)),
        "by_seed_rows": int(len(result)),
        "case_snapshot_rows": int(len(case_rows)),
        "scenarios": list(NETWORK_SCENARIOS),
    }
    write_json(status, output / "run_status.json")
    return output
