"""将原始运行整理为校准、网络、下游、路由、统计和效率结果。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

import numpy as np
import pandas as pd

from data.reader import read_region
from data.tensor_builder import build_region_tensor
from evaluation.calibration import calibrated_interval_metrics, fit_quantile_scaling
from evaluation.downstream import downstream_forecast_metrics
from evaluation.network_metrics import network_recovery_metrics
from evaluation.statistics import holm_adjust, paired_bootstrap_difference
from training.experiment import resolve_project_path
from utils.config import stable_json
from utils.io import write_json
from experiments.configs import experiment_base_config
from experiments.protocol import (
    MODELS, REGIONS, REPRESENTATIVE_SCENARIOS, ROBUSTNESS_SCENARIOS,
)


def experiment_results_root(smoke: bool = False) -> Path:
    cfg = experiment_base_config("MRDP-OD", "protocol_audit", smoke)
    root = resolve_project_path(cfg["paths"]["results_dir"])
    return root / "_smoke" if smoke else root


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def collect_standard_runs(smoke: bool = False) -> pd.DataFrame:
    root = experiment_results_root(smoke)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()
    for status_path in root.rglob("run_status.json"):
        relative = status_path.relative_to(root)
        if not smoke and relative.parts and relative.parts[0].startswith("_"):
            continue
        status = _json(status_path)
        if status.get("status") != "completed":
            continue
        run_dir = status_path.parent
        cfg = _json(run_dir / "config_resolved.json")
        if not cfg or cfg.get("search", {}).get("role") == "transfer_train":
            continue
        metrics = _json(run_dir / "metrics.json")
        runtime = _json(run_dir / "runtime.json")
        metadata = _json(run_dir / "run_metadata.json")
        model_summary = _json(run_dir / "model_summary.json")
        rows.append({
            "run_dir": str(run_dir),
            # The Results folder is the canonical experiment identity. This
            # also keeps renamed historical runs readable without legacy IDs.
            "experiment": relative.parts[0],
            "experiment_name": relative.parts[0],
            "stage": cfg.get("experiment", {}).get("stage"),
            "model": cfg.get("model", {}).get("name"),
            "region": cfg.get("data", {}).get("region"),
            "seed": cfg.get("project", {}).get("seed"),
            "role": cfg.get("search", {}).get("role"),
            "scenario": cfg.get("search", {}).get("scenario"),
            "grid_id": cfg.get("search", {}).get("grid_id"),
            "variant": cfg.get("model", {}).get("variant"),
            "experiment_variant": cfg.get("experiment", {}).get("variant_name"),
            "calibration_target_experiment": cfg.get("experiment", {}).get("calibration_target_experiment"),
            "calibration_target_variant": cfg.get("experiment", {}).get("calibration_target_variant"),
            "calibration_target_scenario": cfg.get("experiment", {}).get("calibration_target_scenario"),
            "run_subdir": cfg.get("paths", {}).get("run_subdir"),
            "checkpoint_training_config": cfg.get("checkpoint", {}).get("training_config"),
            **{f"PARAM_{key}": value for key, value in model_summary.items()},
            **metrics,
            **runtime,
            "parameter_count": metadata.get("parameter_count"),
            "evaluation_samples": metadata.get("evaluation_samples", runtime.get("n_samples")),
        })
    return pd.DataFrame(rows)


def _standard_summaries(runs: pd.DataFrame, output: Path) -> None:
    if runs.empty:
        return
    completed = runs[runs["role"].isin(["final", "ablation"])].copy()
    metric_candidates = (
        "MAE", "RMSE", "WMAPE", "TOP20_WMAPE", "CPC", "CRPS", "NCRPS", "WIS",
        "CALIBRATION_ERROR", "PICP50", "MPIW50", "PICP80", "MPIW80",
        "PICP90", "MPIW90", "PICP95", "MPIW95", "AURC",
        "GATE_MEAN", "TEMPORAL_PRIOR_MAE", "RELATIONAL_PRIOR_MAE", "FUSED_PRIOR_MAE",
        "parameter_count", "training_seconds", "test_seconds", "peak_gpu_memory_mb",
        "inference_benchmark_seconds_mean", "inference_benchmark_seconds_sample_sd",
        "inference_benchmark_seconds_per_window", "inference_benchmark_windows_per_second",
        "inference_benchmark_peak_gpu_memory_mb",
    )
    metrics = [name for name in metric_candidates if name in completed]
    groups = [
        name for name in ("experiment", "model", "region", "experiment_variant", "scenario")
        if name in completed
    ]
    if completed.empty or not metrics:
        return
    summary = completed.groupby(groups, dropna=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "all_experiments_mean_sample_sd.tsv", sep="\t", index=False, encoding="utf-8")
    for experiment, part in summary.groupby("experiment", dropna=False):
        part.to_csv(output / f"{experiment}_summary.tsv", sep="\t", index=False, encoding="utf-8")


def _prediction_arrays(run_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(run_dir) / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as values:
        return {key: values[key] for key in values.files}


def _calibration_analysis(runs: pd.DataFrame, output: Path) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    validation = runs[(runs["experiment"] == "05_calibration") & (runs["role"] == "calibration")]
    tests = runs[runs["role"].isin(["final", "ablation"])]
    rows: list[dict[str, Any]] = []
    for _, val in validation.iterrows():
        calibration_scenario = str(val["scenario"])
        target_scenario = val.get("calibration_target_scenario")
        scenario_name = (
            calibration_scenario
            if target_scenario is None or pd.isna(target_scenario)
            else str(target_scenario)
        )
        candidates = tests[
            (tests["model"] == val["model"])
            & (tests["region"] == val["region"])
            & (tests["seed"] == val["seed"])
            & (tests["scenario"] == scenario_name)
        ].copy()
        target_experiment = val.get("calibration_target_experiment")
        if target_experiment is not None and not pd.isna(target_experiment):
            candidates = candidates[candidates["experiment"].eq(str(target_experiment))]
        target_variant = val.get("calibration_target_variant")
        if target_variant is not None and not pd.isna(target_variant):
            candidates = candidates[candidates["experiment_variant"].eq(str(target_variant))]
        if candidates.empty:
            continue
        # 优先 01_overall 主测试；MNAR 则优先直接复用 01_overall checkpoint 的 02_robustness test-only 结果。
        candidates["priority"] = candidates.apply(
            lambda row: 0 if row["experiment"] == "01_overall" else (
                1 if "01_overall_main" in str(row.get("checkpoint_training_config")) else 2
            ), axis=1,
        )
        test = candidates.sort_values(["priority", "run_dir"]).iloc[0]
        val_arrays = _prediction_arrays(val["run_dir"])
        test_arrays = _prediction_arrays(test["run_dir"])
        factors = fit_quantile_scaling(val_arrays["y_true"], val_arrays["samples"])
        calibrated = calibrated_interval_metrics(test_arrays["y_true"], test_arrays["samples"], factors)
        rows.append({
            "model": val["model"], "region": val["region"], "seed": val["seed"],
            "scenario": scenario_name, "calibration_scenario": calibration_scenario,
            "shift_variant": None if target_variant is None or pd.isna(target_variant) else str(target_variant),
            "validation_run": val["run_dir"], "test_run": test["run_dir"],
            **{f"SCALE_{level}": factor for level, factor in factors.items()},
            "RAW_CRPS": test.get("CRPS"), "RAW_NCRPS": test.get("NCRPS"),
            "RAW_WIS": test.get("WIS"), "RAW_CALIBRATION_ERROR": test.get("CALIBRATION_ERROR"),
            **{f"RAW_{name}": test.get(name) for name in (
                "PICP50", "MPIW50", "PICP80", "MPIW80", "PICP90", "MPIW90", "PICP95", "MPIW95"
            )},
            **calibrated,
            # interval-only quantile scaling 不改变样本分布本身，故 CRPS/nCRPS 不伪造“校准后”值。
            "CAL_CRPS": test.get("CRPS"), "CAL_NCRPS": test.get("NCRPS"),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(output / "05_calibration_by_seed.tsv", sep="\t", index=False, encoding="utf-8")
    return result


def _region_data_for_run(row: pd.Series, cache: dict[str, Any]):
    cfg = _json(Path(row["run_dir"]) / "config_resolved.json")
    key = stable_json({"region": cfg["data"]["region"], "data": cfg["data"], "graph": cfg["graph"]})
    if key not in cache:
        dataset_dir = resolve_project_path(cfg["paths"]["dataset_dir"])
        raw = read_region(dataset_dir, cfg["data"]["region"])
        cache[key] = build_region_tensor(cfg["data"]["region"], raw, cfg)
    return cache[key]


def _network_downstream_analysis(runs: pd.DataFrame, output: Path) -> pd.DataFrame:
    selected = runs[
        runs["scenario"].isin(["persistent_50", "city_1", "mnar_high_30"])
        & runs["role"].eq("final")
        & runs["experiment"].isin(["01_overall", "02_robustness"])
    ].copy()
    # 02_robustness 同时可能存在 frozen 与 extended checkpoint；网络主表采用 frozen 主训练结果。
    if not selected.empty:
        selected = selected[
            (selected["experiment"] == "01_overall")
            | selected["checkpoint_training_config"].astype(str).str.contains("01_overall_main")
        ]
    rows: list[dict[str, Any]] = []
    cache: dict[str, Any] = {}
    reference_keys: set[tuple[str, int, str]] = set()
    for _, row in selected.iterrows():
        arrays = _prediction_arrays(row["run_dir"])
        region_data = _region_data_for_run(row, cache)
        rows.append({
            "model": row["model"], "region": row["region"], "seed": row["seed"],
            "scenario": row["scenario"], "run_dir": row["run_dir"],
            **network_recovery_metrics(region_data, arrays),
            **downstream_forecast_metrics(region_data, arrays),
        })
        reference_key = (str(row["region"]), int(row["seed"]), str(row["scenario"]))
        if reference_key not in reference_keys:
            reference_keys.add(reference_key)
            for reference_name, reference_prediction in (
                ("Raw incomplete", np.zeros_like(arrays["y_true"])),
                ("Oracle", np.asarray(arrays["y_true"]).copy()),
            ):
                reference_arrays = dict(arrays)
                reference_arrays["y_pred"] = reference_prediction
                rows.append({
                    "model": reference_name, "region": row["region"], "seed": row["seed"],
                    "scenario": row["scenario"], "run_dir": row["run_dir"],
                    **network_recovery_metrics(region_data, reference_arrays),
                    **downstream_forecast_metrics(region_data, reference_arrays),
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(output / "06_network_downstream_by_seed.tsv", sep="\t", index=False, encoding="utf-8")
        metrics = [column for column in result if column.startswith("NETWORK_") or column.startswith("TOP_") or column.startswith("FORECAST_")]
        summary = result.groupby(["model", "region", "scenario"], dropna=False)[metrics].agg(["mean", "std", "count"])
        summary.columns = [f"{left}_{right}" for left, right in summary.columns]
        summary.reset_index().to_csv(output / "06_network_downstream_summary.tsv", sep="\t", index=False, encoding="utf-8")
    return result


def _routing_analysis(runs: pd.DataFrame, output: Path) -> pd.DataFrame:
    selected = runs[
        runs["model"].eq("MRDP-OD") & runs["role"].isin(["final", "ablation"])
        & runs["experiment"].isin(["01_overall", "04_ablation", "02_robustness"])
    ]
    rows: list[dict[str, Any]] = []
    for _, run in selected.iterrows():
        path = Path(run["run_dir"]) / "target_diagnostics.csv"
        if not path.exists():
            continue
        data = pd.read_csv(path)
        required = {"gate", "temporal_prior", "relational_prior", "prior", "y_true", "y_pred"}
        if not required.issubset(data.columns) or data.empty:
            continue
        temporal_error = np.abs(data["temporal_prior"] - data["y_true"])
        relational_error = np.abs(data["relational_prior"] - data["y_true"])
        routed_error = np.abs(data["prior"] - data["y_true"])
        oracle_error = np.minimum(temporal_error, relational_error)
        router_correct = ((data["gate"] >= 0.5) == (temporal_error <= relational_error)).mean()
        base = {
            "experiment": run["experiment"], "model": run["model"], "region": run["region"],
            "seed": run["seed"], "scenario": run["scenario"], "run_dir": run["run_dir"],
            "router_better_prior_accuracy": float(router_correct),
            "routed_oracle_regret": float(np.mean(routed_error - oracle_error)),
            "prior_disagreement_mean": float(np.mean(np.abs(data["temporal_prior"] - data["relational_prior"]))),
            "final_error_mean": float(np.mean(np.abs(data["y_pred"] - data["y_true"]))),
            "uncertainty_mean": float(data["sample_std"].mean()),
        }
        for index, label in enumerate(("temporal_local", "temporal_weekly", "temporal_long", "pair", "origin", "destination")):
            column = f"router_features_{index}"
            base[f"gate_corr_{label}_support"] = (
                float(data["gate"].corr(data[column], method="spearman")) if column in data else math.nan
            )
        base["gate_corr_prior_error_gap"] = float(
            data["gate"].corr(pd.Series(relational_error - temporal_error), method="spearman")
        )
        rows.append(base)
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(output / "04_ablation_routing_mechanism_by_seed.tsv", sep="\t", index=False, encoding="utf-8")
    return result


def _probabilistic_subgroups(runs: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    probabilistic = runs[
        runs["model"].eq("MRDP-OD")
        & runs["role"].isin(["final", "ablation"])
    ]
    calibration_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    interval_columns = {
        50: ("q25", "q75"), 80: ("q10", "q90"),
        90: ("q05", "q95"), 95: ("q025", "q975"),
    }
    for _, run in probabilistic.iterrows():
        path = Path(run["run_dir"]) / "target_diagnostics.csv"
        if not path.exists():
            continue
        data = pd.read_csv(path)
        if data.empty or not {"y_true", "y_pred", "sample_std"}.issubset(data):
            continue
        dimensions: list[tuple[str, pd.Series]] = []
        for name, column in (("flow_quantile", "y_true"), ("distance_quantile", "distance_line")):
            unique = int(data[column].nunique(dropna=True))
            if unique >= 2:
                dimensions.append((name, pd.qcut(data[column], q=min(4, unique), duplicates="drop").astype(str)))
        dimensions.append((
            "gap_length",
            pd.cut(data["gap_length"], [0, 3, 7, 14, np.inf], labels=["1-3", "4-7", "8-14", ">14"], include_lowest=True).astype(str),
        ))
        support_column = "support_pair" if "support_pair" in data else (
            "router_features_3" if "router_features_3" in data else None
        )
        if support_column is not None:
            dimensions.append((
                "pair_support_quantile",
                pd.qcut(data[support_column], q=4, duplicates="drop").astype(str),
            ))
        for dimension, labels in dimensions:
            grouped = data.assign(_group=labels).groupby("_group", observed=False)
            for group, part in grouped:
                if part.empty:
                    continue
                base = {
                    "model": run["model"], "region": run["region"], "seed": run["seed"],
                    "scenario": run["scenario"], "dimension": dimension,
                    "group": str(group), "count": len(part),
                    "MAE": float(np.mean(np.abs(part["y_pred"] - part["y_true"]))),
                    "mean_uncertainty": float(part["sample_std"].mean()),
                }
                for level, (lower_name, upper_name) in interval_columns.items():
                    if lower_name in part and upper_name in part:
                        covered = (part["y_true"] >= part[lower_name]) & (part["y_true"] <= part[upper_name])
                        base[f"PICP{level}"] = float(covered.mean())
                        base[f"MPIW{level}"] = float((part[upper_name] - part[lower_name]).mean())
                calibration_rows.append(base)
        ordered = data.sort_values("sample_std", kind="mergesort")
        for coverage in np.linspace(0.1, 1.0, 10):
            count = max(1, int(np.ceil(len(ordered) * coverage)))
            part = ordered.iloc[:count]
            risk_rows.append({
                "model": run["model"], "region": run["region"], "seed": run["seed"],
                "scenario": run["scenario"], "retained_fraction": coverage,
                "MAE": float(np.mean(np.abs(part["y_pred"] - part["y_true"]))),
                "mean_uncertainty": float(part["sample_std"].mean()), "count": count,
            })
    calibration = pd.DataFrame(calibration_rows)
    risk = pd.DataFrame(risk_rows)
    if not calibration.empty:
        calibration.to_csv(output / "05_calibration_grouped_calibration.tsv", sep="\t", index=False, encoding="utf-8")
    if not risk.empty:
        risk.to_csv(output / "05_calibration_selective_risk.tsv", sep="\t", index=False, encoding="utf-8")
    return calibration, risk


def _statistical_comparisons(runs: pd.DataFrame, output: Path) -> pd.DataFrame:
    core = runs[(runs["experiment"] == "01_overall") & (runs["role"] == "final")].copy()
    rows: list[dict[str, Any]] = []
    for (region, scenario), group in core.groupby(["region", "scenario"]):
        mrdp_runs = group[group["model"].eq("MRDP-OD")]
        if mrdp_runs.empty:
            continue
        for baseline in sorted(set(group["model"]) - {"MRDP-OD"}):
            baseline_runs = group[group["model"].eq(baseline)]
            baseline_units: list[np.ndarray] = []
            candidate_units: list[np.ndarray] = []
            reused_deterministic = False
            for _, candidate_run in mrdp_runs.iterrows():
                match = baseline_runs[baseline_runs["seed"].eq(candidate_run["seed"])]
                if match.empty and len(baseline_runs) == 1:
                    match = baseline_runs
                    reused_deterministic = True
                if match.empty:
                    continue
                baseline_path = Path(match.iloc[0]["run_dir"]) / "target_diagnostics.csv"
                candidate_path = Path(candidate_run["run_dir"]) / "target_diagnostics.csv"
                if not baseline_path.exists() or not candidate_path.exists():
                    continue
                left = pd.read_csv(baseline_path, usecols=["date_index", "y_true", "y_pred"])
                right = pd.read_csv(candidate_path, usecols=["date_index", "y_true", "y_pred"])
                def by_date(frame: pd.DataFrame) -> pd.Series:
                    frame = frame.assign(_error=np.abs(frame["y_pred"] - frame["y_true"]))
                    grouped = frame.groupby("date_index", observed=False)
                    return grouped["_error"].sum() / grouped["y_true"].apply(lambda value: np.abs(value).sum()).clip(lower=1e-12)
                paired_dates = pd.concat(
                    [by_date(left).rename("baseline"), by_date(right).rename("candidate")], axis=1,
                ).dropna()
                baseline_units.append(paired_dates["baseline"].to_numpy())
                candidate_units.append(paired_dates["candidate"].to_numpy())
            if not baseline_units:
                continue
            baseline_values = np.concatenate(baseline_units)
            candidate_values = np.concatenate(candidate_units)
            stats = paired_bootstrap_difference(
                baseline_values, candidate_values, n_resamples=5000
            )
            probability = stats["probability_improved"]
            rows.append({
                "region": region, "scenario": scenario, "baseline": baseline,
                "pairing_level": "model-seed × test-date block",
                "n_paired_blocks": len(baseline_values),
                "deterministic_baseline_reused_across_model_seeds": reused_deterministic,
                **stats,
                "relative_improvement": float(
                    (baseline_values.mean() - candidate_values.mean()) / max(abs(baseline_values.mean()), 1e-12)
                ),
                "p_value_two_sided": float(min(1.0, 2.0 * min(probability, 1.0 - probability))),
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_holm"] = holm_adjust(result["p_value_two_sided"].tolist())
        result.to_csv(output / "paired_bootstrap_holm.tsv", sep="\t", index=False, encoding="utf-8")
    return result


def _failure_and_efficiency(runs: pd.DataFrame, output: Path) -> None:
    failure_frames: list[pd.DataFrame] = []
    for _, run in runs[runs["role"].isin(["final", "ablation"])].iterrows():
        path = Path(run["run_dir"]) / "subgroup_metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame.insert(0, "scenario", run["scenario"])
            frame.insert(0, "seed", run["seed"])
            frame.insert(0, "region", run["region"])
            frame.insert(0, "model", run["model"])
            failure_frames.append(frame)
    if failure_frames:
        pd.concat(failure_frames, ignore_index=True).to_csv(
            output / "failure_boundaries.tsv", sep="\t", index=False, encoding="utf-8"
        )
    columns = [column for column in (
        "experiment", "model", "region", "seed", "scenario", "parameter_count",
        "training_seconds", "test_seconds", "total_seconds", "peak_gpu_memory_mb", "evaluation_samples",
        "inference_benchmark_seconds_mean", "inference_benchmark_seconds_sample_sd",
        "inference_benchmark_seconds_per_window", "inference_benchmark_windows_per_second",
        "inference_benchmark_requested_samples", "inference_benchmark_actual_samples",
        "inference_benchmark_batch_size", "inference_benchmark_warmup_runs", "inference_benchmark_repeats",
        "inference_benchmark_peak_gpu_memory_mb",
        "MAE", "WMAPE", "CRPS", "WIS",
    ) if column in runs]
    if columns:
        runs[runs["role"].eq("final")][columns].to_csv(
            output / "efficiency_accuracy_tradeoff.tsv", sep="\t", index=False, encoding="utf-8"
        )


def analyze_results(smoke: bool = False) -> Path:
    root = experiment_results_root(smoke)
    output = root / "Analysis"
    output.mkdir(parents=True, exist_ok=True)
    runs = collect_standard_runs(smoke)
    runs.to_csv(output / "experiment_registry.tsv", sep="\t", index=False, encoding="utf-8")
    _standard_summaries(runs, output)
    calibration = _calibration_analysis(runs, output)
    network = _network_downstream_analysis(runs, output)
    routing = _routing_analysis(runs, output)
    grouped_calibration, selective_risk = _probabilistic_subgroups(runs, output)
    statistics = _statistical_comparisons(runs, output)
    transfer = pd.DataFrame()
    _failure_and_efficiency(runs, output)
    write_json({
        "standard_completed_runs": len(runs),
        "calibration_rows": len(calibration), "network_downstream_rows": len(network),
        "routing_rows": len(routing), "statistical_comparisons": len(statistics),
        "grouped_calibration_rows": len(grouped_calibration),
        "selective_risk_rows": len(selective_risk),
        "cross_region_rows": len(transfer), "output": str(output),
    }, output / "analysis_status.json")
    return output
