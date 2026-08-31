"""MRDP-OD public-release experiment entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import argparse
import json

from analysis.results import analyze_results, experiment_results_root
from experiments.audit import run_protocol_audit
from experiments.configs import GENERATORS, generate_overall, generate_overall_tuning
from experiments.protocol import MODELS
from experiments.runner import run_config_paths
from experiments.readiness import verify_results_readiness
from experiments.network import run_network_downstream_experiment
from experiments.selection import select_models


EXPERIMENT_ORDER = (
    "01_overall",
    "02_robustness",
    "03_cross_pattern",
    "04_ablation",
    "05_calibration",
    "06_network",
)
PIPELINE_ORDER = ("audit", *EXPERIMENT_ORDER, "analyze", "verify")


# ---------------- 参数规范化 ----------------
def normalize_models(values: Iterable[str] | None) -> tuple[str, ...]:
    """将用户填写的模型名转换为协议中的标准名称，并保持填写顺序。"""
    if not values:
        return MODELS
    lookup = {name.lower(): name for name in MODELS}
    selected: list[str] = []
    for value in values:
        key = str(value).strip().lower()
        if key not in lookup:
            raise ValueError(f"未知模型 {value!r}；可选：{', '.join(MODELS)}")
        selected.append(lookup[key])
    return tuple(dict.fromkeys(selected))


def normalize_steps(values: Iterable[str] | None) -> tuple[str, ...]:
    """检查执行步骤；``all`` 会展开为完整实验流水线。"""
    steps = tuple(str(value).strip() for value in (values or ("status",)))
    if "all" in steps:
        return PIPELINE_ORDER
    valid = {"audit", "generate", "tune", "select", "main", "analyze", "verify", "status", *EXPERIMENT_ORDER}
    unknown = [step for step in steps if step not in valid]
    if unknown:
        raise ValueError(f"未知执行步骤：{unknown}；可选：{sorted(valid)}")
    return steps


# ---------------- 原子配置执行 ----------------
def _run_standard(paths: list[Path], continue_on_error: bool) -> list[dict[str, Any]]:
    return run_config_paths(
        [str(path) for path in paths],
        stop_on_error=not continue_on_error,
    )


def run_tuning(
    models: tuple[str, ...],
    smoke: bool = False,
    continue_on_error: bool = False,
) -> dict[str, dict[str, Any]]:
    """展开全部候选组合、运行固定验证场景并选择一个全局最佳组合。"""
    paths = generate_overall_tuning(models=models, smoke=smoke, clean=True)
    outcomes = _run_standard(paths, continue_on_error)
    failed = [item for item in outcomes if item.get("status") == "failed"]
    if failed:
        raise RuntimeError(
            f"超参数搜索有 {len(failed)} 个配置失败；修复后重新运行，完整前不会选参"
        )
    selected = select_models(models=models, smoke=smoke)
    for model, payload in selected.items():
        print(
            f"[selected] {model}: {payload['best_grid_id']} / "
            f"macro-WMAPE={payload['best_validation_score']:.6f}"
        )
    return selected


def run_main_experiment(
    models: tuple[str, ...],
    smoke: bool = False,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    """每个模型种子训练一次，并依次测试八个固定场景。"""
    return _run_standard(generate_overall(models=models, smoke=smoke), continue_on_error)


def run_named_experiment(
    name: str,
    models: tuple[str, ...] = MODELS,
    smoke: bool = False,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    """生成并运行指定的扩展实验。"""
    if name not in EXPERIMENT_ORDER:
        raise ValueError(f"不可直接执行的实验：{name}")
    if name == "01_overall":
        return run_main_experiment(models, smoke, continue_on_error)
    if name == "06_network":
        return [{"status": "completed", "output": str(run_network_downstream_experiment(smoke))}]
    paths = GENERATORS[name](smoke=smoke)
    return _run_standard(paths, continue_on_error)


# ---------------- 配置生成与状态查看 ----------------
def generate_configurations(
    experiment_ids: Iterable[str],
    models: tuple[str, ...],
    smoke: bool = False,
) -> dict[str, int]:
    """只生成 JSON 配置，不启动训练或测试。"""
    counts: dict[str, int] = {}
    for name in experiment_ids:
        if name == "overall_tuning":
            paths = generate_overall_tuning(models=models, smoke=smoke)
        elif name == "overall_final":
            paths = generate_overall(models=models, smoke=smoke)
        elif name in EXPERIMENT_ORDER:
            if name == "01_overall":
                paths = generate_overall(models=models, smoke=smoke)
                counts[name] = len(paths)
                print(f"[generated] {name}: {len(paths)} JSON")
                continue
            if name == "06_network":
                counts[name] = 0
                print("[generated] 06_network: analysis-only job，直接复用 01_overall/02_robustness 原始预测")
                continue
            paths = GENERATORS[name](smoke=smoke)
        else:
            raise ValueError(f"未知配置组：{name}")
        counts[name] = len(paths)
        print(f"[generated] {name}: {len(paths)} JSON")
    return counts


def print_status(smoke: bool = False) -> dict[str, dict[str, int]]:
    """统计各实验中已完成、失败和运行中的原子任务。"""
    root = experiment_results_root(smoke)
    counts: dict[str, dict[str, int]] = {}
    if root.exists():
        for path in root.rglob("run_status.json"):
            relative = path.relative_to(root)
            if not smoke and relative.parts and relative.parts[0].startswith("_"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            experiment = relative.parts[0] if relative.parts else "unknown"
            status = str(payload.get("status", "unknown"))
            counts.setdefault(experiment, {}).setdefault(status, 0)
            counts[experiment][status] += 1
    print(f"Results root: {root}")
    if not counts:
        print("尚无运行结果。")
    for experiment, values in sorted(counts.items()):
        summary = ", ".join(f"{key}={value}" for key, value in sorted(values.items()))
        print(f"{experiment}: {summary}")
    return counts


# ---------------- 统一流水线 ----------------
def run_pipeline(
    steps: Iterable[str] = ("status",),
    models: Iterable[str] | None = None,
    experiment_ids: Iterable[str] = EXPERIMENT_ORDER,
    smoke: bool = False,
    continue_on_error: bool = False,
) -> None:
    """按列表顺序运行多个步骤，支持一次完成全部实验。"""
    selected_models = normalize_models(models)
    selected_steps = normalize_steps(steps)
    requested_experiments = tuple(experiment_ids)
    for step in selected_steps:
        if step == "audit":
            print(run_protocol_audit(smoke))
        elif step == "generate":
            generate_configurations(requested_experiments, selected_models, smoke)
        elif step == "tune":
            run_tuning(selected_models, smoke, continue_on_error)
        elif step == "select":
            payload = select_models(selected_models, smoke)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif step == "main":
            run_main_experiment(selected_models, smoke, continue_on_error)
        elif step in EXPERIMENT_ORDER:
            run_named_experiment(step, selected_models, smoke, continue_on_error)
        elif step == "analyze":
            print(analyze_results(smoke))
        elif step == "verify":
            print(verify_results_readiness(smoke, strict=True))
        elif step == "status":
            print_status(smoke)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MRDP-OD experiment pipeline with repository-local sample data."
    )
    parser.add_argument(
        "--steps", nargs="+", default=["status"],
        help="Pipeline steps (default: status). Use 'all' only for a complete run.",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=list(EXPERIMENT_ORDER),
        choices=EXPERIMENT_ORDER, help="Experiment groups used by the generate step.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use one epoch and one batch for a quick integration check.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Record a failed configuration and continue with the next one.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        steps=args.steps,
        models=["MRDP-OD"],
        experiment_ids=args.experiments,
        smoke=args.smoke,
        continue_on_error=args.continue_on_error,
    )
