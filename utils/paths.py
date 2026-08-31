"""代码、数据、分模型 Results 与 Run_X 的路径管理。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

from utils.model_names import RESULT_FOLDERS, canonical_model_name

RUN_PATTERN = re.compile(r"^Run_(\d+)$")


def code_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return code_root()


def resolve_code_relative(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (code_root() / value).resolve()


def default_data_dir() -> Path:
    return code_root() / "sample_data"


def default_results_dir() -> Path:
    return code_root() / "outputs"


def model_folder(model_name: str) -> str:
    return RESULT_FOLDERS[canonical_model_name(model_name)]


def _validated_result_folder(result_folder: str | None) -> Path | None:
    """校验自定义结果目录，防止实验写到 Results 根目录之外。"""

    if result_folder is None:
        return None
    value = str(result_folder).strip()
    folder = Path(value)
    if not value or value == "." or folder.is_absolute() or ".." in folder.parts:
        raise ValueError(
            "paths.result_folder 必须是 Results 下的安全相对路径，"
            f"当前值为 {result_folder!r}"
        )
    return folder


def _validated_relative_subpath(value: str | None, field: str) -> Path | None:
    """校验 Results 内部的 domain/stage 子目录，拒绝绝对路径和目录穿越。"""
    if value is None:
        return None
    text = str(value).strip()
    path = Path(text)
    if not text or text == "." or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} 必须是 Results 下的安全相对路径，当前值为 {value!r}")
    return path


def result_domain(config: dict[str, Any], default_region: str | None = None) -> str:
    """返回结果目录使用的实验域；同区域实验默认等于 data.region。"""
    value = config.get("paths", {}).get("result_domain")
    if value is None:
        value = default_region if default_region is not None else config.get("data", {}).get("region")
    path = _validated_relative_subpath(str(value), "paths.result_domain")
    if path is None:
        raise ValueError("无法确定结果 domain")
    return path.as_posix()


def model_results_root(
    results_root: str | Path,
    model_name: str,
    result_folder: str | None = None,
) -> Path:
    """返回并创建实验集合的结果根目录。

    默认使用模型注册表中的目录；配置 ``paths.result_folder`` 后则直接使用
    ``Results/<result_folder>``，但不会改变实际构建的模型名称。
    """

    custom_folder = _validated_result_folder(result_folder)
    path = Path(results_root) / (
        custom_folder if custom_folder is not None else Path(model_folder(model_name))
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_empty_dir(path: Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def allocate_run_dir(
    results_root: str | Path,
    model_name: str,
    region: str,
    result_folder: str | None = None,
    run_subdir: str | None = None,
) -> Path:
    """在模型与区域对应的结果目录下分配 ``Run_X`` 子目录。"""
    region_dir = model_results_root(
        results_root,
        model_name,
        result_folder=result_folder,
    ) / region
    safe_subdir = _validated_relative_subpath(run_subdir, "paths.run_subdir")
    if safe_subdir is not None:
        region_dir = region_dir / safe_subdir
    region_dir.mkdir(parents=True, exist_ok=True)
    numbered: list[tuple[int, Path]] = []
    for child in region_dir.iterdir():
        match = RUN_PATTERN.match(child.name)
        if match and child.is_dir():
            numbered.append((int(match.group(1)), child))
    for _, path in sorted(numbered):
        if _is_empty_dir(path):
            return path
    next_index = max((index for index, _ in numbered), default=0) + 1
    run_dir = region_dir / f"Run_{next_index}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def find_completed_run(
    results_root: str | Path,
    model_name: str,
    region: str,
    hash_value: str,
    result_folder: str | None = None,
    run_subdir: str | None = None,
) -> Path | None:
    region_dir = model_results_root(
        results_root,
        model_name,
        result_folder=result_folder,
    ) / region
    safe_subdir = _validated_relative_subpath(run_subdir, "paths.run_subdir")
    if safe_subdir is not None:
        region_dir = region_dir / safe_subdir
    if not region_dir.exists():
        return None
    for run_dir in sorted(region_dir.glob("Run_*")):
        status_path = run_dir / "run_status.json"
        if not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status.get("status") == "completed" and status.get("config_hash") == hash_value:
            return run_dir
    return None


def allocate_summary_dir(results_root: str | Path) -> Path:
    root = Path(results_root) / "Paper_Summaries"
    root.mkdir(parents=True, exist_ok=True)
    numbered: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("Summary_"):
            try:
                numbered.append((int(child.name.split("_")[-1]), child))
            except ValueError:
                continue
    for _, path in sorted(numbered):
        if _is_empty_dir(path):
            return path
    index = max((number for number, _ in numbered), default=0) + 1
    path = root / f"Summary_{index}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_number(run_dir: str | Path) -> int:
    match = RUN_PATTERN.match(Path(run_dir).name)
    if not match:
        raise ValueError(f"不是合法 Run_X 目录：{run_dir}")
    return int(match.group(1))


def experiment_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_name": Path(config.get("_meta", {}).get("source_config", "unknown.json")).name,
        "model_name": canonical_model_name(config["model"].get("name", "MRDP-OD")),
        "region": config["data"]["region"],
        "variant": config["model"].get("variant", "standard"),
        "missing_type": config["mask"]["missing_type"],
        "missing_rate": config["mask"].get("missing_rate"),
        "city_count": config["mask"].get("city_count"),
        "seed": config["project"]["seed"],
        "batch_size": config.get("training", {}).get("batch_size"),
        "grad_accumulation_steps": config.get("training", {}).get("grad_accumulation_steps", 1),
        "effective_batch_size": config.get("training", {}).get(
            "effective_batch_size",
            config.get("training", {}).get("batch_size"),
        ),
        "learning_rate": config.get("training", {}).get("lr"),
        "grid_id": config.get("search", {}).get("grid_id"),
        "search_space_hash": config.get("search", {}).get("search_space_hash"),
        "search_role": config.get("search", {}).get("role", "final"),
        "scenario": config.get("search", {}).get("scenario"),
        "result_folder": config.get("paths", {}).get("result_folder"),
        "result_domain": config.get("paths", {}).get("result_domain"),
        "run_subdir": config.get("paths", {}).get("run_subdir"),
        "experiment": config.get("experiment", {}).get("experiment"),
        "stage": config.get("experiment", {}).get("stage"),
        "source_regions": config.get("experiment", {}).get("source_regions"),
        "target_region": config.get("experiment", {}).get("target_region"),
        "token_mode": config.get("model", {}).get("city_token_mode", "transductive"),
    }
