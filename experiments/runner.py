"""依次执行原子 JSON 配置，并复用相同的数据张量。"""
from __future__ import annotations

import gc
from pathlib import Path
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]

from data.reader import read_region
from data.tensor_builder import build_region_tensor
from training.experiment import classify_error, resolve_project_path, run_single_experiment
from utils.config import load_json, stable_json
from utils.model_names import canonical_model_name


def resolve_config_path(config_path: str | Path) -> Path:
    value = Path(config_path)
    return value.resolve() if value.is_absolute() else (CODE_ROOT / value).resolve()


def _data_cache_key(config: dict) -> str:
    return stable_json({
        "dataset_dir": config["paths"]["dataset_dir"],
        "data": config["data"],
        "graph": config["graph"],
    })


def run_config_paths(
    config_paths: list[str | Path],
    stop_on_error: bool = True,
) -> list[dict]:
    """顺序运行配置文件；失败时可立即停止或记录后继续。"""
    if not config_paths:
        raise ValueError("CONFIG_PATHS 不能为空，请至少填写一个 JSON 配置路径")
    data_cache: dict[str, object] = {}
    outcomes: list[dict] = []
    total = len(config_paths)
    for index, raw_path in enumerate(config_paths, 1):
        config_path = resolve_config_path(raw_path)
        config = load_json(config_path)
        region = str(config["data"]["region"])
        model_name = canonical_model_name(config["model"]["name"])
        print("\n" + "=" * 88)
        print(f"配置 {index}/{total}：{config_path.name}")
        print(f"模型：{model_name}")
        print(f"城市群：{region}")
        print(f"场景：{config.get('search', {}).get('scenario')}")
        print(f"Grid/角色：{config.get('search', {}).get('grid_id')} / {config.get('search', {}).get('role', 'final')}")
        print(f"随机种子：{config['project']['seed']}")
        print("=" * 88)
        try:
            cache_key = _data_cache_key(config)
            if cache_key not in data_cache:
                dataset_dir = resolve_project_path(config["paths"]["dataset_dir"])
                raw_data = read_region(dataset_dir, region)
                data_cache[cache_key] = build_region_tensor(region, raw_data, config)
            outcomes.append(run_single_experiment(config, region_data=data_cache[cache_key]))
        except Exception as error:
            outcomes.append({
                "status": "failed",
                "config": str(config_path),
                "error_type": classify_error(error),
                "error": repr(error),
            })
            if stop_on_error:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    completed = sum(item.get("status") in {"completed", "skipped"} for item in outcomes)
    failed = sum(item.get("status") == "failed" for item in outcomes)
    print(f"\n全部配置处理结束：完成或跳过 {completed} 个，失败 {failed} 个。")
    return outcomes
