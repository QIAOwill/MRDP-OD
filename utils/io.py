"""JSON、CSV、NPZ 和环境信息写入工具。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 Path。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"无法 JSON 序列化：{type(obj)}")


def write_json(data: Any, path: str | Path) -> None:
    """保存中文友好的 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_default)


def append_csv_row(row: dict[str, Any], path: str | Path) -> None:
    """向 CSV 安全追加一条记录，并在字段变化时扩展既有表头。

    不同模型拥有不同的概率指标和诊断字段，直接 ``mode="a"`` 会在
    列集合变化时生成错位 CSV。这里统一读取既有注册表、取列并集后原子重写。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame([row])
    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path, encoding="utf-8-sig")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        columns = list(existing.columns)
        columns.extend(column for column in incoming.columns if column not in columns)
        existing = existing.reindex(columns=columns)
        incoming = incoming.reindex(columns=columns)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """保存 DataFrame。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def git_commit(project_root: str | Path) -> str:
    """读取当前 Git commit；非仓库环境返回 unavailable。"""
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unavailable"


def environment_snapshot() -> dict[str, Any]:
    """收集复现实验所需的软件和硬件信息。"""
    cuda_name = None
    cuda_capability = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0)
        cuda_capability = torch.cuda.get_device_capability(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": cuda_name,
        "gpu_capability": cuda_capability,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pid": os.getpid(),
    }
