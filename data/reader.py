"""读取并校验三个城市群的原始 TSV 数据。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

REGIONS = ("Beijing_Group", "Chang_Delta", "Zhu_Delta")


def _read(path: Path, dates: bool = False) -> pd.DataFrame:
    """读取制表符分隔数据。"""
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig", parse_dates=["date"] if dates else None)


def read_region(data_dir: str | Path, region: str) -> Dict[str, pd.DataFrame]:
    """读取区域的 OD、城市属性、城市动态、城市对属性和城市对天气。"""
    if region not in REGIONS:
        raise ValueError(f"未知城市群：{region}；可选值为 {REGIONS}")
    root = Path(data_dir) / region
    data = {
        "od": _read(root / f"{region}_OD.txt", True),
        "city_static": _read(root / f"{region}_city_static.txt"),
        "city_dynamic": _read(root / f"{region}_city_dynamic.txt", True),
        "pair_static": _read(root / f"{region}_city_pair_static.txt"),
        "pair_weather": _read(root / f"{region}_pair_weather.txt", True),
    }
    validate_region(data, region)
    return data


def validate_region(data: Dict[str, pd.DataFrame], region: str) -> None:
    """检查主键、城市引用、城市对覆盖和流量合法性。"""
    keys = {
        "od": ["date", "origin_id", "destination_id"],
        "city_static": ["city_id"],
        "city_dynamic": ["date", "city_id"],
        "pair_static": ["origin_id", "destination_id"],
        "pair_weather": ["date", "origin_id", "destination_id"],
    }
    required = {
        "od": keys["od"] + ["od_flow"],
        "city_static": ["city_id"],
        "city_dynamic": keys["city_dynamic"],
        "pair_static": keys["pair_static"],
        "pair_weather": keys["pair_weather"],
    }
    for name, df in data.items():
        missing = [c for c in required[name] if c not in df.columns]
        if missing:
            raise ValueError(f"{region}/{name} 缺少列：{missing}")
        duplicate_count = int(df.duplicated(keys[name]).sum())
        if duplicate_count:
            raise ValueError(f"{region}/{name} 存在 {duplicate_count} 条重复主键")

    city_ids = set(data["city_static"]["city_id"].tolist())
    pair_keys = set(map(tuple, data["pair_static"][["origin_id", "destination_id"]].to_numpy()))
    expected = len(city_ids) * (len(city_ids) - 1)
    if len(pair_keys) != expected:
        raise ValueError(f"{region} 城市对数量为 {len(pair_keys)}，应为 {expected}")
    for table in ("od", "pair_weather"):
        refs = set(map(tuple, data[table][["origin_id", "destination_id"]].drop_duplicates().to_numpy()))
        unknown = refs - pair_keys
        if unknown:
            raise ValueError(f"{region}/{table} 存在未知城市对：{list(unknown)[:5]}")

    # NaN 表示原生未观测；真实零流量仍是合法观测。仅拒绝无穷和负值。
    flow = pd.to_numeric(data["od"]["od_flow"], errors="coerce").to_numpy(float)
    invalid = np.isinf(flow) | (np.isfinite(flow) & (flow < 0))
    if invalid.any():
        raise ValueError(f"{region} OD 流量存在无穷或负值")
    data["od"]["od_flow"] = flow
