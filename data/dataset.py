"""按时间切分生成不跨边界的 OD 窗口样本。"""
from __future__ import annotations

from copy import deepcopy
from typing import Literal
import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import make_target_mask
from .entity_splits import held_out_pair_mask
from .tensor_builder import RegionTensorData

def _make_seed(*parts: int) -> int:
    """
    根据多个整数生成稳定的 32 位随机种子。

    相同输入始终产生相同种子，同时避免 epoch、样本序号和
    数据切分偏移量累加后超过 RandomState 的合法范围。
    """
    values = [
        int(value) & 0xFFFFFFFF
        for value in parts
    ]

    sequence = np.random.SeedSequence(values)

    return int(
        sequence.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )

class ODWindowDataset(Dataset):
    """输出 [L,E] OD 流量、原生掩码、条件掩码、目标掩码和上下文。"""

    def __init__(
        self,
        region_data: RegionTensorData,
        split: Literal["train", "valid", "test"],
        cfg: dict,
        mask_spec: dict | None = None,
        scenario_name: str | None = None,
    ) -> None:
        self.data = region_data
        self.split = split
        self.cfg = cfg
        self.mask_cfg = deepcopy(cfg["mask"])
        if mask_spec is not None:
            self.mask_cfg.update({key: deepcopy(value) for key, value in mask_spec.items() if key != "weight"})
        self.scenario_name = str(
            scenario_name
            or self.mask_cfg.get("name")
            or cfg.get("search", {}).get("scenario")
            or split
        )
        self.window_len = int(cfg["data"].get("window_len", 30))
        split_stride = cfg["data"].get("split_stride", {})
        self.stride = int(split_stride.get(split, cfg["data"].get("stride", 7)))
        self.epoch = 0
        self.entity_spec = deepcopy(cfg.get("data", {}).get("entity_split") or {})
        self.held_out_pairs = held_out_pair_mask(
            region_data.origin_idx,
            region_data.destination_idx,
            region_data.num_cities,
            self.entity_spec,
        )
        indices = {"train": region_data.train_idx, "valid": region_data.valid_idx, "test": region_data.test_idx}[split]
        self.windows = self._make_windows(indices)
        if split == "train":
            fraction = float(cfg.get("data", {}).get("train_fraction", 1.0))
            if not 0.0 < fraction <= 1.0:
                raise ValueError("data.train_fraction 必须位于 (0,1]")
            if fraction < 1.0:
                # 严格保留训练时段最早的一段，避免随机抽取造成未来窗口进入训练。
                keep = max(1, int(np.ceil(len(self.windows) * fraction)))
                self.windows = self.windows[:keep]
        max_windows = cfg["data"].get("max_windows", {}).get(split)
        if max_windows:
            self.windows = self.windows[: int(max_windows)]
        if not self.windows:
            raise ValueError(f"{region_data.region}/{split} 没有合法窗口，请检查 window_len 和 split")

    def set_epoch(self, epoch: int) -> None:
        """训练集每个 epoch 使用新的、可复现的人工缺失。"""
        self.epoch = int(epoch)

    def _make_windows(self, indices: np.ndarray) -> list[np.ndarray]:
        """生成严格位于同一切分且日期连续的滑动窗口。"""
        indices = np.asarray(indices, dtype=np.int64)
        windows: list[np.ndarray] = []
        for start in range(0, max(0, len(indices) - self.window_len + 1), self.stride):
            window = indices[start : start + self.window_len]
            if len(window) != self.window_len:
                continue
            dates = self.data.dates[window]
            if len(dates) > 1 and not np.all(np.diff(dates.values).astype("timedelta64[D]") == np.timedelta64(1, "D")):
                continue
            windows.append(window.copy())
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def _missing_spec(self, index: int) -> tuple[str, float, int | None]:
        """训练时按配置混合缺失模式，验证和测试使用固定模式。"""
        mask_cfg = self.mask_cfg
        train_scenarios = mask_cfg.get("train_scenarios")
        if self.split == "train" and train_scenarios:
            names = [str(item["name"]) for item in train_scenarios]
            probabilities = np.asarray([float(item.get("weight", 1.0)) for item in train_scenarios], dtype=float)
            probabilities /= probabilities.sum()
            base_seed = int(mask_cfg.get("seed", 2026))
            seed = _make_seed(base_seed, 1, self.epoch, index)
            chosen_index = int(np.random.RandomState(seed).choice(len(train_scenarios), p=probabilities))
            chosen = train_scenarios[chosen_index]
            return (
                str(chosen["missing_type"]),
                float(chosen.get("missing_rate", 0.5)),
                chosen.get("city_count"),
            )
        if self.split != "train" or not mask_cfg.get("train_mix"):
            return (
                str(mask_cfg["missing_type"]),
                float(mask_cfg.get("missing_rate", 0.5)),
                mask_cfg.get("city_count"),
            )
        mix = mask_cfg["train_mix"]
        names = list(mix)
        probabilities = np.asarray([mix[name] for name in names], dtype=float)
        probabilities /= probabilities.sum()
        base_seed = int(mask_cfg.get("seed", 2026))
        # 1为随机流编号：用于选择训练缺失类型
        seed = _make_seed(base_seed, 1, self.epoch, index, )
        rng = np.random.RandomState(seed)
        chosen = str(rng.choice(names, p=probabilities))
        return chosen, float(mask_cfg.get("missing_rate", 0.5)), mask_cfg.get("city_count")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        x = self.data.x[window].astype(np.float32)
        native = self.data.native_mask[window].astype(np.float32)
        available = native.copy()
        entity_target = bool(self.cfg.get("evaluation", {}).get("entity_target", False))
        entity_evaluation = entity_target and self.split in {"valid", "test"}
        if self.held_out_pairs.any() and not entity_evaluation:
            available[:, self.held_out_pairs] = 0.0
        context = self.data.context[window].astype(np.float32)
        missing_type, missing_rate, city_count = self._missing_spec(index)
        split_code = { "train": 0, "valid": 1, "test": 2, }[self.split]
        epoch_code = ( self.epoch if self.split == "train" else 0 )
        base_seed = int(self.mask_cfg.get("seed", 2026))

        seed = _make_seed( base_seed, 2,# 随机流编号：用于生成具体 target mask
            split_code, epoch_code, index, )

        rng = np.random.RandomState(seed)
        if entity_evaluation and self.held_out_pairs.any():
            target = np.zeros_like(native, dtype=np.float32)
            target[:, self.held_out_pairs] = native[:, self.held_out_pairs]
            denominator_mask = native
        else:
            driver = self._driver(missing_type, x, context)
            target = make_target_mask(
                available,
                missing_type,
                missing_rate,
                rng,
                self.data.origin_idx,
                self.data.destination_idx,
                block_len=int(self.mask_cfg.get("temporal_block_len", 7)),
                city_count=city_count,
                driver=driver,
                native_reference=self.data.native_mask[self.data.train_idx],
                driver_strength=float(self.mask_cfg.get("driver_strength", 3.0)),
            )
            denominator_mask = available
        cond = available * (1.0 - target)
        if np.any(target > native) or np.any(cond > native) or np.any(target * cond):
            raise RuntimeError("掩码关系不满足 native/target/condition 约束")
        actual_rate = float(target.sum() / max(denominator_mask.sum(), 1.0))
        native_query = (1.0 - native).astype(np.float32)
        generation = target.astype(np.float32)
        if bool(self.cfg.get("evaluation", {}).get("include_native_queries", False)):
            generation = np.maximum(generation, native_query)
        return {
            "observed_data": torch.from_numpy(x),
            "native_mask": torch.from_numpy(native),
            "cond_mask": torch.from_numpy(cond.astype(np.float32)),
            "target_mask": torch.from_numpy(target.astype(np.float32)),
            "evaluation_mask": torch.from_numpy(target.astype(np.float32)),
            "native_query_mask": torch.from_numpy(native_query),
            "generation_mask": torch.from_numpy(generation.astype(np.float32)),
            "available_mask": torch.from_numpy(available.astype(np.float32)),
            "context": torch.from_numpy(context),
            "start_index": torch.tensor(int(window[0]), dtype=torch.long),
            "configured_missing_rate": torch.tensor(float(missing_rate), dtype=torch.float32),
            "actual_missing_rate": torch.tensor(actual_rate, dtype=torch.float32),
        }

    def _driver(self, missing_type: str, x: np.ndarray, context: np.ndarray) -> np.ndarray | None:
        """构造 MAR/MNAR 的可复现 driver；返回形状可广播到 [L,E]。"""
        if missing_type in {"mnar_high_flow", "mnar_low_flow"}:
            return x
        if missing_type == "mar_distance":
            values = self.data.pair_frame.get("distance_line")
            if values is None:
                values = np.arange(self.data.num_pairs, dtype=np.float32)
            return np.asarray(values, dtype=np.float32).reshape(1, -1)
        if missing_type in {"mar_weather", "mar_calendar"}:
            names = list(self.data.context_feature_names)
            if missing_type == "mar_weather":
                tokens = ("temp", "wind", "snow", "dewpoint")
            else:
                tokens = ("weekend", "holiday")
            selected = [i for i, name in enumerate(names) if any(token in name.lower() for token in tokens)]
            if not selected:
                raise ValueError(f"{missing_type} 无法在 context 中找到 driver 特征")
            return np.max(np.abs(context[..., selected]), axis=-1)
        return None
