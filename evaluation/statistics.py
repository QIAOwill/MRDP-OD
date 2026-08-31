"""配对 bootstrap、效应量和 Holm 多重比较校正。"""
from __future__ import annotations

import numpy as np


def paired_bootstrap_difference(
    baseline: np.ndarray,
    candidate: np.ndarray,
    n_resamples: int = 5000,
    seed: int = 42024,
) -> dict[str, float]:
    """返回 candidate-baseline 的配对均值差与 percentile CI。负值表示误差降低。"""
    left = np.asarray(baseline, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("paired bootstrap 需要相同长度的非空向量")
    difference = right - left
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, len(difference), size=(int(n_resamples), len(difference)))
    means = difference[indices].mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "probability_improved": float(np.mean(means < 0.0)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()
