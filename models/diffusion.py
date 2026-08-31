"""DDPM 前向加噪和条件反向采样。"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiffusionSchedule(nn.Module):
    """保存线性或二次 beta schedule 及其累积量。"""

    def __init__(self, num_steps: int, beta_start: float, beta_end: float, schedule: str = "linear") -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        if schedule == "quadratic":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, self.num_steps) ** 2
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_steps)
        else:
            raise ValueError(f"未知噪声 schedule：{schedule}")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bars[:-1]])
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bars).clamp_min(1e-12)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bars", alpha_bars.float())
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20).float())

    @staticmethod
    def _extract(values: torch.Tensor, steps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return values.gather(0, steps).view(steps.shape[0], *([1] * (target.ndim - 1)))

    def add_noise(self, clean: torch.Tensor, steps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """按闭式公式生成第 steps 步带噪残差。"""
        alpha_bar = self._extract(self.alpha_bars, steps, clean)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def reverse_step(self, current: torch.Tensor, noise_prediction: torch.Tensor, step: int) -> torch.Tensor:
        """执行一个 DDPM 反向步骤。"""
        beta = self.betas[step]
        alpha = self.alphas[step]
        alpha_bar = self.alpha_bars[step]
        mean = (current - beta / torch.sqrt(1.0 - alpha_bar) * noise_prediction) / torch.sqrt(alpha)
        if step == 0:
            return mean
        noise = torch.randn_like(current)
        return mean + torch.sqrt(self.posterior_variance[step]) * noise
