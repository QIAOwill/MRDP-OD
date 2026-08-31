"""MRDP-OD 主模型：双先验路由、先验中心残差扩散和概率补全。"""
from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn

from data.tensor_builder import RegionTensorData
from .denoiser import PriorCenteredDenoiser
from .diffusion import DiffusionSchedule
from .dual_prior import RoutedDualPrior


class MRDPODModel(nn.Module):
    """封装先验构造、训练损失、反向采样和诊断信息。"""

    model_name = "MRDP-OD"
    is_probabilistic = True
    requires_training = True
    implementation_type = "native_project"
    supports_unseen_od = True
    supports_unseen_city = True
    supports_cross_region = True
    supports_sample_cpu_offload = True

    @classmethod
    def capability_manifest(cls) -> dict[str, Any]:
        return {
            "model_name": cls.model_name,
            "probabilistic": True,
            "requires_training": True,
            "unseen_od": True,
            "unseen_city": True,
            "cross_region": True,
            "implementation_type": cls.implementation_type,
            "condition": "city_token_mode=inductive and use_pair_id=false",
        }

    def __init__(self, region_data: RegionTensorData, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]
        relations = list(cfg["graph"]["relations"])
        self.register_buffer("pair_features", torch.from_numpy(region_data.pair_features).float(), persistent=False)
        self.register_buffer("city_features", torch.from_numpy(region_data.city_features).float(), persistent=False)
        self.register_buffer("origin_idx", torch.from_numpy(region_data.origin_idx).long(), persistent=False)
        self.register_buffer("destination_idx", torch.from_numpy(region_data.destination_idx).long(), persistent=False)
        self.relation_edges = region_data.relation_edges
        self.region_name = str(region_data.region)
        self.use_residual_diffusion = bool(model_cfg.get("use_residual_diffusion", True))
        self.dual_prior = RoutedDualPrior(
            region_data.num_cities,
            region_data.num_pairs,
            region_data.pair_features.shape[-1],
            region_data.city_features.shape[-1],
            region_data.context.shape[-1],
            relations,
            model_cfg,
        )
        self.denoiser = PriorCenteredDenoiser(
            region_data.num_cities,
            region_data.num_pairs,
            region_data.pair_features.shape[-1],
            region_data.city_features.shape[-1],
            region_data.context.shape[-1],
            relations,
            model_cfg,
        )
        diff_cfg = cfg["diffusion"]
        self.schedule = DiffusionSchedule(
            int(diff_cfg["num_steps"]),
            float(diff_cfg["beta_start"]),
            float(diff_cfg["beta_end"]),
            str(diff_cfg.get("schedule", "linear")),
        )

    @property
    def temporal_scales(self) -> tuple[int, ...]:
        return self.dual_prior.temporal_scales

    def _prior(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.dual_prior(
            batch["observed_data"],
            batch["cond_mask"],
            batch["context"],
            self.pair_features,
            self.city_features,
            self.origin_idx,
            self.destination_idx,
            self.relation_edges,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _residual_domain(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        返回具有真实 ground truth 且不属于条件输入的残差域。

        在标准训练/评价 batch 中该域应与 target mask 完全一致。原生缺失位置不参与
        残差构造、加噪或反向扩散，因而不会将占位零传播为伪训练信号。
        """
        target = batch.get("evaluation_mask", batch.get("target_mask"))
        if target is None:
            raise KeyError("batch 缺少 evaluation_mask/target_mask")
        native = batch["native_mask"]
        cond = batch["cond_mask"]
        if torch.any(target > native + 1e-6) or torch.any(target * cond > 1e-6):
            raise RuntimeError("evaluation mask 必须有真值且不能与 condition mask 重叠")
        return target

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """计算严格限定于有 ground truth 位置的扩散损失和先验辅助损失。"""
        prior_output = self._prior(batch)
        target = batch["observed_data"]
        target_mask = batch["target_mask"]
        residual_domain = self._residual_domain(batch)
        zero = target.new_zeros(())
        diffusion_prior = prior_output["prior"].detach()
        diffusion_gate = prior_output["gate"].detach()

        clean_residual = (target - diffusion_prior) * residual_domain
        steps = torch.randint(0, self.schedule.num_steps, (target.shape[0],), device=target.device)
        noise = torch.randn_like(clean_residual) * residual_domain
        noisy_residual = self.schedule.add_noise(clean_residual, steps, noise) * residual_domain
        predicted_noise = self.denoiser(
            noisy_residual,
            target,
            batch["cond_mask"],
            diffusion_prior,
            diffusion_gate,
            batch["context"],
            steps,
            self.pair_features,
            self.city_features,
            self.origin_idx,
            self.destination_idx,
            self.relation_edges,
        )
        diffusion_loss = (
            self._masked_mean((predicted_noise - noise) ** 2, target_mask)
            if self.use_residual_diffusion else zero
        )

        variant = str(self.cfg["model"].get("variant", "full"))
        relational_loss = (
            self._masked_mean(torch.abs(prior_output["relational_prior"] - target), target_mask)
            if variant in {"full", "fixed_fusion", "relational_only"}
            else zero
        )
        # relational_only 的 prior 与 relational_prior 完全相同，不能重复计算同一辅助损失。
        fused_prior_loss = (
            self._masked_mean(torch.abs(prior_output["prior"] - target), target_mask)
            if variant in {"full", "fixed_fusion", "temporal_only"}
            else zero
        )

        loss_cfg = self.cfg.get("loss", {})
        rel_weight = (
            float(loss_cfg.get("rel_prior_weight", 0.2))
            if variant in {"full", "fixed_fusion", "relational_only"}
            else 0.0
        )
        fused_weight = (
            float(loss_cfg.get("fused_prior_weight", 0.2))
            if variant in {"full", "fixed_fusion", "temporal_only"}
            else 0.0
        )
        if not self.use_residual_diffusion and rel_weight + fused_weight <= 0:
            fused_weight = 1.0
        total = diffusion_loss + rel_weight * relational_loss + fused_weight * fused_prior_loss
        return {
            "loss": total,
            "diffusion_loss": diffusion_loss,
            "relational_prior_loss": relational_loss,
            "fused_prior_loss": fused_prior_loss,
            "gate_mean": self._masked_mean(prior_output["gate"], target_mask),
        }

    @torch.no_grad()
    def impute(
        self,
        batch: dict[str, torch.Tensor],
        n_samples: int = 1,
        sample_output_device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        从高斯噪声生成多个评价补全样本。

        残差反向扩散限制在 generation_mask。训练损失仍只使用有真值的 evaluation_mask；
        operational run 可把真实原生缺失加入 generation_mask，但这些位置不参与指标。
        """
        prior_output = self._prior(batch)
        generation_mask = batch.get("generation_mask", batch.get("evaluation_mask", batch["target_mask"]))
        generation_mask = generation_mask.to(batch["observed_data"].dtype)
        sample_count = max(1, int(n_samples))
        sample_buffer = None
        samples: list[torch.Tensor] = []
        if sample_output_device is not None:
            b, length, pairs = batch["observed_data"].shape
            sample_buffer = torch.empty(
                (b, sample_count, length, pairs),
                dtype=torch.float32,
                device=torch.device(sample_output_device),
            )
        for sample_index in range(sample_count):
            if not self.use_residual_diffusion:
                completed = (
                    batch["cond_mask"] * batch["observed_data"]
                    + (1.0 - batch["cond_mask"]) * prior_output["prior"]
                )
                if sample_buffer is None:
                    samples.append(completed)
                else:
                    sample_buffer[:, sample_index].copy_(completed.detach().float())
                del completed
                continue
            residual = torch.randn_like(batch["observed_data"]) * generation_mask
            for step in range(self.schedule.num_steps - 1, -1, -1):
                steps = torch.full((residual.shape[0],), step, device=residual.device, dtype=torch.long)
                predicted_noise = self.denoiser(
                    residual,
                    batch["observed_data"],
                    batch["cond_mask"],
                    prior_output["prior"],
                    prior_output["gate"],
                    batch["context"],
                    steps,
                    self.pair_features,
                    self.city_features,
                    self.origin_idx,
                    self.destination_idx,
                    self.relation_edges,
                )
                predicted_noise = predicted_noise * generation_mask
                residual = self.schedule.reverse_step(residual, predicted_noise, step) * generation_mask
            completed = prior_output["prior"] + residual
            completed = (
                batch["cond_mask"] * batch["observed_data"]
                + (1.0 - batch["cond_mask"]) * completed
            )
            if sample_buffer is None:
                samples.append(completed)
            else:
                sample_buffer[:, sample_index].copy_(completed.detach().float())
            del completed, residual
        return {
            "samples": sample_buffer if sample_buffer is not None else torch.stack(samples, dim=1),
            **prior_output,
        }

    def bind_region(self, region_data: RegionTensorData) -> None:
        """为 inductive/transfer 实验切换区域特征和图，不改变可训练参数。"""
        if str(self.cfg["model"].get("city_token_mode", "transductive")) != "inductive":
            if region_data.num_cities != int(self.city_features.shape[0]):
                raise ValueError("transductive MRDP-OD 不能绑定具有不同城市集合的区域")
        device = self.pair_features.device
        self.pair_features = torch.from_numpy(region_data.pair_features).float().to(device)
        self.city_features = torch.from_numpy(region_data.city_features).float().to(device)
        self.origin_idx = torch.from_numpy(region_data.origin_idx).long().to(device)
        self.destination_idx = torch.from_numpy(region_data.destination_idx).long().to(device)
        self.relation_edges = region_data.relation_edges
        self.region_name = str(region_data.region)

    def load_transferable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """加载不含区域缓冲区的 inductive checkpoint。"""
        if str(self.cfg["model"].get("city_token_mode", "transductive")) != "inductive":
            raise ValueError("只有 inductive MRDP-OD 可以跨区域加载 checkpoint")
        missing, unexpected = self.load_state_dict(state, strict=False)
        unexpected = [name for name in unexpected if name not in {
            "pair_features", "city_features", "origin_idx", "destination_idx"
        }]
        if unexpected:
            raise RuntimeError(f"跨区域 checkpoint 含未知参数：{unexpected}")
        material_missing = [name for name in missing if not name.endswith("feature_mask")]
        if material_missing:
            raise RuntimeError(f"跨区域 checkpoint 缺少参数：{material_missing}")

    def relation_weights(self) -> dict[str, Any]:
        """返回实际实例化的关系先验和扩散去噪器关系权重。"""
        return {
            "relational_prior": self.dual_prior.relation_weights(),
            **self.denoiser.relation_weights(),
        }

    def parameter_breakdown(self) -> dict[str, int]:
        """返回可直接用于消融公平性报告的参数量分解。"""
        prior = self.dual_prior.parameter_breakdown()
        denoiser = sum(parameter.numel() for parameter in self.denoiser.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"total": total, "denoiser": denoiser, **prior}


    def training_phases(self) -> list[dict[str, Any]]:
        return [{"name": "main"}]

    def set_training_phase(self, phase: str) -> None:
        if phase != "main":
            raise ValueError(f"MRDP-OD 仅支持 main 阶段，收到 {phase}")

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def on_optimizer_step(self, epoch: int) -> None:
        del epoch

    def use_evaluation_weights(self) -> None:
        return None

    def restore_training_weights(self) -> None:
        return None

    def parameter_count(self) -> int:
        return self.parameter_breakdown()["total"]
