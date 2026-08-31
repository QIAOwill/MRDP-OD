"""MRDP-OD model factory for the public research-code release."""
from __future__ import annotations

from typing import Any

from utils.model_names import canonical_model_name
from .mrdp_od import MRDPODModel


MODEL_CLASSES = {
    "MRDP-OD": MRDPODModel,
}


def build_model(region_data, cfg: dict[str, Any]):
    name = canonical_model_name(cfg["model"].get("name", "MRDP-OD"))
    cfg["model"]["name"] = name
    return MODEL_CLASSES[name](region_data, cfg)
