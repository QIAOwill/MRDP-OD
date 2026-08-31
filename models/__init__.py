"""模型工厂与 MRDP-OD 主模型。"""
from .factory import build_model
from .mrdp_od import MRDPODModel

__all__ = ["build_model", "MRDPODModel"]
