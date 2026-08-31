"""不依赖 PyTorch 的模型标准名称与结果目录映射。"""
from __future__ import annotations


RESULT_FOLDERS = {
    "MRDP-OD": "MRDP-OD",
}

ALIASES = {
    "mrdp_od": "MRDP-OD",
    "mrdp-od": "MRDP-OD",
    "mrdpod": "MRDP-OD",
}


def canonical_model_name(name: str) -> str:
    """将配置中的别名转换为论文和结果目录使用的标准模型名。"""
    text = str(name).strip()
    if text in RESULT_FOLDERS:
        return text
    key = text.lower().replace(" ", "_").replace("__", "_")
    if key in ALIASES:
        return ALIASES[key]
    key2 = text.lower()
    if key2 in ALIASES:
        return ALIASES[key2]
    raise ValueError(f"未知模型名称：{name!r}")
