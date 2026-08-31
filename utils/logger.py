"""同时输出到终端和运行日志文件的轻量日志器。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

_LOG_PATH: Path | None = None


def set_log_file(path: str | Path | None) -> None:
    """设置当前进程的日志文件。"""
    global _LOG_PATH
    _LOG_PATH = Path(path) if path is not None else None
    if _LOG_PATH is not None:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    """输出带时间戳的日志。"""
    text = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    # tqdm.write 会先临时擦除活动进度条、输出日志，再把进度条重绘到最下方，
    # 避免普通 print 把尚未关闭的 100% 进度条冲掉或与日志挤在同一行。
    tqdm.write(text)
    if _LOG_PATH is not None:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
