"""轻量实验日志：把每个指标点写成 CSV（零依赖、无需任何账号）。

列固定为：step, wall_time, split, loss, perplexity, lr, tokens_per_sec
- 训练点：split="train"，填 loss / perplexity / lr / tokens_per_sec
- 验证点：split="val"，填 loss / perplexity

这样即使没有 Weights and Biases 账号（或想离线），也能把 loss 曲线按梯度步 / 墙钟时间导出。
"""

from __future__ import annotations

import csv
import os
from typing import Any, TextIO

__all__ = ["CsvLogger"]

_FIELDS = ["step", "wall_time", "split", "loss", "perplexity", "lr", "tokens_per_sec"]


class CsvLogger:
    def __init__(self, path: str | os.PathLike) -> None:
        self.path = str(path)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file: TextIO = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_FIELDS)
        self._writer.writeheader()

    def log(self, **values: Any) -> None:
        """写一行；未提供的字段留空。"""
        self._writer.writerow({key: values.get(key, "") for key in _FIELDS})
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> CsvLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
