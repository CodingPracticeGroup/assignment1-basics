from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    自 1D NumPy 数组语料库中随机采样输入 sequences 和对应偏移 1 位的前向预测 labels。
    """
    # 合法起点范围：[0, len - context_length - 1]。
    # labels 需要多取一位（start + context_length + 1 <= len），所以上界取开区间。
    max_idx = len(dataset) - context_length

    # 一次性采样所有起点，用高级索引一次取出整批（不再逐条切片 + torch.stack）。
    # 高级索引返回可写副本，因此也不会再触发只读 np.memmap 的 “not writable” 警告。
    starts = np.random.randint(0, max_idx, size=batch_size)
    offsets = np.arange(context_length + 1)
    idx = starts[:, None] + offsets[None, :]

    # 输入 x = 每行前 context_length 个；目标 y = 右移一位。.long() 保证 int64。
    x = torch.from_numpy(np.ascontiguousarray(dataset[idx[:, :-1]])).long()
    y = torch.from_numpy(np.ascontiguousarray(dataset[idx[:, 1:]])).long()

    # 投送到目标设备（cpu / mps / cuda:0）
    return x.to(device), y.to(device)
