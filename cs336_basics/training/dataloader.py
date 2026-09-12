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
    # 限制起始位置最大上限，防止切片越界
    max_idx = len(dataset) - context_length
    
    # 随机生成 batch_size 个起始位置
    starts = np.random.randint(0, max_idx, size=batch_size)

    x_list = []
    y_list = []

    for start in starts:
        # 输入 x 从 start 切片到 start + context_length
        # 使用 torch.from_numpy 零拷贝桥接，随后调用 .long() 保证为 LongTensor (int64)
        x_seg = torch.from_numpy(dataset[start : start + context_length]).long()
        # 目标 y 向右偏移一位，从 start + 1 切片到 start + context_length + 1
        y_seg = torch.from_numpy(dataset[start + 1 : start + context_length + 1]).long()
        
        x_list.append(x_seg)
        y_list.append(y_seg)

    # 堆叠并投送到指定的目标计算设备（如 cpu、mps 或 cuda:0）上
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    
    return x, y
