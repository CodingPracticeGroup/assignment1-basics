from __future__ import annotations

import os
from typing import BinaryIO, IO
import torch


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    对模型权重、优化器一二阶动量参数、以及当前迭代 step 步数进行快照序列化归档。
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    自磁盘反序列化还原 checkpoint 快照至内存，热加载更新 model 与 optimizer 状态，并返回归档时的 iteration 步数。
    """
    # 采用 map_location="cpu" 是业界最安全的范式，能防止跨平台（mps/cuda/cpu）读取时的设备绑定卡死问题
    checkpoint = torch.load(src, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["iteration"]
