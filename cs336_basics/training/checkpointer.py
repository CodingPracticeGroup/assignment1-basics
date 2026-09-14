from __future__ import annotations

import os
from typing import BinaryIO, IO
import torch


def _strip_compiled_prefix(state_dict: dict) -> dict:
    """去掉 torch.compile 包装（OptimizedModule）产生的 `_orig_mod.` 前缀。

    `model = torch.compile(model)` 之后 `model.state_dict()` 的键会变成
    `_orig_mod.xxx`；这里统一剥掉，保证 compile / 非 compile 的 checkpoint 互通。
    """
    prefix = "_orig_mod."
    if any(str(key).startswith(prefix) for key in state_dict):
        return {str(key)[len(prefix):] if str(key).startswith(prefix) else key: value for key, value in state_dict.items()}
    return state_dict


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    对模型权重、优化器一二阶动量参数、以及当前迭代 step 步数进行快照序列化归档。

    函数名按 handout 规定（Problem checkpointing）。测试用的 adapter
    是 `tests/adapters.py` 里的 `run_save_checkpoint`，它只是调用本函数。
    """
    checkpoint = {
        "model_state_dict": _strip_compiled_prefix(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    自磁盘反序列化还原 checkpoint 快照至内存，热加载更新 model 与 optimizer 状态，并返回归档时的 iteration 步数。

    函数名按 handout 规定（Problem checkpointing）。测试用的 adapter
    是 `tests/adapters.py` 里的 `run_load_checkpoint`，它只是调用本函数。
    """
    # 采用 map_location="cpu" 是业界最安全的范式，能防止跨平台（mps/cuda/cpu）读取时的设备绑定卡死问题
    checkpoint = torch.load(src, map_location="cpu")
    
    model.load_state_dict(_strip_compiled_prefix(checkpoint["model_state_dict"]))
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["iteration"]
