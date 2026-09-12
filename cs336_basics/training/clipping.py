from __future__ import annotations

import math
from collections.abc import Iterable
import torch


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    基于所有参数整体 L2 范数阈值的梯度截断（Gradient Clipping）。
    修改 parameter.grad 的数据。
    """
    total_norm = 0.0
    # 1. 累加所有参数的平方和
    for p in parameters:
        if p.grad is not None:
            # 必须用 .detach() 或者 .data 获取，避免 PyTorch 构建无谓的梯度图
            param_norm = torch.sum(p.grad.detach() ** 2).item()
            total_norm += param_norm
            
    total_norm = math.sqrt(total_norm)

    # 2. 如果整体 L2 范数超过阈值，等比例缩减每一个参数的梯度
    if total_norm > max_l2_norm:
        # 添加 1e-6 的 epsilon 防止分母为 0 溢出
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)
