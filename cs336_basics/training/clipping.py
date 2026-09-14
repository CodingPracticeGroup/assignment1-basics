from __future__ import annotations

from collections.abc import Iterable

import torch


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    基于所有参数整体 L2 范数阈值的梯度截断（Gradient Clipping），原地修改 parameter.grad。

    性能要点：**在设备上一次性累加所有梯度的平方和**，只在最后 `if total_norm > ...`
    处做一次标量同步；缩放用 `torch._foreach_mul_` 融合。切忌对每个参数调用
    `.item()`——那会产生「参数个数」次 GPU→CPU 同步，是训练时的头号性能陷阱。
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm = torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in grads))
    if total_norm > max_l2_norm:
        # 加 1e-6 防止分母为 0
        clip_coef = (max_l2_norm / (total_norm + 1e-6)).item()
        torch._foreach_mul_(grads, clip_coef)
