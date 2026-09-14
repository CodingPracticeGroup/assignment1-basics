from __future__ import annotations

import math
import torch
from torch.optim import Optimizer

from cs336_basics import notation

if notation.ACTIVE == notation.EINSTEIN:
    from cs336_basics.notation.einstein.cross_entropy import cross_entropy
else:
    from cs336_basics.notation.no_einstein.cross_entropy import cross_entropy

__all__ = ["cross_entropy", "AdamW"]


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        """
        手写状态 AdamW 优化器，符合 PyTorch 优化器规范。
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        执行单个权重参数优化更新步骤。
        """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 初始化状态变量：Step, First Moment (exp_avg), Second Moment (exp_avg_sq)
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                # 更新迭代计数
                state["step"] += 1
                t = state["step"]

                # 1. 权重衰减 (Weight Decay) 与梯度解耦应用（乘法形式，与 torch.optim.AdamW 一致）
                p.data.mul_(1.0 - lr * wd)

                # 2. 动量更新
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # m = beta1 * m + (1 - beta1) * g
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # v = beta2 * v + (1 - beta2) * g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. 偏差校正与自适应学习率调整
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2_sqrt = math.sqrt(1.0 - beta2 ** t)

                # 4. 执行更新（eps 放置与 torch.optim.AdamW 完全一致：denom = sqrt(v)/sqrt(bc2) + eps）
                step_size = lr / bias_correction1
                denom = torch.sqrt(v) / bias_correction2_sqrt + eps
                p.data -= step_size * m / denom

        return loss
