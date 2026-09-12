from __future__ import annotations

import math
import torch
from torch.optim import Optimizer


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    手写数值稳定的交叉熵损失函数。
    支持任意 Batch 维度，最终输出平均 Loss 标量。
    """
    # 扁平化多维 batch，将其整理成 (N, vocab_size) 与 (N,)
    flat_inputs = inputs.reshape(-1, inputs.size(-1))
    flat_targets = targets.reshape(-1)

    # 每一行减去最大值以实现数值稳定
    max_vals = torch.max(flat_inputs, dim=-1, keepdim=True).values
    stable_inputs = flat_inputs - max_vals

    # Log-sum-exp 算子
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_inputs), dim=-1))

    # 获取 target 对应索引上的特征得分
    target_logits = stable_inputs[torch.arange(flat_targets.size(0)), flat_targets]

    # 平均交叉熵
    loss = -target_logits + log_sum_exp
    return torch.mean(loss)


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

                # 1. 权重衰减 (Weight Decay) 与梯度解耦应用
                p.data -= lr * wd * p.data

                # 2. 动量更新
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                
                # m = beta1 * m + (1 - beta1) * g
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                # v = beta2 * v + (1 - beta2) * g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 3. 偏差校正与自适应学习率调整
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # 4. 执行更新
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)

        return loss
