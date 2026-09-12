"""Plain-PyTorch cross-entropy (no Einstein notation)."""

from __future__ import annotations

import torch


__all__ = ["cross_entropy"]


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """数值稳定的交叉熵：显式减最大值、显式 log-sum-exp、显式高级索引取 target。"""
    flat_inputs = inputs.reshape(-1, inputs.size(-1))
    flat_targets = targets.reshape(-1)

    max_vals = torch.max(flat_inputs, dim=-1, keepdim=True).values
    stable_inputs = flat_inputs - max_vals

    log_sum_exp = torch.log(torch.sum(torch.exp(stable_inputs), dim=-1))

    # 行索引 + 列索引的高级索引
    target_logits = stable_inputs[torch.arange(flat_targets.size(0)), flat_targets]

    loss = -target_logits + log_sum_exp
    return torch.mean(loss)
