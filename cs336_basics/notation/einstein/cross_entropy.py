"""Einstein-notation cross-entropy -- einx.logsumexp + einx.get_at."""

from __future__ import annotations

import einx
import torch


__all__ = ["cross_entropy"]


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """交叉熵：用 einx.logsumexp 归约词表轴 [v]，用 einx.get_at 取出 target logit。"""
    flat_inputs = inputs.reshape(-1, inputs.size(-1))
    flat_targets = targets.reshape(-1)

    # 数值稳定的 log-sum-exp，沿词表轴 [v] 归约
    log_sum_exp = einx.logsumexp("n [v]", flat_inputs)

    # 沿词表轴 [v] 用 target 索引取值
    target_logits = einx.get_at("n [v], n -> n", flat_inputs, flat_targets)

    loss = -target_logits + log_sum_exp
    return torch.mean(loss)
