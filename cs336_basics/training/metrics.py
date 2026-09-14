"""评测指标：困惑度 perplexity。

handout Eqn 18：对长度 m、每 token 交叉熵损失 l_1..l_m，
    perplexity = exp( (1/m) * sum_i l_i )
即「平均每 token 交叉熵」取 exp，等价于模型在每个位置平均面对多少种等概率选择。
"""

from __future__ import annotations

import math

import torch

from cs336_basics import notation

if notation.ACTIVE == notation.EINSTEIN:
    from cs336_basics.notation.einstein.cross_entropy import cross_entropy
else:
    from cs336_basics.notation.no_einstein.cross_entropy import cross_entropy

__all__ = ["perplexity", "perplexity_from_loss"]


def perplexity(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    困惑度 = exp(平均每 token 交叉熵)（handout Eqn 18）。

    inputs : logits，形状 (..., vocab_size)；
    targets: token id，形状 (...)（与 logits 的 batch 维对齐）。
    """
    return torch.exp(cross_entropy(inputs, targets))


def perplexity_from_loss(mean_loss: float | torch.Tensor) -> float:
    """由已经算好的「平均交叉熵损失」直接换算 perplexity = exp(loss)。"""
    return math.exp(float(mean_loss))
