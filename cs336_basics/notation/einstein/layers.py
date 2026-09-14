"""Einstein-notation layers -- exclusively the unified einx API.

`einx.dot` replaces `torch.einsum` / `einops.einsum`, and `einx.id`,
`einx.get_at`, `einx.mean`, `einx.multiply` cover the structure / indexing /
normalisation ops. No einops and no torch.einsum.
"""

from __future__ import annotations

import math

import einx
import torch
import torch.nn as nn


__all__ = ["Linear", "Embedding", "RMSNorm", "silu", "SwiGLU"]


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """无 Bias 线性变换层，权重 W 形状为 (out_features, in_features)。"""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 轴名即文档：无需手动转置 W，也自动支持任意前置 batch 维
        return einx.dot("... d_in, d_out d_in -> ... d_out", x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """嵌入查找层。"""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # einx.get_at = 广义「按索引取子张量」(gather / 高级索引)，是 einsum 的索引版对偶。
        #
        # 表达式 "[v] d, ... -> ... d" 按「左操作数, 右操作数 -> 输出」读：
        #   [v] d  —— 被索引的张量 self.weight，形状 (V, d)：
        #            [v] 用**方括号**标记「要沿这条轴、按整数下标取值」的轴（词表轴 V）；
        #            d   是每条索引取出的「行」的长度（embedding 维度），原样保留。
        #   ...    —— 索引张量 token_ids；... 表示任意多个前置 batch 维，
        #            其中每个整数都被当作 v 轴上的下标。
        #   ... d  —— 对每个下标取 weight 的第 i 行（d 维），再按索引张量原有的 batch 维拼回。
        #
        # 例：token_ids (B, S) -> 输出 (B, S, d)；token_ids (B,) -> (B, d)；token_ids 标量 -> (d,)
        # 大白话：token_ids 里每个数字 i，就取 self.weight 的第 i 行放回原位置。
        # 精确等价于 no_einstein 版的 self.weight[token_ids]（PyTorch 高级索引）。
        return einx.get_at("[v] d, ... -> ... d", self.weight, token_ids)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """均方根层归一化 (RMSNorm)。"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)

        # 沿特征轴 [d] 归约（([d]) 表示保留该轴），再用 einx.multiply 广播 gain
        mean_square = einx.mean("... ([d])", x_f32 ** 2)
        rms = torch.sqrt(mean_square + self.eps)
        normalized = (x_f32 / rms).to(in_dtype)

        return einx.multiply("... d, d -> ... d", normalized, self.weight)


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish) 激活函数：f(x) = x * sigmoid(x)。"""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """SwiGLU FFN：W2(SiLU(W1 x) * W3 x)。"""
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))
        # 沿特征轴 f 的逐元素门控乘
        return self.w2(einx.multiply("... f, ... f -> ... f", gate, self.w3(x)))
