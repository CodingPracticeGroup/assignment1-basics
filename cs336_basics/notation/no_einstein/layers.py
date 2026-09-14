"""Plain-PyTorch layers -- explicit operators, no Einstein notation."""

from __future__ import annotations

import math

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

        # 截断正态初始化
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 行向量约定：y = x @ W^T，需要显式转置权重
        return x @ self.weight.t()


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """嵌入查找层。"""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 直接高级索引查表：等价于 einstein 版的
        # einx.get_at("[v] d, ... -> ... d", self.weight, token_ids)（见那边注释的逐段解释）
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """均方根层归一化 (RMSNorm)。"""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)

        # 沿最后一维手写均值、开方与除法
        mean_square = torch.mean(x_f32 ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        normalized = (x_f32 / rms).to(in_dtype)

        return self.weight * normalized


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
        return self.w2(gate * self.w3(x))
