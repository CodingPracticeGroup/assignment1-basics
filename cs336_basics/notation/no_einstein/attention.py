"""Plain-PyTorch attention -- explicit matmul / reshape / transpose, no einops."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cs336_basics.notation.no_einstein.layers import Linear


__all__ = [
    "softmax",
    "scaled_dot_product_attention",
    "RotaryPositionalEmbedding",
    "CausalMultiHeadSelfAttention",
]


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """手写数值稳定的 Softmax：先减去最大值防止指数溢出。"""
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_vals)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """缩放点积注意力：Q (..., q, d_k)，K (..., k, d_k)，V (..., k, d_v)。"""
    d_k = Q.size(-1)

    # scores = Q K^T / sqrt(d_k) —— 显式矩阵乘 + 显式转置
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = softmax(scores, dim=-1)
    return torch.matmul(probs, V)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """旋转位置编码 (RoPE)。"""
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freqs = theta ** (-2.0 * torch.arange(d_k // 2, device=device).float() / d_k)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # angles: (..., seq_len, d_k // 2)
        angles = token_positions.unsqueeze(-1).float() * self.freqs
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # 奇偶切片配对后做 2D 坐标旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos_angles - x_odd * sin_angles
        out_odd = x_even * sin_angles + x_odd * cos_angles

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        """因果多头自注意力。"""
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 缓存的因果掩码（按 (s, device) 惰性重建，避免每次 forward 重新 tril）
        self._causal_mask: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, d = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (b, s, d) -> (b, h, s, d_k)：显式 reshape + transpose
        q = q.reshape(b, s, self.num_heads, self.d_k).transpose(1, 2)
        k = k.reshape(b, s, self.num_heads, self.d_k).transpose(1, 2)
        v = v.reshape(b, s, self.num_heads, self.d_v).transpose(1, 2)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
            q = rope(q, token_positions.unsqueeze(1))
            k = rope(k, token_positions.unsqueeze(1))

        # 下三角因果掩码（按 (s, device) 缓存）
        mask = self._causal_mask
        if mask is None or mask.size(0) != s or mask.device != x.device:
            mask = torch.tril(torch.ones((s, s), device=x.device, dtype=torch.bool))
            self._causal_mask = mask

        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        # (b, h, s, d_v) -> (b, s, d)：先转置回连续内存再 reshape
        out = attn_out.transpose(1, 2).contiguous().reshape(b, s, d)
        return self.output_proj(out)
