from __future__ import annotations

import math
import torch
import torch.nn as nn
from einops import rearrange

from cs336_basics.model.layers import Linear


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    手写数值稳定的 Softmax 算子。
    在计算指数前，先减去对应维度上的最大值，防止溢出。
    """
    # 找到 dim 维度上的最大值并保持维度形状
    max_vals = torch.max(x, dim=dim, keepdim=True).values
    # 减去最大值后计算指数
    exp_x = torch.exp(x - max_vals)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    手写缩放点积注意力 (Scaled Dot-Product Attention) 算子。
    Q: (..., seq_len_q, d_k)
    K: (..., seq_len_k, d_k)
    V: (..., seq_len_k, d_v)
    mask: (..., seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)
    
    # 1. 计算 Q K^T 并缩放
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 如果存在掩码，将 mask 值为 False 处的得分填充为负无穷
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        
    # 3. 经过 Softmax 归一化并与 Value 矩阵做加权聚合
    probs = softmax(scores, dim=-1)
    return torch.matmul(probs, V)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        手写旋转位置编码 (RoPE) 层。
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 预先计算并缓存不同维度的偏置基数
        # theta_k = theta ** (-2k / d_k)
        freqs = theta ** (-2.0 * torch.arange(d_k // 2, device=device).float() / d_k)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        对输入 x 的奇偶项做 2D 坐标对旋转。
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len) 表示在序列中的绝对位置索引
        """
        # 计算每个位置对应维度的弧角
        # angles 形状为 (..., seq_len, d_k // 2)
        angles = token_positions.unsqueeze(-1).float() * self.freqs
        
        # 获取 cos 和 sin
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # 配对奇偶维度元素进行坐标旋转
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos_angles - x_odd * sin_angles
        out_odd = x_even * sin_angles + x_odd * cos_angles

        # 组装返回旋转后的结果
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        """
        因果多头自注意力机制 (Causal Multi-Head Self-Attention)。
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # 1. 定义 Q、K、V 投影层
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # 2. 输出投影层
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        b, s, d = x.shape

        # 1. 投影映射为 Query、Key、Value
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 分头并重塑形状为：(batch, num_heads, seq_len, head_dim)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # 3. 如果启用了 RoPE 位置编码，执行旋转
        if rope is not None:
            if token_positions is None:
                # 默认绝对位置索引 0..seq_len-1
                token_positions = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
            # 在分头状态下应用 RoPE（head 维度作为 batch 处理）
            q = rope(q, token_positions.unsqueeze(1))
            k = rope(k, token_positions.unsqueeze(1))

        # 4. 构建下三角因果注意力掩码 (Causal Mask)
        # 允许第 i 个 token 只能 attending to 第 j <= i 个 token
        mask = torch.tril(torch.ones((s, s), device=x.device, dtype=torch.bool))

        # 5. 执行 Scaled Dot-Product Attention
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        # 6. 合拢多头并将形状重构回 (batch, seq_len, d_model)
        out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.output_proj(out)
