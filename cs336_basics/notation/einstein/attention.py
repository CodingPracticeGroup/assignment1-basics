"""Einstein-notation attention -- exclusively the unified einx API.

Operators used:
* `einx.id`       -- (b s (h d) -> b h s d) head split / pair split, i.e. the
                     einops.rearrange equivalent
* `einx.dot`      -- QK^T, PV, rotation-matrix contraction, i.e. the einsum equivalent
* `einx.softmax`  -- numerically stable softmax along a named axis

`einx` does not broadcast operands with different ranks inside `dot`, so
`_broadcast_leading` aligns the position angles to the query/key leading
dimensions before the rotation contraction.
"""

from __future__ import annotations

import math

import einx
import torch
import torch.nn as nn

from cs336_basics.notation.einstein.layers import Linear


__all__ = [
    "softmax",
    "scaled_dot_product_attention",
    "RotaryPositionalEmbedding",
    "CausalMultiHeadSelfAttention",
]


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """沿指定轴做 Softmax：用 einx 轴名表达要归约（[..]）的轴。"""
    dim = dim % x.ndim
    names = [f"a{i}" for i in range(x.ndim)]
    expression = " ".join(names[:dim] + [f"[{names[dim]}]"] + names[dim + 1 :])
    return einx.softmax(expression, x)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """缩放点积注意力：Q (..., q, d_k)，K (..., k, d_k)，V (..., k, d_v)。"""
    d_k = Q.size(-1)

    # scores = Q K^T / sqrt(d_k)：q/k 命名让转置消失在缩合里
    scores = einx.dot("... q d, ... k d -> ... q k", Q, K) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = softmax(scores, dim=-1)
    return einx.dot("... q k, ... k v -> ... q v", probs, V)


def _broadcast_leading(src: torch.Tensor, target_leading: tuple[int, ...]) -> torch.Tensor:
    """把 src 的前置 batch 维广播到 target_leading（序列/特征轴 's'/'p' 保持不变）。

    einx 要求张量秩与表达式的轴数一致，且不隐式跨秩广播，因此这里按
    numpy 的右对齐规则，用 `einx.id` 显式补出缺失/大小为 1 的前置轴，
    例如 (s, p) -> (b, s, p) 或 (b, 1, s, p) -> (b, h, s, p)。
    """
    src_leading = src.shape[:-2]
    target = tuple(target_leading)
    n_src, n_tgt = len(src_leading), len(target)
    assert n_src <= n_tgt, (tuple(src.shape), target)

    in_terms: list[str] = []
    out_terms: list[str] = []
    sizes: dict[str, int] = {}
    for i in range(n_tgt):
        j = i - (n_tgt - n_src)
        if j < 0:
            # 新补出来的前置轴
            name = f"z{i}"
            out_terms.append(name)
            sizes[name] = target[i]
        elif src_leading[j] == 1:
            # 大小为 1 的轴可以广播到目标大小
            name = f"z{i}"
            in_terms.append("1")
            out_terms.append(name)
            sizes[name] = target[i]
        else:
            assert src_leading[j] == target[i], (src_leading, target)
            name = f"q{j}"
            in_terms.append(name)
            out_terms.append(name)

    expression = " ".join(in_terms + ["s", "p"]) + " -> " + " ".join(out_terms + ["s", "p"])
    return einx.id(expression, src, **sizes)


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
        # angles: (..., seq_len, d_k // 2)，先对齐到 x 的前置维度
        angles = token_positions.unsqueeze(-1).float() * self.freqs
        angles = _broadcast_leading(angles, x.shape[:-2])
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # 把相邻特征轴重组为 d x pair，pair 轴大小为 2
        x_pairs = einx.id("... s (d pair) -> ... s d pair", x, pair=2)

        # 每个位置/特征对构造一个 2x2 旋转矩阵，再用 einx.dot 一次性完成旋转
        rotation = torch.stack(
            [
                torch.stack([cos_angles, -sin_angles], dim=-1),
                torch.stack([sin_angles, cos_angles], dim=-1),
            ],
            dim=-2,
        )
        out_pairs = einx.dot("... p i j, ... p j -> ... p i", rotation, x_pairs)
        return einx.id("... s d pair -> ... s (d pair)", out_pairs)


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

        # einx.id 一步完成投影后的分头：(b, s, d) -> (b, h, s, d_k)
        q = einx.id("b s (h d) -> b h s d", self.q_proj(x), h=self.num_heads)
        k = einx.id("b s (h d) -> b h s d", self.k_proj(x), h=self.num_heads)
        v = einx.id("b s (h d) -> b h s d", self.v_proj(x), h=self.num_heads)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
            token_positions = token_positions.unsqueeze(1)
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        # 下三角因果掩码（按 (s, device) 缓存）
        mask = self._causal_mask
        if mask is None or mask.size(0) != s or mask.device != x.device:
            mask = torch.tril(torch.ones((s, s), device=x.device, dtype=torch.bool))
            self._causal_mask = mask

        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        # 合拢多头：(b, h, s, d_v) -> (b, s, d)
        out = einx.id("b h s d -> b s (h d)", attn_out)
        return self.output_proj(out)
