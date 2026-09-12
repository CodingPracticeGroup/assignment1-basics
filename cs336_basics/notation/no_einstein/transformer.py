"""Plain-PyTorch Transformer -- composes the no_einstein submodules."""

from __future__ import annotations

import torch
import torch.nn as nn

from cs336_basics.notation.no_einstein.attention import (
    CausalMultiHeadSelfAttention,
    RotaryPositionalEmbedding,
)
from cs336_basics.notation.no_einstein.layers import Embedding, Linear, RMSNorm, SwiGLU


__all__ = ["TransformerBlock", "BasicsTransformerLM"]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        """Pre-Norm Transformer Decoder Block。"""
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = x + self.attn(self.ln1(x), rope=rope, token_positions=token_positions)
        return z + self.ffn(self.ln2(z))


class BasicsTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        """Decoder-only Transformer 语言模型。"""
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(rope_theta, d_k, context_length, device=device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape
        token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, -1)

        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, rope=self.rope, token_positions=token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)
