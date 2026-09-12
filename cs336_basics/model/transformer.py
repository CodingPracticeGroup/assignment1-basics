from __future__ import annotations

import torch
import torch.nn as nn

from cs336_basics.model.layers import Embedding, RMSNorm, SwiGLU, Linear
from cs336_basics.model.attention import CausalMultiHeadSelfAttention, RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        """
        手写 Pre-Norm 结构的 Transformer Decoder Block。
        """
        super().__init__()
        # 第一个子层：RMSNorm 和多头自注意力机制
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        
        # 第二个子层：RMSNorm 和 SwiGLU Feed-Forward Network (FFN)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pre-norm 结构下的前向传递。
        """
        # 第一子层残差：z = x + Attention(RMSNorm(x))
        z = x + self.attn(self.ln1(x), rope=rope, token_positions=token_positions)
        
        # 第二子层残差：y = z + FFN(RMSNorm(z))
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
        """
        Decoder-only Transformer 语言模型。
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        # 1. 词嵌入查找层 (Embedding)
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # 2. 预先构造 RotaryPositionalEmbedding (RoPE)
        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(rope_theta, d_k, context_length, device=device)

        # 3. 构造 num_layers 层 Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        # 4. 尾部的 RMSNorm 层
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        # 5. LM Head 输出投影层
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        进行模型前向传递，返回预测的 logits 概率矩阵。
        token_ids: (batch_size, seq_len)
        """
        b, s = token_ids.shape
        
        # 生成对应的位置索引序列并对齐至设备
        token_positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, -1)

        # 查找词嵌入向量
        x = self.token_embeddings(token_ids)

        # 迭代通过多层 Transformer Block
        for layer in self.layers:
            x = layer(x, rope=self.rope, token_positions=token_positions)

        # 尾部均方根层归一化
        x = self.ln_final(x)

        # 通过 LM Head 投影为预测的未归一化 logits
        return self.lm_head(x)
