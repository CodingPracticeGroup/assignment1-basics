
"""把 handout 7.3 的消融变体应用到模型上（实验用；默认 baseline 不动）。

支持的 variant：
  - baseline   : 不变（pre-norm + RMSNorm + RoPE + SwiGLU）
  - no_rmsnorm : 用 nn.Identity 替换所有 RMSNorm
  - post_norm  : 把 TransformerBlock 改成 post-norm
  - no_rope    : 去掉 RoPE（model.rope = None）
  - silu       : 把 SwiGLU 换成 2 矩阵的 SiLU FFN，d_ff 取 1.5x 以匹配参数量
"""

from __future__ import annotations

import torch.nn as nn


def _replace(model: nn.Module, pred, factory) -> None:
    for name, module in list(model.named_modules()):
        if pred(module):
            parent_path, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, attr, factory(module))


def apply_variant(model: nn.Module, variant: str) -> nn.Module:
    variant = (variant or "baseline").lower()
    if variant in ("baseline", "none"):
        return model

    if variant == "no_rope":
        model.rope = None
        return model

    if variant == "no_rmsnorm":
        from cs336_basics.model.layers import RMSNorm

        _replace(model, lambda m: isinstance(m, RMSNorm), lambda m: nn.Identity())
        return model

    if variant == "post_norm":
        from cs336_basics.model.transformer import TransformerBlock

        def forward(self, x, rope=None, token_positions=None):
            z = self.ln1(x + self.attn(x, rope=rope, token_positions=token_positions))
            return self.ln2(z + self.ffn(z))

        TransformerBlock.forward = forward
        return model

    if variant == "silu":
        from cs336_basics.model.layers import Linear

        class SiluFFN(nn.Module):
            def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
                super().__init__()
                self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
                self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

            def forward(self, x):
                return self.w2(nn.functional.silu(self.w1(x)))

        for layer in model.layers:
            old = layer.ffn
            d_model = old.w1.in_features
            d_ff = int(round(1.5 * old.w1.out_features))  # 2*d_model*d_ff' = 3*d_model*d_ff
            ref = next(old.parameters())
            layer.ffn = SiluFFN(d_model, d_ff, device=ref.device, dtype=ref.dtype)
        return model

    raise ValueError(f"unknown variant: {variant}")
