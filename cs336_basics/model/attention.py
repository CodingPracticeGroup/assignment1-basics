"""Assignment 1 attention implementations.

Import-time dispatcher over `cs336_basics.notation.{einstein,no_einstein}.attention`,
selected by the `CS336_NOTATION` environment variable. See
`cs336_basics.notation` and `NOTATION.md`.
"""

from __future__ import annotations

from cs336_basics import notation

if notation.ACTIVE == notation.EINSTEIN:
    from cs336_basics.notation.einstein.attention import (
        CausalMultiHeadSelfAttention,
        RotaryPositionalEmbedding,
        scaled_dot_product_attention,
        softmax,
    )
else:
    from cs336_basics.notation.no_einstein.attention import (
        CausalMultiHeadSelfAttention,
        RotaryPositionalEmbedding,
        scaled_dot_product_attention,
        softmax,
    )

__all__ = [
    "softmax",
    "scaled_dot_product_attention",
    "RotaryPositionalEmbedding",
    "CausalMultiHeadSelfAttention",
]
