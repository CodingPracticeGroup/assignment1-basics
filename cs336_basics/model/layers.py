"""Assignment 1 layer implementations.

This is a thin, import-time dispatcher: the actual code lives in
`cs336_basics.notation.no_einstein.layers` (plain PyTorch) or
`cs336_basics.notation.einstein.layers` (Einstein notation), selected by the
`CS336_NOTATION` environment variable. See `cs336_basics.notation` and the
concept-repo `Stanford-CS336/Spring2026/assignments/assignment1/NOTATION.md`.
"""

from __future__ import annotations

from cs336_basics import notation

if notation.ACTIVE == notation.EINSTEIN:
    from cs336_basics.notation.einstein.layers import Embedding, Linear, RMSNorm, SwiGLU, silu
else:
    from cs336_basics.notation.no_einstein.layers import Embedding, Linear, RMSNorm, SwiGLU, silu

__all__ = ["Linear", "Embedding", "RMSNorm", "SwiGLU", "silu"]
