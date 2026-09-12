"""Assignment 1 Transformer implementations.

Import-time dispatcher over
`cs336_basics.notation.{einstein,no_einstein}.transformer`, selected by the
`CS336_NOTATION` environment variable. See `cs336_basics.notation` and
`NOTATION.md`.
"""

from __future__ import annotations

from cs336_basics import notation

if notation.ACTIVE == notation.EINSTEIN:
    from cs336_basics.notation.einstein.transformer import BasicsTransformerLM, TransformerBlock
else:
    from cs336_basics.notation.no_einstein.transformer import BasicsTransformerLM, TransformerBlock

__all__ = ["TransformerBlock", "BasicsTransformerLM"]
