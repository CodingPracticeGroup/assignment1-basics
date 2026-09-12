"""Einstein-notation implementation of Assignment 1.

Uses exclusively the unified `einx` API: `einx.dot` for every contraction,
`einx.id` for axis (re)structuring, and `einx.softmax` / `einx.logsumexp` /
`einx.get_at` / `einx.mean` / `einx.multiply` for normalisation, gathering and
element-wise arithmetic. No `einops` and no `torch.einsum`.

See `cs336_basics.notation` for the runtime selector.
"""
