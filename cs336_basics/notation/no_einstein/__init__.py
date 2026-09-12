"""Plain-PyTorch implementation of Assignment 1 (no Einstein notation).

Contractions use `@` / `torch.matmul`, reshapes use `reshape` / `view` /
`transpose` / `permute`, normalisation uses `torch.mean` / `torch.sum`, and
gathering uses advanced indexing. See `cs336_basics.notation` for the runtime
selector.
"""
