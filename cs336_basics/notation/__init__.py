"""Runtime selection between the two Assignment 1 implementations.

Assignment 1 ships **two complete, numerically equivalent** implementations of
the model code (and of the cross-entropy loss):

* `no_einstein` -- plain PyTorch: `@` / `torch.matmul`, `reshape` /
  `transpose` / `permute` / slicing and manual index arithmetic.
* `einstein` -- Einstein notation everywhere, using exclusively the unified
  `einx` API: `einx.dot` for every contraction, `einx.id` for axis
  (re)structuring, and `einx.softmax` / `einx.logsumexp` / `einx.get_at` /
  `einx.mean` / `einx.multiply` for normalisation, gathering and element-wise
  arithmetic. (No `einops` and no `torch.einsum`.)

The `CS336_NOTATION` environment variable picks which one the canonical
`cs336_basics.model.*` and `cs336_basics.training.optimizers` modules
re-export. It must be set *before* the first import of `cs336_basics`.
Accepted values (case-insensitive; `-` and `_` are interchangeable):

* `no_einstein` -- aliases: `plain`, `explicit`, `none`
* `einstein`    -- aliases: `einsum`, `einops`, `elegant`

Examples
--------
Run the official test-suite against the plain implementation::

    CS336_NOTATION=no_einstein uv run pytest

Run it against the Einstein-notation implementation::

    CS336_NOTATION=einstein uv run pytest

See `NOTATION.md` in the assignment root for the full side-by-side mapping.
"""

from __future__ import annotations

import os

__all__ = ["NO_EINSTEIN", "EINSTEIN", "VALID", "DEFAULT", "ACTIVE", "active"]

NO_EINSTEIN = "no_einstein"
EINSTEIN = "einstein"
VALID = (NO_EINSTEIN, EINSTEIN)
DEFAULT = EINSTEIN

_ALIASES = {
    "no_einstein": NO_EINSTEIN,
    "noeinstein": NO_EINSTEIN,
    "plain": NO_EINSTEIN,
    "explicit": NO_EINSTEIN,
    "none": NO_EINSTEIN,
    "einstein": EINSTEIN,
    "einsum": EINSTEIN,
    "einops": EINSTEIN,
    "elegant": EINSTEIN,
}


def active() -> str:
    """Return the notation implementation selected by `CS336_NOTATION`."""
    raw = os.environ.get("CS336_NOTATION", DEFAULT).strip().lower().replace("-", "_")
    try:
        return _ALIASES[raw]
    except KeyError:
        valid = ", ".join(VALID)
        raise ValueError(f"Unknown CS336_NOTATION={raw!r}. Expected one of: {valid}.") from None


# Resolved once, when this package is first imported.
ACTIVE = active()
