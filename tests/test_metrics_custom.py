"""Extra self-test (not part of the official suite): perplexity == exp(cross-entropy)."""

from __future__ import annotations

import math

import torch

from cs336_basics.training.metrics import perplexity, perplexity_from_loss
from cs336_basics.training.optimizers import cross_entropy


def test_perplexity_matches_exp_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(8, 17)
    targets = torch.randint(0, 17, (8,))
    expected = math.exp(cross_entropy(logits, targets).item())
    assert abs(perplexity(logits, targets).item() - expected) < 1e-5
    assert abs(perplexity_from_loss(cross_entropy(logits, targets)) - expected) < 1e-5


def test_perplexity_uniform_equals_vocab_size():
    # logits 全 0 -> 均匀分布 -> loss = log(V) -> perplexity = V
    vocab = 11
    logits = torch.zeros(4, vocab)
    targets = torch.arange(4) % vocab
    assert abs(perplexity(logits, targets).item() - vocab) < 1e-4
