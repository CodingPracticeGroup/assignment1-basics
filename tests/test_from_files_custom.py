"""Extra self-test (not part of the official suite): Tokenizer.from_files.

Verifies that loading the GPT-2 fixture via `Tokenizer.from_files` reproduces the
tokenizer built by the official `tests/test_tokenizer.py` loader.
"""

from __future__ import annotations

from cs336_basics.bpe_tokenizer.base import Tokenizer

from .common import FIXTURES_PATH

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


def test_from_files_matches_fixture_loader():
    from .test_tokenizer import get_tokenizer_from_vocab_merges_path

    reference = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    loaded = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH)

    assert loaded.vocab == reference.vocab
    assert loaded.merges == reference.merges
    for text in ["hello world", "中文测试 123", "I'll go", "#hashtag ##md", "🙃 emoji"]:
        assert loaded.encode(text) == reference.encode(text)


def test_from_files_with_special_tokens_roundtrip():
    loaded = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])
    text = "a<|endoftext|>b 中文<|endoftext|>c"
    assert loaded.decode(loaded.encode(text)) == text
