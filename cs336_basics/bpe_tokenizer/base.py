from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator

import regex as re

# 预分词正则表达式（GPT-2 经典风格）
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
compiled_pat = re.compile(PAT)


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2 的「字节 -> 可打印 unicode 字符」双射，用于把 0..255 的原始字节
    可逆地写进 JSON / 文本文件（否则 b'\x00' 之类的不可打印字节无法安全序列化）。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


# unicode 字符 -> 原始字节（from_files 反序列化用）
_UNICODE_TO_BYTE = {char: byte for byte, char in _gpt2_bytes_to_unicode().items()}


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        根据给定的词表、合并顺序以及（可选的）特殊 Token 列表，初始化 Tokenizer 实例。
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens) if special_tokens else []

        # 构建快速字节序列到 Token ID 的查表哈希字典
        self.byte_to_id = {v: k for k, v in vocab.items()}

        # 将 merges 链转换为相邻对的哈希优先级字典（合并顺序索引越小，优先级越高，需最先合并）
        self.merge_priorities = {pair: idx for idx, pair in enumerate(merges)}

        # 如果存在特殊 Token，编译一个高效匹配特殊 Token 的正则表达式
        if self.special_tokens:
            # 必须按照特殊 Token 字符串长度降序排列（由长到短拼接正则），避免贪心匹配产生子串截断错误
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped_specials = [re.escape(token) for token in sorted_specials]
            self.special_pattern = re.compile(f"({'|'.join(escaped_specials)})")
        else:
            self.special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """
        类方法：从磁盘上的 GPT-2 序列化文件加载并构造 Tokenizer。

        文件格式（与 GPT-2 / 官方测试 fixture 一致）：
          - vocab_filepath : JSON，{"<unicode 可打印形式的 token>": <int id>}
          - merges_filepath: 文本，每行 "tokenA tokenB"（空格分隔，按合并顺序排列）
        两个文件里的 token 都经过 GPT-2 的 byte<->unicode 双射编码，这里反解回原始 bytes。
        若 special_tokens 中有词表里不存在的项，则按传入顺序追加到词表末尾。
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab: dict[str, int] = json.load(f)
        vocab: dict[int, bytes] = {
            idx: bytes([_UNICODE_TO_BYTE[ch] for ch in token]) for token, idx in gpt2_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip()
                # 只跳过 GPT-2 官方 merges.txt 的版本头（"#version: ..."）；
                # 注意 "# e" 这类是**合法** merge（'#' 字节本身可参与合并），不能一并跳过。
                if not cleaned or cleaned.startswith("#version"):
                    continue
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    continue
                merges.append(tuple(bytes([_UNICODE_TO_BYTE[ch] for ch in part]) for part in parts))

        if special_tokens:
            existing = set(vocab.values())
            for special_token in special_tokens:
                encoded = special_token.encode("utf-8")
                if encoded not in existing:
                    vocab[len(vocab)] = encoded
                    existing.add(encoded)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        将输入的原始文本字符串编码为 Token ID 整数序列。
        """
        if not text:
            return []

        # 1. 根据特殊 Token 模式对文本进行拆分，保留特殊 Token 作为独立部分
        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]

        ids = []
        special_set = set(self.special_tokens)

        for part in parts:
            if not part:
                continue

            # 如果这部分恰好是一个特殊 Token，直接查表获取其对应 ID，不进行合并
            if part in special_set:
                part_bytes = part.encode("utf-8")
                if part_bytes in self.byte_to_id:
                    ids.append(self.byte_to_id[part_bytes])
            else:
                # 否则，对标准文本片段执行正则预分词，并对每一个子词执行高能 BPE 优先合并
                for match in compiled_pat.finditer(part):
                    word_bytes = match.group(0).encode("utf-8")
                    ids.extend(self._encode_word(word_bytes))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        返回一个惰性（Lazy）生成器，用于逐个产出 iterable 中字符串的 Token ID。
        通过按行或按块处理来节省内存，符合 1MB 常数级低显存测试。
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        将一串 Token ID 解码还原为自然语言文本。
        不合法的 UTF-8 字节序列会被优雅地替换为 U+FFFD（Unicode 替换字符 🙃）。
        """
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        all_bytes = b"".join(byte_parts)
        return all_bytes.decode("utf-8", errors="replace")

    def _encode_word(self, word_bytes: bytes) -> list[int]:
        """
        辅助方法：利用哈希优先级字典（self.merge_priorities）对单个预分词（子词字节流）进行基于优先级的 BPE 合并。
        时间复杂度为 $O(L)$，执行速度极快。
        """
        if not word_bytes:
            return []

        # 初始化词表示为单字节列表
        symbols = [bytes([b]) for b in word_bytes]

        while len(symbols) > 1:
            # 寻找当前相邻符号对中，合并优先级最高（在 self.merge_priorities 中索引最小）的一对
            best_pair = None
            best_priority = float("inf")
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                priority = self.merge_priorities.get(pair, float("inf"))
                if priority < best_priority:
                    best_priority = priority
                    best_pair = pair

            # 如果不存在任何可以继续合并的对，则退出循环
            if best_pair is None or best_priority == float("inf"):
                break

            # 贪心地从左到右合并所有等于 best_pair 的相邻对
            new_symbols = []
            i = 0
            best_0, best_1 = best_pair
            merged = best_0 + best_1
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_0 and symbols[i+1] == best_1:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        # 将最终合并后的各个子词字节流映射回它们的词表整数 ID
        return [self.byte_to_id[sym] for sym in symbols]
