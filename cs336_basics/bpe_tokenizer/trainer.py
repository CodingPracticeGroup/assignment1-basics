from __future__ import annotations

import os
import regex as re
from collections import defaultdict
import multiprocessing

# ===================================================================
# 🎯 大白话拆解 GPT-2 官方“切片机”正则表达式 PAT
# ===================================================================
# 这个正则表达式极其固执：它的核心原则是“宁可切碎，也绝不把不相关的字/标点强行粘在一起”。
# 内部通过 | (或) 符分立了 5 大分支，匹配时按从左到右的优先级：
# 1. '(?:[sdmt]|ll|ve|re) : 匹配英语里的所有常见简写，如 's, 'd, 'm, 't, 'll, 've, 're。
# 2. ?\p{L}+              : 匹配单词/文字（包括英、中、日、韩、德等所有文字）。
#                           前面的问号代表可选的前置空格。这保证了单词前面的空格会和单词黏在一起，不被单拆。
# 3. ?\p{N}+              : 匹配数字（如 123）。数字前的可选空格也会黏在一起。
# 4. ?[^\s\p{L}\p{N}]+    : 匹配除空白、字母、数字外的所有乱七八糟的标点/特殊符号（如 !@# 等）。
# 5. \s+(?!\S)|\s+        : 匹配纯空格、回车、换行、制表符等。
#
# -------------------------------------------------------------------
# 💡 经典运行匹配示例：
# -------------------------------------------------------------------
# 示例一：缩写片段 —— "I'll study"
#   - 切分结果：['I', "'ll", ' study']
#   - 拆解："I" 匹配字母组；"'ll" 命中规则 1 (英语简写)；" study" 命中字母组 (前置空格与单词黏在一起)。
#
# 示例二：中英数标点混合 —— "Room 101, 牛!"
#   - 切分结果：['Room', ' 101', ',', ' ', '牛', '!']
#   - 拆解：
#     - "Room" 命中字母组；
#     - " 101" 命中数字组 3 (前置空格与数字 101 黏在一起)；
#     - "," 命中标点组 4；
#     - " " (空格) 后面是汉字，但因为汉字 "牛" 也是字母，" 牛" 应该属于一类，
#       不过如果句子中包含多重空格，独立的 " " 会命中规则 5 (纯空格)；
#     - "牛" 命中字母组（汉字符合 \p{L} 字符集）；
#     - "!" 命中符号组 4。
#
# 示例三：连续多空格 —— "hello   world" (三个空格)
#   - 切分结果：['hello', '  ', ' world']
#   - 拆解："hello" 命中字母组；" world" 命中字母组并将最邻近的一个前置空格吸附黏在一起；
#     剩下的两个连续空格命中规则 5 (纯空格) 输出为 '  '。
# ===================================================================
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
compiled_pat = re.compile(PAT)


def pretokenize_text(text: str) -> dict[tuple[bytes, ...], int]:
    """
    根据正则表达式 PAT 匹配文本中的所有预分词（pre-tokens），并统计字节元组（byte-tuple）的频次。

    ===================================================================
    🎯 极致清晰的中英文夹杂句子 "Hi 牛" 运行实例拆解：
    ===================================================================
    输入：text = "Hi 牛" （英文字母 "Hi"、一个空格 " "、以及汉字 "牛"）

    1. 正则雷达扫描（compiled_pat.finditer）:
       - 匹配项 1: "Hi"（匹配 ?\\p{L}+ 规则）
       - 匹配项 2: " 牛"（匹配 ?\\p{L}+ 规则，前置空格会与汉字 "牛" 黏在一起不被单拆）

    2. 处理第一个匹配项 "Hi"：
       - match.group(0): "Hi"
       - .encode("utf-8"): 翻译为底层 UTF-8 字节串 b"Hi" (底层字节整数为 [72, 105])
       - bytes([b]): 拆碎为单个独立单字节串 b"H", b"i"
       - tuple(...): 塞入不可变元组 (b"H", b"i")

    3. 处理第二个匹配项 " 牛"：
       - match.group(0): " 牛"
       - .encode("utf-8"): 翻译为底层 UTF-8 字节串 b" \\xe7\\x89\\x9b" (底层字节整数为 [32, 231, 137, 155])
       - bytes([b]): 拆碎为 4 个独立单字节串 b" ", b"\\xe7", b"\\x89", b"\\x9b"
       - tuple(...): 塞入不可变元组 (b" ", b"\\xe7", b"\\x89", b"\\x9b")

    4. freqs 最终哈希表的输出结果：
       {
           (b"H", b"i"): 1,
           (b" ", b"\\xe7", b"\\x89", b"\\x9b"): 1
       }
    ===================================================================
    """
    freqs = defaultdict(int)
    
    # ===================================================================
    # 🎯 逐行大白话拆解 match 雷达扫描循环：
    # ===================================================================
    # 1. finditer(text): 正则引擎里最省内存的生成器。它像雷达扫雷一样从左到右扫描 text，
    #    一旦发现符合 PAT 的片段，就拿 match 对象裹着它“吐”出来，用一个处理一个，不占内存。
    # 2. match.group(0): 获取雷达扫描到的这个单词文本字符串（如 "hello"）。
    # 3. .encode("utf-8"): 将 Unicode 字符串翻译编码为最底层的 UTF-8 字节串（如 b'hello'）。
    #    英文 "hello" 会转成 5 字节，中文 "牛" 会转成 3 字节（b'\xe7\x89\x9b'），没有 OOV 问题。
    # 4. bytes([b]) for b in word_bytes: 将整个字节串打碎，把每一个字节（0-255整数）单独
    #    包装成长为 1 的独立小字节流（如 b'h', b'e', b'l'...）。
    # 5. tuple(...): 塞入不可变的元组：(b'h', b'e', b'l', b'l', b'o')。
    #    由于 BPE 只能合并相邻字符，用 tuple 承载是最快、最完美的哈希键（Keys）。
    # ===================================================================
    for match in compiled_pat.finditer(text):
        word_bytes = match.group(0).encode("utf-8")
        word_tuple = tuple(bytes([b]) for b in word_bytes)
        freqs[word_tuple] += 1
    return freqs


def pre_merge(
    corpus: str,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """
    把整篇语料变成「词型（byte-tuple）-> 频次」的统计表，供 BPE 合并阶段使用。

    这就是 pre-tokenization（预分词），也叫 pre-merge：它是「主要 tokenize / 合并
    之前」的准备步骤。输入一篇 string 语料，输出可直接喂给合并循环的词频表。

    步骤：
    1. 特殊 Token 硬分割：re.escape 转义后按「或」拼成正则，遇到特殊 Token 就把
       语料拦腰切断，保证 BPE 绝不跨越文档边界瞎合并。
    2. 对每个切片做正则预分词（pretokenize_text），统计字节元组词型频次。
    3. 切片够大且不止一个时，用 multiprocessing 并行统计，再合并各进程结果。
    """
    # 1. 特殊 Token 硬分割（hard boundaries）
    if special_tokens:
        escaped_specials = [re.escape(token) for token in special_tokens]
        split_pattern = re.compile("|".join(escaped_specials))
        pieces = split_pattern.split(corpus)
    else:
        pieces = [corpus]

    pieces = [p for p in pieces if p]
    if not pieces:
        return {}

    # 2. 预分词并汇总频次（大语料用多进程：每个 worker 动态抢任务）
    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)
    total_len = sum(len(p) for p in pieces)

    if total_len > 500_000 and len(pieces) > 1:
        num_workers = min(multiprocessing.cpu_count(), 8)
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(pretokenize_text, pieces)
        for res in results:
            for k, v in res.items():
                word_freqs[k] += v
    else:
        for piece in pieces:
            for k, v in pretokenize_text(piece).items():
                word_freqs[k] += v

    return word_freqs


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    在输入语料库上训练字节级（byte-level）的 BPE 分词器，并输出最终词表与合并顺序。
    """
    # 1. 读取输入文本文件
    with open(input_path, encoding="utf-8", errors="ignore") as f:
        corpus = f.read()

    # 2. 预分词（pre-tokenization / pre-merge）：特殊 Token 硬分割 + 正则预分词 + 频次汇总
    word_freqs = pre_merge(corpus, special_tokens)

    # 3. 执行 BPE 合并循环
    merges: list[tuple[bytes, bytes]] = []
    
    # 初始化统计所有相邻字节/Token对的频次
    pair_freqs = defaultdict(int)
    for word_tuple, count in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            pair_freqs[(word_tuple[i], word_tuple[i+1])] += count

    # ===================================================================
    # 🎯 为什么我们需要合并的步数是 vocab_size - 256 - len(special_tokens)？
    # ===================================================================
    # 最终词表大小（vocab_size）是由以下三个互不重叠、界限分明的部分组成的：
    #   1. 基础单字节：固定 256 个（b'\x00' 到 b'\xff'，保证 100% 拒绝 OOV 错误）
    #   2. 特殊 Token：len(special_tokens) 个（如 <|endoftext|>，单独占坑）
    #   3. BPE 合并产生的新词：M 个（每次合并 1 对相邻 Token，就诞生且仅诞生 1 个新词）
    #
    # 极简公式关系：
    #   最终词表大小 (vocab_size) = 256 + 特殊 Token 数量 + 合并次数 (M)
    #
    # 反推合并次数：
    #   合并次数 (M) = vocab_size - 256 - 特殊 Token 数量
    #
    # 例如：vocab_size = 500，special_tokens 数量为 1，则合并次数 = 500 - 256 - 1 = 243 次。
    # ===================================================================
    num_merges = vocab_size - 256 - len(special_tokens)
    
    for _ in range(num_merges):
        if not pair_freqs:
            break

        # 寻找频次最高的相邻对，若有并列最高频次，按照 tuple 字典序最大者（max）打破平局（Tie-breaking）
        max_freq = -1
        best_pair = None
        for pair, freq in pair_freqs.items():
            if freq > max_freq:
                max_freq = freq
                best_pair = pair
            elif freq == max_freq:
                if pair > best_pair:
                    best_pair = pair

        if max_freq <= 0 or best_pair is None:
            break

        merges.append(best_pair)

        # 在词频表中对最佳对进行增量合并更新
        best_0, best_1 = best_pair
        merged = best_0 + best_1

        new_word_freqs = defaultdict(int)
        for word_tuple, count in word_freqs.items():
            # 只有当词元组中同时包含要合并的两个元素时，才进行重构
            if best_0 in word_tuple and best_1 in word_tuple:
                new_tuple = []
                i = 0
                changed = False
                while i < len(word_tuple):
                    if i < len(word_tuple) - 1 and word_tuple[i] == best_0 and word_tuple[i+1] == best_1:
                        new_tuple.append(merged)
                        i += 2
                        changed = True
                    else:
                        new_tuple.append(word_tuple[i])
                        i += 1
                if changed:
                    t = tuple(new_tuple)
                    new_word_freqs[t] += count
                    
                    # 极其关键的性能优化：仅增量更新受影响词的相邻对频次（避免 $O(N^2)$ 全量统计）
                    # 扣除旧词元组产生的相邻对频次
                    for j in range(len(word_tuple) - 1):
                        pair_freqs[(word_tuple[j], word_tuple[j+1])] -= count
                    # 累加新词元组产生的相邻对频次
                    for j in range(len(t) - 1):
                        pair_freqs[(t[j], t[j+1])] += count
                else:
                    new_word_freqs[word_tuple] += count
            else:
                new_word_freqs[word_tuple] += count

        word_freqs = new_word_freqs

        # 及时清理掉频次已经降为 0 或以下的相邻对，缩减字典体积，提升查找速度
        keys_to_del = [pair for pair, freq in pair_freqs.items() if freq <= 0]
        for k in keys_to_del:
            del pair_freqs[k]

    # 5. 构筑并输出词表：
    # 256 个基础单字节 + 特殊 Token 列表 + 训练学习出的 BPE merges
    vocab = {}
    
    # 注入标准基础单字节 0..255
    for b in range(256):
        vocab[b] = bytes([b])
        
    curr_id = 256
    # 注入特殊 Token
    for s_token in special_tokens:
        vocab[curr_id] = s_token.encode("utf-8")
        curr_id += 1
        
    # 注入合并产物
    for pair in merges:
        vocab[curr_id] = pair[0] + pair[1]
        curr_id += 1

    return vocab, merges
