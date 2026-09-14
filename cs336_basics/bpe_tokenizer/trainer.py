from __future__ import annotations

import heapq
import multiprocessing
import os
from collections import defaultdict

import regex as re

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

# 单字节 token 查表（0..255 -> b"\x00"..b"\xff"），模块层复用，避免热路径反复分配
_BYTE_TOKENS = tuple(bytes([b]) for b in range(256))
_wt_get = _BYTE_TOKENS.__getitem__

# pre-token 字符串 -> 字节元组的缓存：预分词里同一个词会重复出现成千上万次，
# 缓存把「构造字节元组」的成本从 O(出现次数) 降到 O(唯一词型数)。
# 每个 worker 进程各有一份；设上限防止超长语料上无限增长。
_word_tuple_cache: dict[str, tuple[bytes, ...]] = {}
_WORD_TUPLE_CACHE_LIMIT = 1 << 20


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
    cache = _word_tuple_cache
    cache_limit = _WORD_TUPLE_CACHE_LIMIT

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
        word = match.group(0)
        word_tuple = cache.get(word)
        if word_tuple is None:
            word_tuple = tuple(map(_wt_get, word.encode("utf-8")))
            if len(cache) < cache_limit:
                cache[word] = word_tuple
        freqs[word_tuple] += 1
    return freqs


def pre_merge_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    chunk_bytes: int = 1 << 27,
    num_workers: int | None = None,
) -> dict[tuple[bytes, ...], int]:
    """
    从文件做 pre-tokenization（pre-merge）：把语料变成「词型（byte-tuple）-> 频次」，
    供 BPE 合并阶段使用。

    有特殊 Token 时**流式**处理（内存有界）：每次读一个 chunk，拼上上一轮留下的尾巴，
    从缓冲区里找到最后一个特殊 Token，把它之前（含）的部分当作完整文本预分词；它之后的
    尾巴留到下一轮。切点紧跟 ASCII 特殊 Token，天然是合法 UTF-8 边界，不需要处理字符被
    切开的问题；尾巴实测最多 ~164 KiB（见 DATA.md 7.4），且不需要 seek（可用于 pipe）。

    没有特殊 Token 时没有安全切点，只能整篇读入（本作业的语料一定含 <|endoftext|>）。
    """
    chunk_bytes = max(chunk_bytes, 1 << 20)
    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)
    pool = None

    def _accumulate(freqs: dict[tuple[bytes, ...], int]) -> None:
        for key, value in freqs.items():
            word_freqs[key] += value

    def _pretokenize_pieces(pieces: list[str]) -> None:
        nonlocal pool
        if len(pieces) > 1 and sum(len(piece) for piece in pieces) > 500_000:
            if pool is None:
                pool = multiprocessing.Pool(processes=num_workers)
            # 用 imap_unordered 边完成边累加，避免 pool.map 把整个 chunk 的结果列表一次性驻留内存；
            # 频次是求和，与顺序无关，所以无序返回不影响结果。
            chunksize = max(1, len(pieces) // (num_workers * 4))
            for result in pool.imap_unordered(pretokenize_text, pieces, chunksize=chunksize):
                _accumulate(result)
        else:
            for piece in pieces:
                _accumulate(pretokenize_text(piece))

    if not special_tokens:
        # 没有安全切点，整篇读入（本作业数据一定含特殊 Token，这里是兜底）
        with open(input_path, encoding="utf-8", errors="ignore") as f:
            _pretokenize_pieces([f.read()])
    else:
        special_bytes = [token.encode("utf-8") for token in special_tokens]
        split_pattern = re.compile("|".join(re.escape(token) for token in special_tokens))
        remainder = b""
        with open(input_path, "rb") as f:
            while True:
                chunk = f.read(chunk_bytes)
                eof = chunk == b""
                buffer = remainder + chunk

                if not eof:
                    # 切在缓冲区里最后一个特殊 Token 之后，尾巴留到下一轮
                    cut = -1
                    for token in special_bytes:
                        pos = buffer.rfind(token)
                        if pos != -1:
                            cut = max(cut, pos + len(token))
                    if cut == -1:
                        remainder = buffer
                        continue
                    process, remainder = buffer[:cut], buffer[cut:]
                else:
                    process, remainder = buffer, b""

                if process:
                    text = process.decode("utf-8", errors="ignore")
                    _pretokenize_pieces([p for p in split_pattern.split(text) if p])

                if eof:
                    break

    if pool is not None:
        pool.close()
        pool.join()

    return word_freqs


class _MaxPair:
    """
    heapq 是小顶堆，但我们想每次弹出「频次最高；并列时 pair 字典序最大」的对。

    因此把比较键「反向」定义：频次越高、pair 越大，就在 __lt__ 下越「小」，从而被优先弹出。
    堆里允许存放**过期记录**（stale），取出时用当前 pair_freqs 校验（lazy invalidation）。
    """

    __slots__ = ("freq", "pair")

    def __init__(self, freq: int, pair: tuple[bytes, bytes]) -> None:
        self.freq = freq
        self.pair = pair

    def __lt__(self, other: _MaxPair) -> bool:
        if self.freq != other.freq:
            return self.freq > other.freq  # 频次高者视为更「小」-> 先弹出
        return self.pair > other.pair  # 并列时 pair 字典序大者视为更「小」-> 先弹出


def _push_updates(
    heap: list[_MaxPair],
    pair_freqs: dict[tuple[bytes, bytes], int],
    touched: set[tuple[bytes, bytes]],
) -> None:
    """把本轮频次发生变化的 pair 的**当前**频次重新入堆（旧记录成为 stale，取出时丢弃）。"""
    for pair in touched:
        freq = pair_freqs.get(pair, 0)
        if freq > 0:
            heapq.heappush(heap, _MaxPair(freq, pair))


def apply_merge(
    word_freqs: dict[tuple[bytes, ...], int],
    pair_freqs: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    best_pair: tuple[bytes, bytes],
) -> tuple[
    dict[tuple[bytes, ...], int],
    dict[tuple[bytes, bytes], int],
    dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    set[tuple[bytes, bytes]],
]:
    """
    把一次 merge 应用到当前状态，返回下一轮循环需要的
    (word_freqs, pair_freqs, pair_to_words)。

    用**倒排索引** pair_to_words（pair -> 含它的词型集合）只遍历「真正包含 best_pair」
    的词型，不再每轮扫描全部词型。对每个受影响的词型做增量更新：
      - 扣除旧词型贡献的相邻对频次，并从倒排索引里摘掉它；
      - 重写词型：把所有相邻的 best_pair 合并成 merged；
      - 累加新词型贡献的相邻对频次，并把它加入倒排索引；
    最后清理频次已经降为 0 或以下的相邻对（连同倒排索引项）。

    第 4 个返回值是本次**频次发生过变化**的相邻对集合 touched，供最大堆做增量入堆。
    """
    best_0, best_1 = best_pair
    merged = best_0 + best_1

    affected = list(pair_to_words.get(best_pair, ()))
    touched: set[tuple[bytes, bytes]] = set()

    for word_tuple in affected:
        count = word_freqs.pop(word_tuple, 0)

        # 扣除旧词型贡献的相邻对频次，并从倒排索引里摘掉它
        for j in range(len(word_tuple) - 1):
            pair = (word_tuple[j], word_tuple[j + 1])
            touched.add(pair)
            pair_freqs[pair] -= count
            # 不变量：当前词型里的 pair 一定已在倒排索引中（初始化时加入，清理只删频次为 0 的 pair）
            pair_to_words[pair].discard(word_tuple)

        # 重写词型：把所有相邻的 best_pair 合并成 merged
        new_tuple = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == best_0 and word_tuple[i + 1] == best_1:
                new_tuple.append(merged)
                i += 2
            else:
                new_tuple.append(word_tuple[i])
                i += 1
        t = tuple(new_tuple)

        # 累加新词型贡献的相邻对频次，并把它加入倒排索引
        word_freqs[t] = word_freqs.get(t, 0) + count
        for j in range(len(t) - 1):
            pair = (t[j], t[j + 1])
            touched.add(pair)
            pair_freqs[pair] += count
            pair_to_words.setdefault(pair, set()).add(t)

    # 及时清理掉频次已经降为 0 或以下的相邻对，缩减字典体积，提升查找速度
    for pair in touched:
        if pair_freqs.get(pair, 0) <= 0:
            pair_freqs.pop(pair, None)
            pair_to_words.pop(pair, None)

    # touched 是「频次发生过变化」的相邻对集合，供堆（lazy invalidation）重新入堆
    return word_freqs, pair_freqs, pair_to_words, touched


def build_vocab(
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> dict[int, bytes]:
    """
    合并（merge）结束后的词表组装（post-merge）：把三部分按固定顺序拼成最终 vocab。

    最终词表 = 256 个基础单字节 + 特殊 Token + 训练学出的 BPE merges，ID 分配顺序：
      - 0..255          : 标准单字节 0..255
      - 256..256+|S|-1  : special_tokens（按传入顺序）
      - 其后            : 每次合并产生的新 token（按 merges 顺序）
    """
    vocab: dict[int, bytes] = {}

    # 1. 标准基础单字节 0..255
    for b in range(256):
        vocab[b] = bytes([b])

    # 2. 特殊 Token
    curr_id = 256
    for s_token in special_tokens:
        vocab[curr_id] = s_token.encode("utf-8")
        curr_id += 1

    # 3. 合并产物
    for pair in merges:
        vocab[curr_id] = pair[0] + pair[1]
        curr_id += 1

    return vocab


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    在输入语料库上训练字节级（byte-level）的 BPE 分词器，并输出最终词表与合并顺序。
    """
    # 1+2. 流式预分词（pre-tokenization / pre-merge）：
    #   分块读取 + 在特殊 Token 处硬分割 + 正则预分词 + 频次汇总（内存有界）
    word_freqs = pre_merge_file(input_path, special_tokens)

    # 3. 执行 BPE 合并循环
    merges: list[tuple[bytes, bytes]] = []
    
    # 初始化统计所有相邻对的频次，并建立倒排索引 pair -> 含它的词型集合
    pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for word_tuple, count in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pair_freqs[pair] += count
            pair_to_words[pair].add(word_tuple)

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

    # 用**最大堆**维护「下一个要合并的 pair」，把每轮的选择从 O(pair 数) 降到摊还 O(log heap)。
    # 堆元素是 _MaxPair(freq, pair)，__lt__ 已按「频次高优先、并列 pair 字典序大优先」反向定义；
    # 每次 apply_merge 后只把频次变化的 pair（touched）重新入堆，旧记录成为 stale，取出时校验丢弃。
    heap = [_MaxPair(freq, pair) for pair, freq in pair_freqs.items()]
    heapq.heapify(heap)

    for _ in range(num_merges):
        # lazy invalidation：弹掉堆顶那些频次已与当前 pair_freqs 不一致（或已消失）的过期记录
        while heap:
            node = heap[0]
            if node.freq > 0 and pair_freqs.get(node.pair, 0) == node.freq:
                break
            heapq.heappop(heap)
        if not heap:
            break

        # 此时堆顶就是「频次最高；并列时 pair 字典序最大」的合法 pair（tie-break 与原实现一致）
        best_pair = heap[0].pair
        merges.append(best_pair)

        # 应用这次 merge，得到下一轮迭代需要的状态
        word_freqs, pair_freqs, pair_to_words, touched = apply_merge(
            word_freqs, pair_freqs, pair_to_words, best_pair
        )

        # 把频次变化的 pair 重新入堆；堆过大时重建，防止 stale 记录无限累积（空间换时间）
        _push_updates(heap, pair_freqs, touched)
        if len(heap) > 8 * (len(pair_freqs) + 1):
            heap = [_MaxPair(freq, pair) for pair, freq in pair_freqs.items()]
            heapq.heapify(heap)

    # 4. 合并结束后的词表组装（post-merge）
    return build_vocab(merges, special_tokens), merges
