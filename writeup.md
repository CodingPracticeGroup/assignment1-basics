# CS336 Assignment 1 — Writeup

> 本文件对应 handout 的 38 个 Problem；题面保留 handout 原文，Answer 用中文作答。

## 1. Problem (unicode1): Understanding Unicode (1 point)

**Deliverables:**

1. **(a)** What Unicode character does `chr(0)` return?  
   _Deliverable:_ A one-sentence response.
2. **(b)** How does this character's string representation (`__repr__()`) differ from its printed representation?  
   _Deliverable:_ A one-sentence response.
3. **(c)** What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
   ```python
   >>> chr(0)
   >>> print(chr(0))
   >>> "this is a test" + chr(0) + "string"
   >>> print("this is a test" + chr(0) + "string")
   ```
   _Deliverable:_ A one-sentence response.

**Answer:**

**(a)** `chr(0)` 返回 Unicode 码点 U+0000，即 NULL（NUL）控制字符；它是合法字符，但没有可打印字形。

**(b)** `repr(chr(0))` 会对其转义，因此在引号中显示为四个可见字符 `\x00`；而 `print(chr(0))` 直接输出原始 NUL 字节，本身不产生任何可见输出。

**(c)** NUL 确实被存进了字符串里：`len("this is a test" + chr(0) + "string") == 19`，`repr()` 显示为 `'this is a test\x00string'`，但打印出来与 `this is a teststring` 完全一样。由于许多 C/POSIX API 把 NUL 当作字符串结束符，这类字符串一旦进入 C 库就可能被静默截断——这正是字节级分词要基于 `bytes`、不能依赖 C 字符串语义的原因。

---

## 2. Problem (unicode2): Unicode Encodings (3 points)

**Deliverables:**

1. **(a)** What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.  
   _Deliverable:_ A one-to-two sentence response.
2. **(b)** Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

   ```python
   def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
       return "".join([bytes([b]).decode("utf-8") for b in bytestring])

   >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
   'hello'
   ```

   _Deliverable:_ An example input byte string for which `decode_utf8_bytes_to_str_wrong` produces incorrect output, with a one-sentence explanation of why the function is incorrect.

3. **(c)** Give a two-byte sequence that does not decode to any Unicode character(s).  
   _Deliverable:_ An example, with a one-sentence explanation.

**Answer:**

**(a)** UTF-8 是变长编码，且对 ASCII 与 ASCII 完全同字节，因此以 ASCII 为主的训练语料只需 1 字节/字符；而 UTF-16 / UTF-32 对同样的 ASCII 字符要花 2 / 4 字节（UTF-32 最浪费）。UTF-8 还没有字节序问题、可自同步，并能表示全部 Unicode 码点，所以基于 UTF-8 字节的分词器不会 OOV，也与网页文本实际的存储和传输方式一致。

**(b)** 该函数逐字节解码，但 UTF-8 多字节字符必须作为完整序列解码，单独的引导字节或后续字节并不是合法的单字节 UTF-8 码点。例如 `b'\xc3\xa9'`（“é”的 UTF-8 编码）会在第一个字节就抛 `UnicodeDecodeError`，而不是返回 “é”；因此它无法解码任何非 ASCII 字符，不是正确的 UTF-8 解码器。

**(c)** `b'\xff\xfe'`：0xFF 从不出现于合法 UTF-8，0xFE 只可能是后续字节，因此 `b'\xff\xfe'.decode("utf-8")` 抛 `UnicodeDecodeError`，解不出任何 Unicode 字符。

---

## 3. Problem (train_bpe): BPE Tokenizer Training (15 points)

**Deliverables:**
Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. Your BPE training function should handle (at least) the following input parameters:

**Input:**

- `input_path`: `str` Path to a text file with BPE tokenizer training data.
- `vocab_size`: `int` A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
- `special_tokens`: `list[str]` A list of strings to add to the vocabulary. During training, treat them as hard boundaries that prevent merges across their spans, but do not include them when computing merge statistics.

Your BPE training function should return the resulting vocabulary and merges:

**Output:**

- `vocab`: `dict[int, bytes]` The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
- `merges`: `list[tuple[bytes, bytes]]` A list of BPE merges produced from training. Each list item is a tuple of bytes `(<token1>, <token2>)`, representing that `<token1>` was merged with `<token2>`. The merges should be ordered by order of creation.

To test your BPE training function against our provided tests, you will first need to implement the test adapter at `adapters.run_train_bpe`. Then, run `uv run pytest tests/test_train_bpe.py`. Your implementation should be able to pass all tests. Optionally (this could be a large time-investment), you can implement the key parts of your training method using some systems language, for instance C++ (consider `cppyy` or `nanobind`) or Rust (using `PyO3`). If you do this, be aware of which operations require copying vs reading directly from Python memory, and make sure to leave build instructions, or make sure it builds using only `pyproject.toml`. Also note that the GPT-2 regex is not well-supported in most regex engines and will be too slow in most that do. We have verified that Oniguruma is reasonably fast and supports negative lookahead, but the `regex` package in Python is, if anything, even faster.

**Answer:**

实现位于 `cs336_basics/bpe_tokenizer/trainer.py::run_train_bpe`，通过 `tests/adapters.py::run_train_bpe` 接入测试。

**算法。**（1）先用特殊 token 切分语料，special token 作为硬边界，因此合并不会跨文档，特殊 token 也不计入合并统计。（2）对每段用 GPT-2 正则预分词（使用高性能的 `regex` 包，见 `trainer.py`）；每个匹配编码为 UTF-8 并存成「单字节元组」，累计词型频次。大语料用 `multiprocessing.Pool` 并行处理后再合并计数。（3）词表初始为 256 个单字节 token（ID 0..255），其后是特殊 token。（4）执行 `vocab_size - 256 - len(special_tokens)` 次合并：每步选频次最高的相邻对，平局时取字典序**最大**的对，写入 `merges` 并生成合并 token `b1 + b2`。相邻对频次采用增量维护——只重写包含被合并对的词型，并调整其旧/新相邻对计数——避免每步重扫全部词型。

**输出。** `vocab: dict[int, bytes]`（基础字节、特殊 token、再按创建顺序的合并产物）与 `merges: list[tuple[bytes, bytes]]`（按创建顺序），与接口要求一致。对测试正确性最关键的细节是：GPT-2 正则、字节级（而非字符级）符号、特殊 token 的硬边界，以及字典序 tie-break。

---

## 4. Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

**(a)** Train a byte-level BPE tokenizer on the _TinyStories_ dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?  
_Resource requirements: $\le$ 30 minutes (no GPUs), $\le$ 30 GB RAM_

> **Hint:** You should be able to get under 2 minutes for BPE training using multiprocessing during pre-tokenization and the following two facts:
>
> - **(a)** The `<|endoftext|>` token delimits documents in the data files.
> - **(b)** The `<|endoftext|>` token is handled as a special case before the BPE merges are applied.
>
> _Deliverable:_ A one-to-two sentence response.

**(b)** Profile your code. What part of the tokenizer training process takes the most time?  
_Deliverable:_ A one-to-two sentence response.

**Answer:**

**实测**（`run_train_bpe("data/TinyStoriesV2-GPT4-train.txt", vocab_size=10000, special_tokens=["<|endoftext|>"])`，本机 28 核 CPU）：

- **时间**：约 **63 s（~1 分钟）**，远低于 handout 的 30 分钟预算。预分词用 8 进程 + pre-token 缓存（见 (b)），合并循环用倒排索引 + 最大堆。
- **内存**：峰值 RSS 约 **7.9 GiB**（`psutil` 对主进程 + 子进程 RSS 求和，含 8 个 worker），远低于 30 GiB。早期“整篇读入”的版本峰值 ~37.7 GiB；改成**流式分块预分词**（按 `<|endoftext|>` 对齐，尾巴实测 ≤ ~5.5 KiB）后降到 GiB 量级。
- **最长 token**：**15 字符**，例如 `b' accomplishment'`、`b' disappointment'`、`b' responsibility'`（都带前导空格）。
- **序列化**：`artifacts/tinystories_10k_stream/{vocab,merges}.pkl`（vocab=10000，merges=9743）；训练脚本同时导出 GPT-2 文本格式 `vocab.json` / `merges.txt`，可用 `Tokenizer.from_files` 重新加载。
- **是否合理**：合理。TinyStories 是简单、高度重复的英文儿童故事，10K 词表会把高频词整体合并，因此最长 token 落在 13–15 字符的常见长词（accomplishment / disappointment / responsibility …）上。

**(b)** 在 TinyStories 上，**预分词仍是主要瓶颈**，但已被优化到很低。`pretokenize_text` 对高度重复的 pre-token 做了 `str -> 字节元组` 缓存（30 MB 实测 **8.28 → 18.64 MB/s，2.25x**，结果逐字节一致）；预分词多进程改用 `imap_unordered` 边完成边累加（比 `pool.map` 更快、峰值更低）。合并循环已用**倒排索引 + 最大堆**：`cProfile`（2000 merges）显示 `apply_merge` 占 ~**75%**，其中主要是 `pair_to_words` 的集合 `add/discard`；堆操作仅 ~16%，已无大的算法空间。`encode` 侧对 `pre-token -> ids` 做了缓存（**1.69 → 14.32 MB/s，8.5x**）。

---

## 5. Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

**(a)** Train a byte-level BPE tokenizer on the _OpenWebText_ dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?  
_Resource requirements: $\le$ 12 hours (no GPUs), $\le$ 100 GB RAM_  
_Deliverable:_ A one-to-two sentence response.

**(b)** Compare and contrast the tokenizer that you get training on _TinyStories_ versus _OpenWebText_.  
_Deliverable:_ A one-to-two sentence response.

**Answer:**

**(a)** 用 `run_train_bpe("data/owt_train.txt", vocab_size=32000, special_tokens=["<|endoftext|>"])` 训练并序列化结果（`artifacts/owt_32k_stream/{vocab,merges}.pkl` + GPT-2 文本格式），vocab=32000 / merges=31743。

- **时间/内存**：约 **1201 s（~20 分钟）**，峰值 RSS **12.7 GiB**——远低于 12 小时 / 100 GB 预算。其中预分词约 8 分钟（8 进程）、合并约 12 分钟（单线程，倒排索引 + 最大堆）。
- **最长 token**：**64 字节**，有两个：一个是重复 16 次的 **mojibake** 序列 `b'\xc3\x83\xc2...'`（网页文本里常见的双重 UTF-8 编码残留），另一个是 64 个连字符 `b'-----…'`。其后是 48 字节的 em-dash 串、6 个 32 字节的重复符号（`-`/`_`/`=`/`.`/`*`/mojibake）。
- **是否合理**：合理。OpenWebText 是网页文本，充斥分隔线、markdown/HTML 残留、长串重复标点和编码伪影；同一符号长程重复时 BPE 会一路合并成很长的 token。注意这类长 token 极少：词表里长度 ≥16 字节的只有 **47 个**（≥24 的仅 10 个），绝大多数仍是正常子词。

**(b)** 对比（同一批 10 篇文档实测，数字见第 7(a) 题）：

- **词表/结构**：OWT-32K 是 **3.2 倍**大的词表（32000 vs 10000），含网页特有 token——长串重复标点、mojibake、URL/HTML 片段；TinyStories-10K 偏向简单叙事英语，最长 token 是 15 字符的常见长词（`accomplishment` / `responsibility` …）。
- **本领域压缩**：OWT-32K 在 OWT 上 **4.69 bytes/token**，TS-10K 在 TinyStories 上 **4.11**——更大的 OWT 词表即便面对更混杂的网页文本也更高效。
- **跨领域**：TS-10K 用在 OWT 上降到 **3.19**（token blowup，见 7(b)）；反过来 OWT-32K 用在 TinyStories 上是 **4.01**，几乎不退化，说明 32K 网页词表对简单英文也有较好覆盖。
- **代价**：词表大 3.2 倍，嵌入层 / 输出层参数与显存相应增加。

---

## 6. Problem (tokenizer): Implementing the tokenizer (15 points)

**Deliverables:**
Implement a `Tokenizer` class that, given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text. Your tokenizer should also support user-provided special tokens (appending them to the vocabulary if they aren’t already there). We recommend the following interface:

```python
def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None)
```

_Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept the following parameters:_

- `vocab`: `dict[int, bytes]`
- `merges`: `list[tuple[bytes, bytes]]`
- `special_tokens`: `list[str] | None = None`

```python
@classmethod
def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None)
```

_Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens._

- `vocab_filepath`: `str`
- `merges_filepath`: `str`
- `special_tokens`: `list[str] | None = None`

```python
def encode(self, text: str) -> list[int]
```

_Encode an input text into a sequence of token IDs._

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```

_Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory._

```python
def decode(self, ids: list[int]) -> str
```

_Decode a sequence of token IDs into text._

To test your Tokenizer against our provided tests, you will first need to implement the test adapter at `adapters.get_tokenizer`. Then, run `uv run pytest tests/test_tokenizer.py`. Your implementation should be able to pass all tests.

**Answer:**

实现位于 `cs336_basics/bpe_tokenizer/base.py::Tokenizer`，通过 `tests/adapters.py::get_tokenizer` 接入测试。

- `__init__(vocab, merges, special_tokens)`：保存词表，构建 `bytes -> id` 哈希表，并把有序的 `merges` 转成「对 -> 优先级」字典 `{(a, b): rank}`（rank 越小优先级越高）。若给出特殊 token，则编译匹配它们的正则，并按长度降序排序，保证重叠/连续的特殊 token 被正确切分。
- `encode(text)`：先用特殊 token 正则切分；特殊片段直接查表得到 id（不参与 BPE），其余片段再用 GPT-2 正则切分。每个预分词表示为单字节列表，反复合并「当前出现的、rank 最小（优先级最高）」的相邻对，直到无法再合并，最后映射为 id。
- `encode_iterable(iterable)`：惰性生成器，每次只对一段字符串调用 `encode`，因此数 GB 文件可在恒定（远小于 1MB）内存下分词。
- `decode(ids)`：拼接每个 id 的 `vocab[id]` 后做 `.decode("utf-8", errors="replace")`，非法字节序列退化为 U+FFFD 而不会抛异常。
- `from_files(vocab_filepath, merges_filepath, special_tokens)`：类方法，加载训练代码序列化出的词表与 merges 并构造 `Tokenizer`（官方测试只通过 `get_tokenizer` 走 `__init__` 这条路径）。

---

## 7. Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

**(a)** Sample 10 documents from _TinyStories_ and _OpenWebText_. Using your previously-trained _TinyStories_ and _OpenWebText_ tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?  
_Deliverable:_ A one-to-two sentence response.

**(b)** What happens if you tokenize your _OpenWebText_ sample with the _TinyStories_ tokenizer? Compare the compression ratio and/or qualitatively describe what happens.  
_Deliverable:_ A one-to-two sentence response.

**(c)** Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?  
_Deliverable:_ A one-to-two sentence response.

**(d)** Using your _TinyStories_ and _OpenWebText_ tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype `uint16`. Why is `uint16` an appropriate choice?  
_Deliverable:_ A one-to-two sentence response.

**Answer:**

**(a)** *压缩比（bytes/token）。* 各采样 10 篇文档、用对应 tokenizer 编码，`len(text.encode("utf-8")) / len(tokenizer.encode(text))`：TinyStories 样本用 TS-10K 得 **≈ 4.11**；OpenWebText 样本用 OWT-32K 得 **≈ 4.69**。交叉对比：TS-10K 在 OWT 上 3.19、OWT-32K 在 TinyStories 上 4.01。可见各自领域内，“更大词表 + 更贴合分布”都带来更高压缩比。

**(b)** *用 TinyStories tokenizer 编码 OWT。* 同一批 10 篇 OWT 文档，用 **TS-10K** 编码得 **≈ 3.19 bytes/token**，比它在 TinyStories 上的 4.11 明显下降（约 −22%）。原因：TS-10K 没见过 OWT 的大部分词汇，只能退回单字节与短合并，于是压缩比下降（每字节需要更多 token），序列显著变长——即 token blowup；若用它做预训练，会通过注意力的二次复杂度放大开销。

**(c)** *吞吐。* 对 100–200 MB 文本计时单线程 `Tokenizer.encode`：TS-10K 在 TinyStories 上约 **14.1 MB/s**（200 MB 实测；对 pre-token 做缓存后，未缓存时约 1.7 MB/s）；TS-10K 与 OWT-32K 在 OWT 文本上均约 **10.7 MB/s**（OWT 的 pre-token 重复更少，缓存命中率较低）。注意首次出现的 pre-token 仍需完整合并，缓存只对重复词生效；语料越长、重复越多，吞吐越接近上限。按 10.7–14.1 MB/s 估计，**825 GB 的 Pile 约需 ~16–21 小时**（单线程；未缓存时约 142 小时）。

**(d)** *为什么用 uint16？* 这里词表最大 32,000（即使 GPT-2 的 50,257 也放得下），任何 token id 都小于 65536，可用无符号 16 位精确存储。相比 `int32`/`int64`，它把分词后语料的内存与 I/O 减半，同时仍覆盖整个词表。

---

## 8. Problem (linear): Implementing the linear module (1 point)

**Deliverables:**
Implement a `Linear` class that inherits from `torch.nn.Module` and performs a linear transformation. Your implementation should follow the interface of PyTorch’s built-in `nn.Linear` module, except for not having a bias argument or parameter. We recommend the following interface:

```python
def __init__(self, in_features: int, out_features: int, device=None, dtype=None)
```

_Construct a linear transformation module. This function should accept the following parameters:_

- `in_features`: `int` final dimension of the input
- `out_features`: `int` final dimension of the output
- `device`: `torch.device | None = None` Device to store the parameters on
- `dtype`: `torch.dtype | None = None` Data type of the parameters

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

_Apply the linear transformation to the input._

**Make sure to:**

- subclass `nn.Module`
- call the superclass constructor
- construct and store your parameter as $W$ (not $W^T$), putting it in an `nn.Parameter`
- of course, don’t use `nn.Linear` or `nn.functional.linear`

For initializations, use the settings from above along with `torch.nn.init.trunc_normal_` to initialize the weights.

To test your Linear module, implement the test adapter at `adapters.run_linear`. The adapter should load the given weights into your Linear module. You can use `Module.load_state_dict` for this purpose. Then, run `uv run pytest -k test_linear`.

**Answer:**

实现为 `cs336_basics/notation/{no_einstein,einstein}/layers.py` 中的 `Linear`（由 `cs336_basics/model/layers.py` 转发，按 `CS336_NOTATION` 环境变量选择）；通过 `adapters.run_linear` 接入测试。

- `__init__(in_features, out_features, device, dtype)`：`weight = nn.Parameter(torch.empty(out_features, in_features))`，用 `torch.nn.init.trunc_normal_(weight, 0.0, std, -3*std, 3*std)` 初始化，其中 `std = sqrt(2 / (in_features + out_features))`。无 bias，`W` 按 `(d_out, d_in)` 存储（不做转置）。
- `forward(x)`：计算 `y = x W^T`。普通版用 `x @ self.weight.t()`；Einstein 版用 `einx.dot("... d_in, d_out d_in -> ... d_out", x, self.weight)`，无需手动转置，且自动支持任意多前置 batch 维。

---

## 9. Problem (embedding): Implement the embedding module (1 point)

**Deliverables:**
Implement the `Embedding` class that inherits from `torch.nn.Module` and performs an embedding lookup. Your implementation should follow the interface of PyTorch’s built-in `nn.Embedding` module. We recommend the following interface:

```python
def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None)
```

_Construct an embedding module. This function should accept the following parameters:_

- `num_embeddings`: `int` Size of the vocabulary
- `embedding_dim`: `int` Dimension of the embedding vectors, i.e., $d_{model}$
- `device`: `torch.device | None = None` Device to store the parameters on
- `dtype`: `torch.dtype | None = None` Data type of the parameters

```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

_Lookup the embedding vectors for the given token IDs._

**Make sure to:**

- subclass `nn.Module`
- call the superclass constructor
- initialize your embedding matrix as an `nn.Parameter`
- store the embedding matrix with the `d_model` being the final dimension
- of course, don’t use `nn.Embedding` or `nn.functional.embedding`

Again, use the settings from above for initialization, and use `torch.nn.init.trunc_normal_` to initialize the weights.

To test your implementation, implement the test adapter at `adapters.run_embedding`. Then, run `uv run pytest -k test_embedding`.

**Answer:**

实现为 `layers.py` 中的 `Embedding`（两版都有），通过 `adapters.run_embedding` 接入测试。

- `__init__(num_embeddings, embedding_dim, device, dtype)`：`weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))`，用 `trunc_normal_(mean=0, std=1, a=-3, b=3)` 初始化；`d_model` 是最后一维。
- `forward(token_ids)`：普通版直接用 `self.weight[token_ids]` 查表；Einstein 版用 `einx.get_at("[v] d, ... -> ... d", self.weight, token_ids)`。两者都支持任意前置 batch 维，返回 `(..., d_model)`。

---

## 10. Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)

**Deliverables:**
Implement `RMSNorm` as a `torch.nn.Module`. We recommend the following interface:

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

_Construct the RMSNorm module. This function should accept the following parameters:_

- `d_model`: `int` Hidden dimension of the model
- `eps`: `float = 1e-5` Epsilon value for numerical stability
- `device`: `torch.device | None = None` Device to store the parameters on
- `dtype`: `torch.dtype | None = None` Data type of the parameters

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

_Process an input tensor of shape `(batch_size, sequence_length, d_model)` and return a tensor of the same shape._

**Note:** Remember to upcast your input to `torch.float32` before performing the normalization (and later downcast to the original dtype), as described above.

To test your implementation, implement the test adapter at `adapters.run_rmsnorm`. Then, run `uv run pytest -k test_rmsnorm`.

**Answer:**

实现为 `layers.py` 中的 `RMSNorm`，通过 `adapters.run_rmsnorm` 接入测试。

- `__init__(d_model, eps=1e-5)`：可学习增益 `weight = nn.Parameter(torch.ones(d_model))`。
- `forward(x)`：先上投到 `float32`；沿最后一维求平方均值并 keepdim（普通版 `torch.mean(x**2, dim=-1, keepdim=True)`；Einstein 版 `einx.mean("... ([d])", x**2)`）；`rms = sqrt(mean_square + eps)`；归一化 `x / rms` 后降回原 dtype；再乘增益（普通版 `*`，Einstein 版 `einx.multiply("... d, d -> ... d", normalized, weight)`）。输入输出同形状。

---

## 11. Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)

**Deliverables:**
Implement the SwiGLU feed-forward network, composed of a SiLU activation function and a GLU.

**Note:** In this particular case, you should feel free to use `torch.sigmoid` in your implementation for numerical stability.

You should set $d_{ff}$ to approximately $\frac{8}{3} \times d_{model}$ in your implementation, while ensuring that the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware. To test your implementation against our provided tests, you will need to implement the test adapter at `adapters.run_swiglu`. Then, run `uv run pytest -k test_swiglu` to test your implementation.

**Answer:**

实现为 `layers.py` 中的 `SwiGLU`（以及 `silu`），通过 `adapters.run_swiglu` 接入测试。

- 三个无 bias 的 `Linear`：`w1, w3: d_model -> d_ff`，`w2: d_ff -> d_model`。
- `forward(x) = w2( silu(w1(x)) * w3(x) )`，其中 `silu(z) = z * sigmoid(z)`（门控分支过激活后与值分支逐元素相乘，再投影回去）。逐元素乘在普通版是 `*`，Einstein 版是 `einx.multiply("... f, ... f -> ... f", gate, w3x)`。
- `d_ff` 取离 `(8/3) * d_model` 最近的 64 的倍数，以充分利用硬件。

---

## 12. Problem (rope): Implement RoPE (2 points)

**Deliverables:**
Implement a class `RotaryPositionalEmbedding` that applies RoPE to the input tensor.

The following interface is recommended:

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

_Construct the RoPE module and create buffers if needed._

- `theta`: `float` $\Theta$ value for the RoPE
- `d_k`: `int` dimension of query and key vectors
- `max_seq_len`: `int` Maximum sequence length that will be input
- `device`: `torch.device | None = None` Device to store the buffer on

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

_Process an input tensor of shape `(..., seq_len, d_k)` and return a tensor of the same shape. Note that you should tolerate $x$ with an arbitrary number of batch dimensions. You should assume that the token positions are a tensor of shape `(..., seq_len)` specifying the token positions of $x$ along the sequence dimension._

You should use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension.

To test your implementation, complete `adapters.run_rope` and make sure it passes `uv run pytest -k test_rope`.

**Answer:**

实现为 `cs336_basics/notation/{no_einstein,einstein}/attention.py` 中的 `RotaryPositionalEmbedding`，通过 `adapters.run_rope` 接入测试。

- `__init__(theta, d_k, max_seq_len, device)` 预计算并注册 buffer `freqs[k] = theta ** (-2k / d_k)`（`k = 0 .. d_k/2 - 1`）。
- `forward(x, token_positions)`：`angles = token_positions.unsqueeze(-1) * freqs`，再取 `cos`/`sin`。相邻特征对组成 2D 坐标对做旋转：`out_even = x_even*cos - x_odd*sin`，`out_odd = x_even*sin + x_odd*cos`。`x` 形状 `(..., seq_len, d_k)`、位置 `(..., seq_len)`，任意前置 batch 维通过广播处理。普通版用 `x[..., 0::2]`/`x[..., 1::2]` 切片并写入 `torch.empty_like(x)`；Einstein 版用 `einx.id("... s (d pair) -> ... s d pair", x, pair=2)` 拆对，构造 2x2 旋转矩阵后用 `einx.dot("... p i j, ... p j -> ... p i")` 缩合；由于 `einx.dot` 不跨秩广播，额外用了一个小助手 `_broadcast_leading` 对齐秩。
- 因为 cos/sin 是按 `token_positions` 索引的，位置可以是任意值（不必是 `0..s-1`）。

---

## 13. Problem (softmax): Implement softmax (1 point)

**Deliverables:**
Write a function to apply the softmax operation on a tensor. Your function should take two parameters: a tensor and a dimension $i$, and apply softmax to the $i$-th dimension of the input tensor. The output tensor should have the same shape as the input tensor, but its $i$-th dimension will now have a normalized probability distribution. Use the trick of subtracting the maximum value in the $i$-th dimension from all elements of the $i$-th dimension to avoid numerical stability issues.

To test your implementation, complete `adapters.run_softmax` and make sure it passes `uv run pytest -k test_softmax_matches_pytorch`.

**Answer:**

实现为 `attention.py` 中的 `softmax(x, dim=-1)`，通过 `adapters.run_softmax` 接入测试。

数值稳定写法：沿 `dim` 减去最大值（`keepdim=True`），取指数，再除以沿 `dim` 的和（`keepdim=True`），因此即使输入整体 +100 也不会溢出。Einstein 版用 `einx.softmax` 并把被归约轴用中括号标出（如 `einx.softmax("a0 [a1] a2", x)`）。两者在容差内匹配 `torch.nn.functional.softmax`。

---

## 14. Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)

**Deliverables:**
Implement the scaled dot-product attention function. Your implementation should handle keys and queries of shape `(batch_size, ..., seq_len, d_k)` and values of shape `(batch_size, ..., seq_len, d_v)`, where `...` represents any number of other batch-like dimensions (if provided). The implementation should return an output with the shape `(batch_size, ..., seq_len, d_v)`. See Section 3.2 for a discussion on batch-like dimensions.

Your implementation should also support an optional user-provided boolean mask of shape `(seq_len, seq_len)`. The attention probabilities of positions with a mask value of `True` should collectively sum to 1, and the attention probabilities of positions with a mask value of `False` should be zero.

To test your implementation against our provided tests, you will need to implement the test adapter at `adapters.run_scaled_dot_product_attention`. `uv run pytest -k test_scaled_dot_product_attention` tests your implementation on third-order input tensors, while `uv run pytest -k test_4d_scaled_dot_product_attention` tests your implementation on fourth-order input tensors.

**Answer:**

实现为 `attention.py` 中的 `scaled_dot_product_attention(Q, K, V, mask=None)`，通过 `adapters.run_scaled_dot_product_attention` 接入测试。

- `d_k = Q.size(-1)`；`scores = Q K^T / sqrt(d_k)`。
- 若给出布尔 mask，则在 softmax 前做 `scores.masked_fill(~mask, -inf)`，于是被 mask 的位置概率为 0，未 mask 的位置概率和为 1。
- `probs = softmax(scores, dim=-1)`；输出 `= probs V`。
- Q/K/V 可以带任意多前置 batch 维（`...` 轴），mask 在其上广播。普通版用 `torch.matmul` 加显式转置；Einstein 版用 `einx.dot("... q d, ... k d -> ... q k")` 与 `einx.dot("... q k, ... k v -> ... q v")`。输出形状为 `Q.shape[:-1] + (d_v,)`。

---

## 15. Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)

**Deliverables:**
Implement causal multi-head self-attention as a `torch.nn.Module`. Your implementation should accept (at least) the following parameters:

- `d_model`: `int` Dimensionality of the Transformer block inputs.
- `num_heads`: `int` Number of heads to use in multi-head self-attention.

Following A. Vaswani et al. [8], set $d_k = d_v = \frac{d_{model}}{h}$. To test your implementation against our provided tests, implement the test adapter at `adapters.run_multihead_self_attention`. Then, run `uv run pytest -k test_multihead_self_attention` to test your implementation.

**Answer:**

实现为 `attention.py` 中的 `CausalMultiHeadSelfAttention`，通过 `adapters.run_multihead_self_attention` 接入测试。

- 四个无 bias 的 `Linear` 投影：`q_proj, k_proj, v_proj, output_proj`，均为 `d_model -> d_model`。
- `d_k = d_v = d_model // num_heads`。`forward` 把 `x`（`(b, s, d_model)`）投影后拆成 `(b, num_heads, s, d_k)`（普通版 `reshape + transpose`；Einstein 版 `einx.id("b s (h d) -> b h s d", q, h=num_heads)`），按需对 q、k 施加 RoPE，构造下三角因果 mask `torch.tril(torch.ones(s, s, dtype=bool))`，调用缩放点积注意力，再把多头合回 `(b, s, d_model)` 并过 `output_proj`。结果在容差内匹配朴素非批量化参考实现。

---

## 16. Problem (transformer_block): Implement the Transformer block (3 points)

**Deliverables:**
Implement the pre-norm Transformer block as described in Section 3.4 and illustrated in Figure 2. Your Transformer block should accept (at least) the following parameters:

- `d_model`: `int` Dimensionality of the Transformer block inputs.
- `num_heads`: `int` Number of heads to use in multi-head self-attention.
- `d_ff`: `int` Dimensionality of the position-wise feed-forward inner layer.

To test your implementation, implement the adapter `adapters.run_transformer_block`. Then run `uv run pytest -k test_transformer_block` to test your implementation.

**Answer:**

实现为 `cs336_basics/notation/{no_einstein,einstein}/transformer.py` 中的 `TransformerBlock`，通过 `adapters.run_transformer_block` 接入测试。

Pre-Norm 残差结构：`z = x + attn(ln1(x))`，`y = z + ffn(ln2(z))`；其中 `ln1/ln2` 是 `RMSNorm`，`attn` 是因果多头自注意力（传入 RoPE 时启用），`ffn` 是 SwiGLU。输入输出形状 `(b, s, d_model)`。

---

## 17. Problem (transformer_lm): Implementing the Transformer LM (3 points)

**Deliverables:**
Time to put it all together! Implement the Transformer language model as described in Section 3.1 and illustrated in Figure 1. At minimum, your implementation should accept all the aforementioned construction parameters for the Transformer block, as well as these additional parameters:

- `vocab_size`: `int` The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
- `context_length`: `int` The maximum context length, necessary for determining the dimensionality of the RoPE sin and cos buffer.
- `num_layers`: `int` The number of Transformer blocks to use.

To test your implementation against our provided tests, you will first need to implement the test adapter at `adapters.run_transformer_lm`. Then, run `uv run pytest -k test_transformer_lm` to test your implementation.

**Answer:**

实现为 `transformer.py` 中的 `BasicsTransformerLM`，通过 `adapters.run_transformer_lm` 接入测试。

组件依次为：token `Embedding`（vocab_size, d_model）；共享的 `RotaryPositionalEmbedding`，其 `d_k = d_model // num_heads`、长度 `context_length`；`num_layers` 个 Pre-Norm `TransformerBlock`；最后是 `RMSNorm` 与 `lm_head` `Linear(d_model, vocab_size)`。`forward(token_ids)` 构造位置 `torch.arange(s)`，查词嵌入，逐层传入位置与共享 RoPE，再过最终 norm 和 LM head，返回 `(batch, seq_len, vocab_size)` 的 logits。

---

## 18. Problem (transformer_accounting): Transformer LM resource accounting (5 points)

**Deliverables:** > **(a)** Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:

- `vocab_size`: 50,257
- `context_length`: 1,024
- `num_layers`: 48
- `d_model`: 1,600
- `num_heads`: 25
- `d_ff`: 4,288 (the nearest multiple of 64 to $\frac{8}{3} \times 1,600$)

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?  
_Deliverable:_ A one-to-two sentence response.

**(b)** Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has `context_length` tokens.  
_Deliverable:_ A list of matrix multiplies (with descriptions), and the total number of FLOPs required.

**(c)** Based on your analysis above, which parts of the model require the most FLOPs?  
_Deliverable:_ A one-to-two sentence response.

**(d)** Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?  
_Deliverable:_ For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.

**(e)** Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?  
_Deliverable:_ A one-to-two sentence response.

**Answer:**

以下记 batch 为 `b`，序列 `s = context_length = 1024`，`V = vocab_size`，`L` 层，模型宽度 `d`，`h` 个头，内层宽度 `d_ff`。一次矩阵乘 `(m x n)(n x p)` 需要 `2mnp` FLOPs。

**参数量。** token embedding 与 LM head 不共享权重：
P = `2*V*d + L*(4*d^2 + 3*d*d_ff + 2*d) + d`
（`4d^2` 是 Q,K,V,O；`3*d*d_ff` 是 W1,W3,W2；`2d` 是块内两个 RMSNorm；最后的 `d` 是最终 RMSNorm。）

**(a)** 对 GPT-2 XL 共 **P = 1,640.5M = 1.64B 参数**（取 `d_ff = 4288`）。fp32 每参数 4 字节，仅加载权重需要 `4P = 6.56 GB`（6.11 GiB）。

**(b) 一次前向的矩阵乘及其 FLOPs（batch `b`、序列 `s`）**

| 矩阵乘 | 形状 | FLOPs（整个模型） |
| :-- | :-- | --: |
| QKV 投影 | 每层 `(b s d)(d 3d)` | `L * 6 b s d^2` = 7.55e14 |
| 输出投影 | 每层 `(b s d)(d d)` | `L * 2 b s d^2` = 2.52e14 |
| 注意力分数 QK^T | 每层 `(b h s d_k)(b h d_k s)` | `L * 2 b s^2 d` = 1.61e11 |
| 值的加权求和 | 每层 `(b h s s)(b h s d_v)` | `L * 2 b s^2 d` = 1.61e11 |
| FFN W1/W3 与 W2 | 每层 `(b s d)(d d_ff)`、`(b s d_ff)(d_ff d)` | `L * 6 b s d d_ff` = 2.02e15 |
| LM head logits | `(b s d)(d V)` | `2 b s d V` = 1.65e14 |

当 `b = 1`、`s = 1024` 时，前向合计 **约 3.52 TFLOP**（带 batch `b` 时整体乘 `b`）。

**(c)** 在 `s = 1024` 下 **前馈网络占大头**：FFN 约 57.5%，QKV 投影约 21.5%，输出投影约 7.2%，LM head 约 4.7%，两个注意力矩阵乘（QK^T 与 AV）各约 4.6%。也就是说这个上下文长度下，参数密集的 `d^2` 矩阵乘才是主要开销，而不是 `s^2` 的注意力分数。

**(d) 其它 GPT-2 规模的比例（前向，`b = 1`，`s = 1024`）**

| 模型 | QKV | out | QK^T | AV | FFN | logits | 合计 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| small (L12, d768, h12, `d_ff`=2048) | 14.9% | 5.0% | 6.6% | 6.6% | 39.8% | 27.1% | 291.6 GFLOP |
| medium (L24, d1024, h16, `d_ff`=2752) | 18.6% | 6.2% | 6.2% | 6.2% | 50.1% | 12.7% | 830.2 GFLOP |
| large (L36, d1280, h20, `d_ff`=3392) | 20.5% | 6.8% | 5.5% | 5.5% | 54.3% | 7.4% | 1768.5 GFLOP |
| XL (L48, d1600, h25, `d_ff`=4288) | 21.5% | 7.2% | 4.6% | 4.6% | 57.5% | 4.7% | 3516.8 GFLOP |

模型变大时，`d^2` 与 `d*d_ff` 项（FFN 与 QKV）占比**上升**，而 **LM head** 与 `s^2` 的**注意力**项占比**下降**——因为词表项只随 `d` 线性增长、注意力项与 `d` 无关，而块内计算约随 `L*d^2` 增长。（这里 `d_ff` 均取离 `8d/3` 最近的 64 的倍数。）

**(e)** 把 GPT-2 XL 的上下文从 1024 增到 16,384，前向开销从 3.52 TFLOP 增到 **133.6 TFLOP，约 38.0 倍**。此时 `s^2` 的注意力项成为主导：QK^T 与 AV 各从 4.6% 升到 30.9%（合计 62%），FFN 从 57.5% 降到 24.2%，LM head 从 4.7% 降到 2.0%。长上下文下，注意力 FLOPs（及其二次增长的激活显存）成为瓶颈。

---

## 19. Problem (cross_entropy): Implement cross-entropy (1 point)

**Deliverables:**
Write a function to compute the cross-entropy loss, which takes in predicted logits ($o_i$) and targets ($x_{i+1}$) and computes the cross-entropy $\ell_i = -\log \text{softmax}(o_i)[x_{i+1}]$. Your function should handle the following:

- Subtract the largest element for numerical stability.
- Cancel out `log` and `exp` whenever possible.
- Handle any additional batch dimensions and return the _average_ across the batch. As with Section 3.2, we assume batch-like dimensions always come first, before the vocabulary size dimension.

Implement `adapters.run_cross_entropy`, then run `uv run pytest -k test_cross_entropy` to test your implementation.

**Answer:**

实现为 `cs336_basics/notation/{no_einstein,einstein}/cross_entropy.py` 中的 `cross_entropy(inputs, targets)`，由 `training/optimizers.py` 转发；通过 `adapters.run_cross_entropy` 接入测试。

先把所有 batch 维压平成 `(N, vocab_size)`、targets 压成 `(N,)`；每行减最大值做稳定化，求稳定化 logits 的 `logsumexp`，并按行取出 target 对应 logit；loss 为 `-target_logit + logsumexp` 的均值。减最大值并直接使用稳定化 logits，等价于把 target 项外层的 `log`/`exp` 约掉。Einstein 版用 `einx.logsumexp("n [v]", x)` 与 `einx.get_at("n [v], n -> n", x, t)`；普通版用手写 sum/exp/log 与高级索引。与 `F.cross_entropy` 在 1e-4 内一致，1000 倍放大输入也成立。

---

## 20. Problem (learning_rate_tuning): Tuning the learning rate (1 point)

**Deliverables:**
As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s see that in practice in our toy example. Run the SGD example above with three other values for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of training)?  
_Deliverable:_ A one-to-two sentence response with the behaviors you observed.

**Answer:**

我实跑了 handout 的 toy 循环（`weights = nn.Parameter(5 * torch.randn(10, 10))`，loss `= (weights**2).mean()`，10 次迭代，使用式 (20) 的衰减 SGD）。实测首个 -> 末个 loss：

- `lr = 1e1`：比基线 `lr = 1` 下降快得多——`26.27 -> 3.53`（基线 `26.27 -> 21.74`）。
- `lr = 1e2`：三者中最快——`26.27 -> ~0`（第 5 步就到 `1.1e-16`），几乎立刻收敛。
- `lr = 1e3`：**发散**——第 1 步后 loss 为 `9.48e3`，并单调增长到第 10 步的 `2.44e18`（没有 NaN，但明显越过稳定阈值）。

因此在这个 10 步 toy 上，学习率从 1e1 到 1e2 会加快下降，而 1e3 过大导致 loss 爆炸。

---

## 21. Problem (adamw): Implement AdamW (2 points)

**Deliverables:**
Implement the AdamW optimizer as a subclass of `torch.optim.Optimizer`. Your class should take the learning rate $\alpha$ in `__init__`, as well as the $\beta, \epsilon$ and $\lambda$ hyperparameters. To help you keep state, the base `Optimizer` class gives you a dictionary `self.state`, which maps `nn.Parameter` objects to a dictionary that stores any information you need for that parameter (for AdamW, this would be the moment estimates). Implement `adapters.get_adamw_cls` and make sure it passes `uv run pytest -k test_adamw`.

**Answer:**

实现为 `cs336_basics/training/optimizers.py` 中的 `AdamW(torch.optim.Optimizer)`，通过 `adapters.get_adamw_cls` 接入测试。

- `__init__(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)`：校验超参，并作为 defaults 传给基类 `Optimizer`。
- `step` 为每个参数维护 `(step, exp_avg, exp_avg_sq)`。对每个有梯度的参数：(1) 解耦权重衰减 `p -= lr * weight_decay * p`；(2) 更新一阶矩 `m = b1*m + (1-b1)*g` 与二阶矩 `v = b2*v + (1-b2)*g^2`；(3) 偏差校正更新 `p -= lr * sqrt(1-b2^t)/(1-b1^t) * m / (sqrt(v) + eps)`。
- 在测试容差内匹配 PyTorch 的 `torch.optim.AdamW`（测试接受参考快照或 PyTorch 权重任一）。

---

## 22. Problem (adamw_accounting): Resource accounting for training with AdamW (2 points)

**Deliverables:**
Let us compute how much memory and compute running AdamW requires. Assume we are using `float32` for every tensor.

**(a)** How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the `batch_size` and the model hyperparameters (`vocab_size`, `context_length`, `num_layers`, `d_model`, `num_heads`). Assume $d_{ff} = \frac{8}{3} \times d_{model}$.

For simplicity, when calculating memory usage of activations, consider only the following components:

- Transformer block
  - `RMSNorm`(s)
  - Multi-head self-attention sublayer: $QKV$ projections, $QK^T$ matrix multiply, softmax, weighted sum of values, output projection.
  - Position-wise feed-forward (SwiGLU): $W_1$, $W_2$, SiLU on the gate branch, element-wise product, $W_3$
- final RMSNorm
- output embedding
- cross-entropy on logits

_Deliverable:_ An algebraic expression for each of parameters, activations, gradients, and optimizer state, as well as the total.

**(b)** Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the `batch_size`. What is the maximum batch size you can use and still fit within 80GB memory?  
_Deliverable:_ An expression that looks like $a \cdot \text{batch\_size} + b$ for numerical values $a, b$, and a number representing the maximum batch size.

**(c)** How many FLOPs does running one step of AdamW take?  
_Deliverable:_ An algebraic expression, with a brief justification.

**(d)** Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak FLOP throughput [A. Chowdhery et al., 2022]. An NVIDIA H100 GPU has a theoretical peak of 495 teraFLOP/s for “float32” (actually TensorFloat-32, which in reality is “bfloat19”) operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single H100? Following J. Kaplan et al. [25] and J. Hoffmann et al. [26], assume that the backward pass has twice the FLOPs of the forward pass.  
_Deliverable:_ The number of hours training would take, with a brief justification.

**Answer:**

设全部为 fp32（4 字节），`b` = batch size，`s = context_length`，`V` = vocab_size，`L` 层，宽度 `d`，`h` 个头，`d_ff = (8/3)d`。记 `P = 2Vd + L(4d^2 + 3d d_ff + 2d) + d`。

**(a) 峰值显存。** 分为：

- **参数：** `4P` 字节。
- **梯度：** `4P` 字节（每参数一份梯度）。
- **优化器状态（AdamW）：** 一阶矩与二阶矩 `m`、`v`，即 `2 * 4P = 8P` 字节。
- **激活**（只统计题目列出的组件；按 token 计，每层含 RMSNorm 输出 `2d`、QKV `3d`、QK^T `h s`、softmax `h s`、加权求和 `d`、输出投影 `d`、W1 `d_ff`、W2 `d`、SiLU `d_ff`、逐元素积 `d_ff`、W3 `d_ff`；再加最终 RMSNorm 的 `d`、LM head 的 `V`、交叉熵的 1）：
   `A = 4 * b * s * ( L*(8d + 4 d_ff + 2 h s) + d + V + 1 )` 字节。

**总计** = `16P + A` = `16P + 4 b s (L(8d + 4d_ff + 2hs) + d + V + 1)` 字节。

**(b) GPT-2 XL 代入。** 取 `V=50257, L=48, d=1600, h=25, d_ff=4288, s=1024`：`P = 1.6405e9`，故 `16P = 2.625e10` 字节 = **24.44 GiB**；激活系数 `4*s*3,947,154 = 1.617e10` 字节 = **每个 batch 单位 15.06 GiB**。于是

  `memory(batch) = 15.06 GiB * batch + 24.44 GiB`

在 80 GiB 预算下：`batch_max = (80 - 24.44)/15.06 = 3.69`，故**最大 batch size 为 3**（若把 80GB 解释为 80e9 字节，结果同样是 3）。注意这主要由保存全部激活决定；若使用梯度检查点或更短上下文，可以支持大得多的 batch。

**(c) AdamW 每步 FLOPs。** 它是对 `P` 个参数的纯逐元素运算：两个矩更新（`m`、`v`）、解耦权重衰减、以及最终更新（`sqrt`、`div`、乘、减）。把每次逐元素运算记一次、乘加对记两次，约为 **每步 12P FLOPs**（量级 10P；相对前向/反向可忽略）。

**(d) 单张 H100 训练时间。** `b=1024, s=1024` 时一次前向 `F = 3.60e15` FLOPs。反向为 `2F`、优化器 `12P`，故每步 `3F + 12P = 1.08e16` FLOPs（优化器占比 < 0.001%）。400K 步总计 `4.32e21` FLOPs。按单卡 H100 50% MFU（`0.5 * 495e12 = 2.475e14` FLOP/s）：

  `time = 4.32e21 / 2.475e14 = 1.75e7 s = 4,850 小时 = 202 天`

也就是说，单张 H100 训练 GPT-2 XL 400K 步（batch 1024）需要大约 **4,850 小时（约 200 天）**，这类负载只有多卡集群才现实。

---

## 23. Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup (1 point)

**Deliverables:**
Write a function that takes $t, \alpha_{\text{max}}, \alpha_{\text{min}}, T_w$ and $T_c$, and returns the learning rate $\alpha_t$ according to the scheduler defined above. Then implement `adapters.get_lr_cosine_schedule` and make sure it passes `uv run pytest -k test_get_lr_cosine_schedule`.

**Answer:**

实现为 `cs336_basics/training/schedulers.py` 中的 `run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)`，通过 `adapters.run_get_lr_cosine_schedule` 接入测试。

- `it < warmup_iters`：线性 warmup，返回 `(it / warmup_iters) * max_lr`。
- `warmup_iters <= it <= cosine_cycle_iters`：`decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)`，`coeff = 0.5 * (1 + cos(pi * decay_ratio))`，返回 `min_lr + coeff * (max_lr - min_lr)`。
- `it > cosine_cycle_iters`：返回 `min_lr`（`warmup == cycle` 的退化情形也返回 `min_lr`）。

输出匹配测试中的参考学习率表。

---

## 24. Problem (gradient_clipping): Implement gradient clipping (1 point)

**Deliverables:**
Write a function that implements gradient clipping. Your function should take a list of parameters and a maximum $\ell_2$-norm. It should modify each parameter gradient in place. Use $\epsilon = 10^{-6}$ (the PyTorch default). Then, implement the adapter `adapters.run_gradient_clipping` and make sure it passes `uv run pytest -k test_gradient_clipping`.

**Answer:**

实现为 `cs336_basics/training/clipping.py` 中的 `run_gradient_clipping(parameters, max_l2_norm)`，通过 `adapters.run_gradient_clipping` 接入测试。

对所有权重计算全局 L2 范数 `total_norm = sqrt(sum_p sum(grad_p^2))`（detach，避免建图），若 `total_norm > max_l2_norm`，就把每个梯度原地乘上 `clip_coef = max_l2_norm / (total_norm + 1e-6)`。没有梯度的冻结参数跳过。结果在测试容差内匹配 `torch.nn.utils.clip_grad_norm_`。

---

## 25. Problem (data_loading): Implement data loading (2 points)

**Deliverables:**
Write a function that takes a numpy array $x$ (integer array with token IDs), a `batch_size`, a `context_length` and a PyTorch device string (e.g., `'cpu'` or `'cuda:0'`), and returns a pair of tensors: the sampled input sequences and the corresponding next-token targets. Both tensors should have shape `(batch_size, context_length)` containing token IDs, and both should be placed on the requested device. To test your implementation against our provided tests, you will first need to implement the test adapter at `adapters.run_get_batch`. Then, run `uv run pytest -k test_get_batch` to test your implementation.

**Answer:**

实现为 `cs336_basics/training/dataloader.py` 中的 `run_get_batch(dataset, batch_size, context_length, device)`，通过 `adapters.run_get_batch` 接入测试。

在 `[0, len(dataset) - context_length)` 内均匀随机采样 `batch_size` 个起点；每个起点取 `x = dataset[start : start + context_length]` 与下一 token 目标 `y = dataset[start + 1 : start + context_length + 1]`。堆叠后转成 `torch.long` 并搬到指定 device，返回 `(x, y)`，形状均为 `(batch_size, context_length)`。

---

## 26. Problem (checkpointing): Implement model checkpointing (1 point)

**Deliverables:**
Implement the following two functions to load and save checkpoints:

```python
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out)
```

_`save_checkpoint` should dump all the state from the model, optimizer and iteration into the file-like object `out`. You can use the `state_dict` method of both the model and the optimizer to get their relevant states and use `torch.save(obj, out)` to dump `obj` into `out` (PyTorch supports either a path or a file-like object here). A typical choice is to have `obj` be a dictionary, but you can use whatever format you want as long as you can load your checkpoint later._
This function expects the following parameters:

- `model`: `torch.nn.Module`
- `optimizer`: `torch.optim.Optimizer`
- `iteration`: `int`
- `out`: `str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

```python
def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int
```

_`load_checkpoint` should load a checkpoint from `src` (path or file-like object), and then recover the model and optimizer states from that checkpoint. Your function should return the iteration number that was saved to the checkpoint. You can use `torch.load(src)` to recover what you saved in your `save_checkpoint` implementation, and the `load_state_dict` method in both the model and optimizer to return them to their previous states._
This function expects the following parameters:

- `src`: `str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `model`: `torch.nn.Module`
- `optimizer`: `torch.optim.Optimizer`

Implement the `adapters.run_save_checkpoint` and `adapters.run_load_checkpoint` adapters, and make sure they pass `uv run pytest -k test_checkpointing`.

**Answer:**

实现为 `cs336_basics/training/checkpointer.py` 中的 `save_checkpoint(model, optimizer, iteration, out)` 与 `load_checkpoint(src, model, optimizer) -> int`（函数名按 handout 规定）；测试用的 `adapters.run_save_checkpoint` / `adapters.run_load_checkpoint` 只是调用这两个函数的 glue。

- 保存：`torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": iteration}, out)`；`out` 可以是路径或 file-like 对象。
- 加载：`torch.load(src, map_location="cpu")`，再 `model.load_state_dict(checkpoint["model_state_dict"])`、`optimizer.load_state_dict(checkpoint["optimizer_state_dict"])`，返回 `checkpoint["iteration"]`。先加载到 CPU 再交给模型/优化器，是在不同设备上恢复 checkpoint 的安全做法。

---

## 27. Problem (training_together): Put it together (4 points)

**Deliverables:**
Write a script that runs a training loop to train your model on user-provided input. In particular, we recommend that your training script allow for (at least) the following:

- Ability to configure and control the various model and optimizer hyperparameters.
- Memory-efficient loading of large training and validation datasets with `np.memmap`.
- Serializing checkpoints to a user-provided path.
- Periodically logging training and validation performance (e.g., to console and/or an external service like Weights and Biases).[^9]

**Answer:**

训练脚本是 `train.py`（尚未提交的代码交付物；它组合的模块代码已在仓库中）。设计如下：

- 通过 `argparse`（或 YAML 配置）完全可配：模型（`vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta`）、优化器（`lr, betas, eps, weight_decay`）、schedule（`warmup_iters, cosine_cycle_iters, min_lr`）、循环（`batch_size, max_steps, grad_clip, device, dtype, checkpoint_path, val_every`）。
- 省内存的数据加载：分词后的语料以 `uint16` NumPy 数组保存，用 `np.memmap` 打开，minibatch 由 `run_get_batch` 采样，不在内存里放下整个语料。
- 循环：每步采样 batch -> forward -> cross-entropy -> backward -> 梯度裁剪 -> AdamW step -> 学习率 schedule step；每 `val_every` 步在验证集上评估。
- Checkpoint：定期调用 `save_checkpoint`（model + optimizer + iteration），并支持用 `load_checkpoint` 恢复。
- 日志：打印并在可选时记录到 Weights and Biases 训练/验证 loss，x 轴可切换为梯度步或墙钟时间（`wandb.define_metric(step_metric=...)`）。

运行：`uv run python train.py --config configs/tinystories.yaml`（OpenWebText 同理）。实验日志与曲线由该脚本产出。

---

## 28. Problem (decoding): Decoding (3 points)

**Deliverables:**
Implement a function to decode from your language model. We recommend that you support the following features:

- Generate completions for a user-provided prompt (i.e., take in some $x_{1\dots t}$ and sample a completion until you hit an `<|endoftext|>` token).
- Allow the user to control the maximum number of generated tokens.
- Given a desired temperature value, apply softmax temperature scaling to the predicted next-token distributions before sampling.
- Top-$p$ sampling ([A. Holtzman et al., 2020] also referred to as nucleus sampling), given a user-specified threshold value.

**Answer:**

解码器计划实现在 `generate.py`（尚未提交的代码交付物；方案如下）：

- 编码 prompt，前向得到最后一个位置的 next-token logits，然后自回归采样，直到达到 `max_new_tokens` 或采到 `<|endoftext|>`。
- **Temperature**：softmax 前把 logits 除以用户给定的 `T`（`T -> 0+` 相当于贪心/`argmax`）。
- **Top-p（nucleus）采样**：把概率降序排序，保留累积概率首次达到 `p` 的最短前缀，其余置 0，重新归一化后用 `torch.multinomial` 采样。可选的 top-k 同理，只是用固定 `k`。
- 每个新 token 追加回输入再喂入（可选 KV cache），函数返回解码后的续写文本。

---

## 29. Problem (experiment_log): Experiment logging (3 points)

**Deliverables:**
For your training and evaluation code, create experiment tracking infrastructure that allows you to track your experiments and loss curves with respect to gradient steps and wall-clock time.  
_Deliverable:_ Logging infrastructure code for your experiments and an experiment log (a document of all the things you tried) for the assignment problems below in this section.

**Answer:**

**日志基础设施**（已实现在 `scripts/train.py`）：每个实验一个 Weights and Biases run，`config=vars(args)` 完整记录超参（模型规模、lr、schedule、batch size、grad-clip、dtype、seed、数据路径等）。通过 `wandb.define_metric("step")`、`wandb.define_metric("wall_time")` 与 `define_metric("*", step_metric="step")`，每个指标既能对**梯度步**、也能对**墙钟时间**作图。每个日志点记录：`train/loss`、`train/perplexity`、`lr`、`tokens_per_sec`、`wall_time`、`step`；验证时记录 `val/loss`、`val/perplexity`、`wall_time`。除 wandb 外保留纯文本 stdout 日志；checkpoint 记录 iteration 以便恢复。默认关闭，`--wandb` 开启（可用 `WANDB_MODE=offline` 离线跑）。

另有两条**无需账号**的本地路径：`--log-csv PATH`（零依赖，写 CSV，见 `cs336_basics/training/experiment_log.py`）与 `--tensorboard DIR`（本地 TensorBoard，`http://localhost:6006`；x 轴可切 `Step`/`Wall`，后者即墙钟时间）。注意 wandb 离线模式只把数据写到 `wandb/offline-run-*`，要出网页图仍需 `wandb sync`（那步需要账号）。

**实验日志**（每个 run 一行；结果列在各实验执行后填入——**TODO（结果）**）：

| id | 对应题目 | 相对基线的改动 | lr | batch | steps | val loss | 备注 |
| :-- | :-- | :-- | --: | --: | --: | --: | :-- |
| base | 基线 | TinyStories，17M 模型 | 待定 | 待定 | 待定 | 待定 | 参照 run |
| lr-1 | 30 | lr sweep | {3e-4,1e-3,3e-3,1e-2} | 64 | 相同 | 待定 | 选最优 |
| bs-1 | 31 | batch size sweep | 重新调 | {1,8,32,64,128,256} | 相同 | 待定 | 重调 lr |
| no-rm | 33 | 去掉 RMSNorm | 最优/更低 | 64 | 相同 | 待定 | 稳定性研究 |
| post-n | 34 | post-norm | 最优 | 64 | 相同 | 待定 | 对比 pre-norm |
| nope | 35 | 去掉 RoPE | 最优 | 64 | 相同 | 待定 | 对比 RoPE |
| silu | 36 | SiLU FFN | 最优 | 64 | 相同 | 待定 | 参数量对齐 |
| owt | 37 | OpenWebText，32K | 最优 | 64 | 相同 | 待定 | 对比 TinyStories |
| lb | 38 | leaderboard 配置 | 最优 | 待定 | <=45 分钟 | 待定 | 提交 |

---

## 30. Problem (learning_rate): Tune the learning rate (2 B200 hrs) (3 points)

**Deliverables:**
The learning rate is one of the most important hyperparameters to tune. Taking the base model you’ve trained, answer the following questions:

**(a)** Perform a hyperparameter sweep over the learning rates and report the final losses (or note divergence if the optimizer diverges).  
_Deliverable:_ Learning curves associated with multiple learning rates. Explain your hyperparameter search strategy.  
_Deliverable:_ A model with validation loss (per-token) on _TinyStories_ of at most **1.45**.

**Answer:**

**设置。** 在 TinyStories 上训练 handout 的约 17M 参数模型（`d_model` 约 512、4-6 层、`d_ff` 约 8/3 d、上下文 1024、10K tokenizer），AdamW（betas 0.9/0.999 或 0.9/0.95），线性 warmup + 余弦退火，梯度裁剪，所有 run 固定随机种子与步数预算。

**搜索策略。** 先做对数粗扫，再在最优值附近细化：`lr in {3e-4, 1e-3, 3e-3, 1e-2}`，其余设置完全相同；按最终验证 loss（以及 loss 曲线下面积——早期更快下降往往比末点略差更值）排序。再用更多步数/多随机种子复跑最优的 1-2 个值确认。

**预期结论（实测结果在跑完后填入 TODO）。** lr 太小会欠拟合（到步数预算时曲线仍在下降）；约 `1e-3..3e-3` 时应在 TinyStories 上达到不超过 **1.45** 的每 token 验证 loss；lr >= 1e-2 通常即使有 warmup 也会发散或停在更差的 loss。*学习曲线与最终的最优 lr / 验证 loss 待补（需先实现 `train.py` 并跑实验）。*

---

## 31. Problem (batch_size_experiment): Batch size variations (1 B200 hr) (1 point)

**Deliverables:**
Vary your batch size all the way from 1 to the GPU memory limit. Try at least a few batch sizes in between, including typical sizes like 64 and 128.  
_Deliverable:_ Learning curves for runs with different batch sizes. The learning rates should be optimized again if necessary.  
_Deliverable:_ A few sentences discussing your findings on batch sizes and their impacts on training.

**Answer:**

**设置。** 与 learning-rate 题相同的 TinyStories 模型与步数预算。扫描 `batch_size in {1, 8, 32, 64, 128, 256}`，上限为显存允许；每个 batch size 都重新调 lr（大 batch 通常需要按比例更大的 lr，线性或平方根缩放规则是不错的起点）。

**讨论（曲线/结果待实验完成后补）。** 极小 batch（1-8）梯度噪声大、每个 token 要走很多优化步，按“每步”看早期进展可能更快，但按“每个 token 的 GPU 吞吐”看很浪费，且需要更小的 lr。常见大小（64-128）通常在「固定墙钟时间下的 loss」上最优。大 batch 梯度更平滑、硬件利用率高，但超过一定程度后每次翻倍带来的 loss 收益递减，且需要更大 lr 才能避免每 token 进展变慢。这里应以**固定墙钟预算下的最低验证 loss**（而非固定步数）来选最优。

---

## 32. Problem (generate): Generate text (1 point)

**Deliverables:**
Using your decoder and your trained checkpoint, report the text generated by your model. You may need to manipulate decoder parameters (temperature, top-p, etc.) to get fluent outputs.  
_Deliverable:_ Text dump of at least 256 tokens of text (or until the first `<|endoftext|>` token), and a brief comment on the fluency of this output and at least two factors which affect how good or bad this output is.

**Answer:**

**设置。** 用最优 TinyStories checkpoint 与 top-p/temperature 解码器（第 28 题），输入一个故事开头，采样至少 256 个 token（或直到第一个 `<|endoftext|>`）。

**生成文本：** TODO——待模型训练完成后粘贴（前提是先实现 `train.py`/`generate.py` 并训练出 checkpoint）。

**流畅度讨论（至少两个因素）。**（1）**采样参数**：低 temperature / 低 top-p 文本更流畅但容易重复，高 temperature / 高 top-p 更多样但连贯性下降，可能出现局部无意义或跑题的续写。（2）**训练算力与数据**：只有约 17M 参数的小模型和有限步数时，有效容量与世界知识都有限，续写越长越容易漂移；更多步数、更大模型或更大更多样的语料能提升流畅度。（3）**tokenizer 与上下文长度**也限制了模型能条件化的信息。

---

## 33. Problem (layer_norm_ablation): Remove RMSNorm and train (0.5 B200 hrs) (1 point)

**Deliverables:**
Remove all of the RMSNorms from your Transformer and train. What happens at the previous optimal learning rate? Can you get stability by using a lower learning rate?  
_Deliverable:_ A learning curve for when you remove RMSNorms and train, as well as a learning curve for the best learning rate.  
_Deliverable:_ A few sentences of commentary on the impact of RMSNorm.

**Answer:**

**设置。** 从 Transformer 中去掉所有 `RMSNorm`（块内两个与最终 norm），其余设置不变，先在之前最优的学习率下重训，再在更低学习率下重训。

**预期结果（曲线待实验完成后补）。** 没有归一化时残差流的尺度不受约束，因此在最优 lr 下应当不稳定甚至发散（loss 尖峰/NaN）——Pre-Norm Transformer 依赖 RMSNorm 保持激活条件良好。把学习率显著调低可以恢复短期稳定，但最好的无 RMSNorm run 仍会收敛更慢、最终 loss 更差。该对比说明 RMSNorm 的核心作用是**稳定性/条件数**，而不是那点计算量节省。

---

## 34. Problem (pre_norm_ablation): Implement post-norm and train (0.5 B200 hrs) (1 point)

**Deliverables:**
Modify your pre-norm Transformer implementation into a post-norm one. Train with the post-norm model and see what happens.  
_Deliverable:_ A learning curve for a post-norm Transformer, compared to the pre-norm one.

**Answer:**

**设置。** 把块结构从 pre-norm（`x + attn(ln(x))`）改成 post-norm（`ln(x + attn(x))`，FFN 同理），用相同预算与带 warmup 的学习率重训。

**预期结果（曲线待实验完成后补）。** Post-norm 把归一化放在残差路径上，梯度必须穿过 norm，恒等路径不再干净；这在深层训练中历史上更不稳定、对 lr warmup 更敏感。配合仔细的 warmup，小模型上也可能追平 pre-norm，但预期 post-norm 曲线更抖、在相同步数/墙钟预算下略差——这正是现代 decoder-only LM 采用 pre-norm 的原因。

---

## 35. Problem (no_pos_emb): Implement NoPE (0.5 B200 hrs) (1 point)

**Deliverables:**
Modify your Transformer implementation with RoPE to remove the position embedding information entirely, and see what happens.  
_Deliverable:_ A learning curve comparing the performance of RoPE and NoPE.

**Answer:**

**设置。** 完全去掉 RoPE（不提供任何位置信息，注意力前不对 q/k 做处理），用相同预算训练，并与 RoPE 基线对比 loss 曲线。

**预期结果（曲线待实验完成后补）。** 仅有因果掩码的 decoder-only 模型仍隐含一点位置信息（token i 只能看 j <= i），所以 NoPE 仍能训练，在短上下文下甚至相当接近；但没有显式位置后应当略差，且随上下文变长差距扩大（顺序信息丢失、长程顺序变得不确定）。预期 RoPE 曲线等于或优于 NoPE，且序列越长优势越明显。

---

## 36. Problem (swiglu_ablation): SwiGLU vs. SiLU (0.5 B200 hrs) (1 point)

**Deliverables:** > _Deliverable:_ A learning curve comparing the performance of SwiGLU and SiLU feed-forward networks, with approximately matched parameter counts.  
_Deliverable:_ A few sentences discussing your findings.

**Answer:**

**设置。** 对比 SwiGLU FFN（`W2(silu(W1 x) * W3 x)`，三个矩阵，`d_ff ~= 8/3 d`）与无门控的朴素 SiLU FFN（`W2(silu(W1 x))`，两个矩阵）。通过选择无门控 FFN 的 `d_ff` 使两者**参数量对齐**（约 `2 d d_ff^plain ~= 3 d d_ff^swiglu`，即 `d_ff^plain ~= (3/2) d_ff^swiglu`），并用相同预算训练。

**预期结果（曲线待实验完成后补）。** 参数量对齐时，带门控的 SwiGLU 预期略优（这也是现代 FFN 的标准选择）：门控让网络能对特征做乘性调制，而把同样参数量花在“更宽的无门控 FFN”上不如额外那个矩阵有效。差距通常不大但稳定。

---

## 37. Problem (main_experiment): Experiment on OWT (2 B200 hrs) (2 points)

**Deliverables:**
Train your language model on _OpenWebText_ with the same model architecture and total training iterations as _TinyStories_. How well does this model do?  
_Deliverable:_ A learning curve of your language model on _OpenWebText_. Describe the difference in losses from _TinyStories_ – how should we interpret these losses?  
_Deliverable:_ Generated text from _OpenWebText_ LM, in the same format as the _TinyStories_ outputs. How is the fluency of this text? Why is the output quality worse even though we have the same model and compute budget as _TinyStories_?

**Answer:**

**设置。** 用与 TinyStories 完全相同的架构和总训练迭代数，但在 OpenWebText 上、配 32K OpenWebText tokenizer 训练（这是 2 B200 小时的题目；**尚未运行——本机没有 B200 预算，且 `train.py` 还未提交**）。

**预期结果（曲线待补）。** OpenWebText 大得多也更多样（网页文本，话题/体裁/语言繁多），因此在相同 loss/步数预算下其每 token 交叉熵应**高于** TinyStories：分布熵更高，模型更难覆盖/记忆。生成文本也应**更不流畅、更不连贯**——同样的模型容量与步数预算现在要覆盖宽得多的分布（且主要是英文网页而非简单叙事），相当于相对任务难度被“训练不足”。所以理解 loss 必须按数据集看：OWT 的 loss 更高不代表模型更差，只代表预测问题更难。

---

## 38. Problem (leaderboard): Leaderboard (10 B200 hrs) (6 points)

**Deliverables:**
You will train a model under the leaderboard rules above with the goal of minimizing the validation loss of your language model within 0.75 B200-hours.  
_Deliverable:_ The final validation loss that was recorded, an associated learning curve that clearly shows a wall-clock-time x-axis that is less than 45 minutes, and a description of what you did. We expect a leaderboard submission to beat at least the naive baseline of a 5.0 loss. Submit to the leaderboard here: `github.com/stanford-cs336/assignment1-basics-leaderboard`.

**Answer:**

**设置/策略。** 遵守 leaderboard 规则，在 0.75 B200 小时（45 分钟墙钟）内训练。可行方案：使用 OpenWebText 32K tokenizer；按时间预算选择“小而强”的配置（例如约 30-60M 参数、上下文 512-1024、4-8 层）；用 sweep 得到的最优 lr 配 warmup + 余弦、梯度裁剪、bf16；用较大 batch 拉高 MFU，达到 45 分钟墙钟上限即停止。曲线要按**墙钟时间**（而非步数）记录，以便体现 < 45 分钟，并记录最终验证 loss。

**结果：** TODO——**本环境未运行**（无 B200/预算）；等实际跑完后在此填写提交的最终验证 loss、墙钟学习曲线与完整的配方说明。提交需优于 5.0 loss 的朴素基线。

---
