# CS336 Assignment 1 — Writeup

> 本文件对应 handout 的 38 个 Problem；每题保留题目要求，答案写在 Answer 下。

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

**(a)** *"What does chr(0) return?"* -- `chr(0)` returns the Unicode code point U+0000, the NULL (NUL) control character. It is a valid character, but it has no printable glyph.

**(b)** *"repr() vs. printed representation?"* -- `repr(chr(0))` escapes it so that the zero byte is shown as the four visible characters `\x00` inside quotes, while `print(chr(0))` emits the raw NUL byte itself, which produces no visible output.

**(c)** *"What happens in text?"* -- The NUL byte is really stored inside the string: `len("this is a test" + chr(0) + "string") == 19` and `repr()` shows `'this is a test\x00string'`, but when printed it looks exactly like `this is a teststring`. Because many C/POSIX APIs treat NUL as the string terminator, such strings can be silently truncated when they cross into C-backed libraries -- which is precisely why byte-level tokenization uses `bytes` and never relies on C string semantics.

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

**(a)** *"Why prefer UTF-8 over UTF-16/UTF-32?"* -- UTF-8 is a variable-length encoding that is byte-identical to ASCII on the ASCII range, so mostly-ASCII training corpora cost only 1 byte per character, whereas UTF-16 and UTF-32 always spend 2 and 4 bytes on the same ASCII character (UTF-32 wastes the most). UTF-8 is also endianness-free, self-synchronizing, and can represent every Unicode code point, so a byte-level tokenizer over UTF-8 has no OOV and matches how web text is actually stored and served.

**(b)** *"Why is decode_utf8_bytes_to_str_wrong incorrect?"* -- It decodes each byte independently, but a UTF-8 multi-byte character is only valid as a complete sequence; a lone lead or continuation byte is not a valid one-byte UTF-8 codepoint. For example, `b'\xc3\xa9'` (the UTF-8 encoding of "é") raises `UnicodeDecodeError` on the first byte instead of returning "é", so the function can never decode any non-ASCII character and is not a correct UTF-8 decoder.

**(c)** *"A two-byte sequence that decodes to nothing?"* -- `b'\xff\xfe'`: byte 0xFF never occurs in well-formed UTF-8 and 0xFE can only be a continuation byte, so `b'\xff\xfe'.decode("utf-8")` raises `UnicodeDecodeError` and decodes to no Unicode character(s).

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

Implemented in `cs336_basics/bpe_tokenizer/trainer.py::run_train_bpe` and exposed to the tests through `tests/adapters.py::run_train_bpe`.

**Algorithm.** (1) The corpus is split on the special tokens, which act as hard boundaries, so no merge can cross a document boundary and special-token strings are excluded from the merge statistics. (2) Each remaining piece is pre-tokenized with the GPT-2 regex (using the fast `regex` package; see `trainer.py`); every match is encoded to UTF-8 and stored as a tuple of single bytes, and word-type frequencies are accumulated. For large corpora the pieces are processed with `multiprocessing.Pool` and the counts are merged. (3) The vocabulary starts as the 256 single-byte tokens (IDs 0..255), followed by the special tokens. (4) We perform `vocab_size - 256 - len(special_tokens)` merges: each step selects the adjacent pair with the highest total frequency, breaking ties by choosing the lexicographically **greatest** pair, appends it to `merges`, and creates the merged token `b1 + b2`. Pair frequencies are maintained incrementally -- only word types containing the merged pair are rewritten and their old/new adjacent-pair counts adjusted -- so the loop does not rescan every word type each step.

**Output.** `vocab: dict[int, bytes]` (base bytes, then special tokens, then merge products in creation order) and `merges: list[tuple[bytes, bytes]]` ordered by creation, exactly as the interface requires. The details that matter for correctness are the GPT-2 regex, byte-level (not character-level) symbols, hard special-token boundaries, and the lexicographic tie-break.

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

**(a)** Train with `run_train_bpe("data/TinyStoriesV2-GPT4-train.txt", vocab_size=10000, special_tokens=["<|endoftext|>"])` and serialize the returned `(vocab, merges)` to disk. Special tokens delimit documents and `multiprocessing` pre-tokenization parallelizes the expensive step, which keeps this within the handout's 30-minute / 30-GB budget (hint: under 2 minutes).

- *Time and memory:* **TODO (measure with the run above)** -- wall-clock with `time` and peak RSS with `psutil` (or `resource.getrusage(RUSAGE_SELF).ru_maxrss`).
- *Longest token:* **TODO** -- `max(vocab.values(), key=len)`. It should be a complete common word or word fragment (e.g. something like a very frequent short word); that makes sense for TinyStories, whose corpus is simple, repetitive English children's stories, so the highest-value merges are whole frequent words.

**(b)** *"What part takes the most time?"* -- **TODO (confirm with cProfile)**: we expect the pre-tokenization pass over the whole corpus (regex matching plus byte-tuple construction) to dominate, because the merge loop only touches the deduplicated word-type table. Multiprocessing pre-tokenization is exactly the optimization that hides this cost.

---

## 5. Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

**(a)** Train a byte-level BPE tokenizer on the _OpenWebText_ dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?  
_Resource requirements: $\le$ 12 hours (no GPUs), $\le$ 100 GB RAM_  
_Deliverable:_ A one-to-two sentence response.

**(b)** Compare and contrast the tokenizer that you get training on _TinyStories_ versus _OpenWebText_.  
_Deliverable:_ A one-to-two sentence response.

**Answer:**

**(a)** Train with `run_train_bpe("data/owt_train.txt", vocab_size=32000, special_tokens=["<|endoftext|>"])` and serialize the result. This is the 12-hour / 100-GB problem. *Longest token:* **TODO (measure)**; for OpenWebText we expect much longer tokens than for TinyStories, because the corpus contains URLs, code, HTML remnants, long numbers and rare compounds that the merge process can still fuse into long byte strings, so long tokens are both plausible and heterogeneous.

**(b)** *"Compare the two tokenizers."* **TODO (measure after training both)**, but qualitatively: the 32K OpenWebText tokenizer has a larger vocabulary and was fit to a much more diverse distribution, so it should compress English web text better (fewer tokens per byte), while the 10K TinyStories tokenizer is biased toward simple narrative English and spends many merges on children's-story vocabulary. Cross-applying the TinyStories tokenizer to OpenWebText (Problem 7(b)) makes the domain mismatch concrete in the compression ratio.

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

Implemented in `cs336_basics/bpe_tokenizer/base.py::Tokenizer` and exposed via `tests/adapters.py::get_tokenizer`.

- `__init__(vocab, merges, special_tokens)`: stores the vocabulary, builds a `bytes -> id` dictionary, and turns the ordered `merges` list into a per-pair priority map `{(a, b): rank}` where a smaller rank means higher priority. If special tokens are given it compiles a regex matching them, sorted longest-first so that overlapping specials split correctly.
- `encode(text)`: first splits on the special-token regex; special spans map straight to their ids (no BPE), and the remaining spans are cut with the GPT-2 regex. Each pre-token is represented as a list of single bytes and merged greedily by repeatedly merging the adjacent pair with the smallest `rank` (the highest-priority merge that actually occurs) until nothing can be merged; the final symbols are mapped to ids.
- `encode_iterable(iterable)`: a lazy generator that feeds one string at a time into `encode`, so a multi-GB file can be tokenized in constant, sub-1-MB memory.
- `decode(ids)`: concatenates `vocab[id]` for each id and does `.decode("utf-8", errors="replace")`, so malformed byte sequences degrade to U+FFFD instead of raising.
- `from_files(vocab_filepath, merges_filepath, special_tokens)`: class method that loads the serialized vocabulary and merges produced by the training code and constructs a `Tokenizer` (the graded tests only exercise the `__init__` path through `get_tokenizer`).

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

**(a)** *Compression ratio (bytes/token).* **TODO (measure)**: sample 10 documents from each dataset, encode with the matching tokenizer, and report `len(text.encode("utf-8")) / len(tokenizer.encode(text))`. We expect roughly 3.5-4.5 bytes/token for TinyStories-10K and a bit higher for OpenWebText-32K, since the larger, more diverse vocabulary captures more subwords.

**(b)** *Tokenize OWT with the TinyStories tokenizer.* **TODO (measure)**, but the TinyStories tokenizer has never seen most OpenWebText vocabulary, so it falls back to single bytes and short merges: the compression ratio **drops** (more tokens per byte) and sequences become much longer. This is the token-blowup effect from the concept notes, and using it for pretraining would inflate cost through the quadratic attention term.

**(c)** *Throughput.* **TODO (measure)**: time `encode` on a large chunk, report bytes/second, then estimate the Pile as 825 GB divided by that throughput. With the hash-priority merge loop we expect on the order of 10^7 bytes/s on CPU, i.e. on the order of tens of hours for the Pile.

**(d)** *Why uint16?* The vocabulary is at most 32,000 here (even GPT-2's 50,257 fits), so every token id is below 65536 and is stored exactly in an unsigned 16-bit integer. That halves the memory and I/O of the tokenized corpus versus `int32`/`int64` while still covering the entire vocabulary.

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

Implemented as `Linear` in `cs336_basics/notation/{no_einstein,einstein}/layers.py` (re-exported by `cs336_basics/model/layers.py` and selected by the `CS336_NOTATION` env var); exposed via `adapters.run_linear`.

- `__init__(in_features, out_features, device, dtype)`: stores `weight = nn.Parameter(torch.empty(out_features, in_features))` and initializes it with `torch.nn.init.trunc_normal_(weight, 0.0, std, -3*std, 3*std)` where `std = sqrt(2 / (in_features + out_features))`. There is no bias and `W` is stored as `(d_out, d_in)` (not transposed).
- `forward(x)`: computes `y = x W^T`. The plain version uses `x @ self.weight.t()`; the Einstein version uses `einx.dot("... d_in, d_out d_in -> ... d_out", x, self.weight)`, which avoids the manual transpose and accepts any number of leading batch dimensions.

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

Implemented as `Embedding` in `layers.py` (both notation versions), exposed via `adapters.run_embedding`.

- `__init__(num_embeddings, embedding_dim, device, dtype)`: `weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))`, initialized with `trunc_normal_(mean=0, std=1, a=-3, b=3)`; `d_model` is the final dimension.
- `forward(token_ids)`: plain version uses direct indexing `self.weight[token_ids]`; the Einstein version uses `einx.get_at("[v] d, ... -> ... d", self.weight, token_ids)`. Both support arbitrary leading batch dimensions and return `(..., d_model)`.

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

Implemented as `RMSNorm` in `layers.py`, exposed via `adapters.run_rmsnorm`.

- `__init__(d_model, eps=1e-5)`: a learnable gain `weight = nn.Parameter(torch.ones(d_model))`.
- `forward(x)`: upcast to `float32`; compute the mean of squares over the last dimension with keepdim (plain: `torch.mean(x**2, dim=-1, keepdim=True)`; Einstein: `einx.mean("... ([d])", x**2)`); `rms = sqrt(mean_square + eps)`; normalize `x / rms`; downcast back to the input dtype; multiply by the gain (plain `*`, Einstein `einx.multiply("... d, d -> ... d", normalized, weight)`). Input and output have the same shape.

---

## 11. Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)

**Deliverables:**
Implement the SwiGLU feed-forward network, composed of a SiLU activation function and a GLU.

**Note:** In this particular case, you should feel free to use `torch.sigmoid` in your implementation for numerical stability.

You should set $d_{ff}$ to approximately $\frac{8}{3} \times d_{model}$ in your implementation, while ensuring that the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware. To test your implementation against our provided tests, you will need to implement the test adapter at `adapters.run_swiglu`. Then, run `uv run pytest -k test_swiglu` to test your implementation.

**Answer:**

Implemented as `SwiGLU` (plus `silu`) in `layers.py`, exposed via `adapters.run_swiglu`.

- Three bias-free `Linear` layers: `w1, w3: d_model -> d_ff` and `w2: d_ff -> d_model`.
- `forward(x) = w2( silu(w1(x)) * w3(x) )` with `silu(z) = z * sigmoid(z)` (the gate branch is activated, multiplied element-wise with the value branch, then projected down). The element-wise product is plain `*` in the no-Einstein version and `einx.multiply("... f, ... f -> ... f", gate, w3x)` in the Einstein version.
- `d_ff` is chosen as the multiple of 64 nearest to `(8/3) * d_model` for hardware efficiency.

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

Implemented as `RotaryPositionalEmbedding` in `cs336_basics/notation/{no_einstein,einstein}/attention.py`, exposed via `adapters.run_rope`.

- `__init__(theta, d_k, max_seq_len, device)` precomputes and registers the buffer `freqs[k] = theta ** (-2k / d_k)` for `k = 0 .. d_k/2 - 1`.
- `forward(x, token_positions)`: `angles = token_positions.unsqueeze(-1) * freqs`, then `cos`/`sin`. Adjacent feature pairs are rotated as a 2D rotation: `out_even = x_even*cos - x_odd*sin`, `out_odd = x_even*sin + x_odd*cos`. `x` has shape `(..., seq_len, d_k)` and positions `(..., seq_len)`; arbitrary leading batch dimensions broadcast. Plain version slices `x[..., 0::2]`/`x[..., 1::2]` and writes into `torch.empty_like(x)`. Einstein version splits with `einx.id("... s (d pair) -> ... s d pair", x, pair=2)`, builds the 2x2 rotation matrices and contracts with `einx.dot("... p i j, ... p j -> ... p i")`, using a small `_broadcast_leading` helper because `einx.dot` does not broadcast across different ranks.
- Because cos/sin are indexed by `token_positions`, positions can be arbitrary (not only `0..s-1`).

---

## 13. Problem (softmax): Implement softmax (1 point)

**Deliverables:**
Write a function to apply the softmax operation on a tensor. Your function should take two parameters: a tensor and a dimension $i$, and apply softmax to the $i$-th dimension of the input tensor. The output tensor should have the same shape as the input tensor, but its $i$-th dimension will now have a normalized probability distribution. Use the trick of subtracting the maximum value in the $i$-th dimension from all elements of the $i$-th dimension to avoid numerical stability issues.

To test your implementation, complete `adapters.run_softmax` and make sure it passes `uv run pytest -k test_softmax_matches_pytorch`.

**Answer:**

Implemented as `softmax(x, dim=-1)` in `attention.py`, exposed via `adapters.run_softmax`.

Numerically stable form: subtract the max along `dim` (with `keepdim=True`), exponentiate, and divide by the sum along `dim` (with `keepdim=True`), so overflow cannot occur even when the inputs are shifted by +100. The Einstein version expresses the reduction with `einx.softmax` by marking the reduced axis in brackets (e.g. `einx.softmax("a0 [a1] a2", x)`). Both match `torch.nn.functional.softmax` within tolerance.

---

## 14. Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)

**Deliverables:**
Implement the scaled dot-product attention function. Your implementation should handle keys and queries of shape `(batch_size, ..., seq_len, d_k)` and values of shape `(batch_size, ..., seq_len, d_v)`, where `...` represents any number of other batch-like dimensions (if provided). The implementation should return an output with the shape `(batch_size, ..., seq_len, d_v)`. See Section 3.2 for a discussion on batch-like dimensions.

Your implementation should also support an optional user-provided boolean mask of shape `(seq_len, seq_len)`. The attention probabilities of positions with a mask value of `True` should collectively sum to 1, and the attention probabilities of positions with a mask value of `False` should be zero.

To test your implementation against our provided tests, you will need to implement the test adapter at `adapters.run_scaled_dot_product_attention`. `uv run pytest -k test_scaled_dot_product_attention` tests your implementation on third-order input tensors, while `uv run pytest -k test_4d_scaled_dot_product_attention` tests your implementation on fourth-order input tensors.

**Answer:**

Implemented as `scaled_dot_product_attention(Q, K, V, mask=None)` in `attention.py`, exposed via `adapters.run_scaled_dot_product_attention`.

- `d_k = Q.size(-1)`; `scores = Q K^T / sqrt(d_k)`.
- If a boolean mask is given it is applied before softmax with `scores.masked_fill(~mask, -inf)`, so masked positions get probability 0 and the unmasked ones sum to 1.
- `probs = softmax(scores, dim=-1)`; output `= probs V`.
- Q/K/V may have any number of leading batch-like dimensions (the `...` axes); the mask broadcasts over them. Plain version uses `torch.matmul` with an explicit transpose; Einstein version uses `einx.dot("... q d, ... k d -> ... q k")` and `einx.dot("... q k, ... k v -> ... q v")`. Output shape is `Q.shape[:-1] + (d_v,)`.

---

## 15. Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)

**Deliverables:**
Implement causal multi-head self-attention as a `torch.nn.Module`. Your implementation should accept (at least) the following parameters:

- `d_model`: `int` Dimensionality of the Transformer block inputs.
- `num_heads`: `int` Number of heads to use in multi-head self-attention.

Following A. Vaswani et al. [8], set $d_k = d_v = \frac{d_{model}}{h}$. To test your implementation against our provided tests, implement the test adapter at `adapters.run_multihead_self_attention`. Then, run `uv run pytest -k test_multihead_self_attention` to test your implementation.

**Answer:**

Implemented as `CausalMultiHeadSelfAttention` in `attention.py`, exposed via `adapters.run_multihead_self_attention`.

- Four bias-free `Linear` projections: `q_proj, k_proj, v_proj, output_proj`, each `d_model -> d_model`.
- `d_k = d_v = d_model // num_heads`. `forward` projects `x` into `(b, s, d_model)`, splits into `(b, num_heads, s, d_k)` (plain: `reshape + transpose`; Einstein: `einx.id("b s (h d) -> b h s d", q, h=num_heads)`), optionally applies RoPE to q and k, builds the lower-triangular causal mask `torch.tril(torch.ones(s, s, dtype=bool))`, calls the scaled dot-product attention, merges heads back to `(b, s, d_model)`, and applies `output_proj`. Matches the naive unbatched reference within tolerance.

---

## 16. Problem (transformer_block): Implement the Transformer block (3 points)

**Deliverables:**
Implement the pre-norm Transformer block as described in Section 3.4 and illustrated in Figure 2. Your Transformer block should accept (at least) the following parameters:

- `d_model`: `int` Dimensionality of the Transformer block inputs.
- `num_heads`: `int` Number of heads to use in multi-head self-attention.
- `d_ff`: `int` Dimensionality of the position-wise feed-forward inner layer.

To test your implementation, implement the adapter `adapters.run_transformer_block`. Then run `uv run pytest -k test_transformer_block` to test your implementation.

**Answer:**

Implemented as `TransformerBlock` in `cs336_basics/notation/{no_einstein,einstein}/transformer.py`, exposed via `adapters.run_transformer_block`.

Pre-norm residual structure: `z = x + attn(ln1(x))` and `y = z + ffn(ln2(z))`, where `ln1/ln2` are `RMSNorm`, `attn` is the causal multi-head self-attention (with RoPE when provided) and `ffn` is the SwiGLU feed-forward network. Input/output shape `(b, s, d_model)`.

---

## 17. Problem (transformer_lm): Implementing the Transformer LM (3 points)

**Deliverables:**
Time to put it all together! Implement the Transformer language model as described in Section 3.1 and illustrated in Figure 1. At minimum, your implementation should accept all the aforementioned construction parameters for the Transformer block, as well as these additional parameters:

- `vocab_size`: `int` The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
- `context_length`: `int` The maximum context length, necessary for determining the dimensionality of the RoPE sin and cos buffer.
- `num_layers`: `int` The number of Transformer blocks to use.

To test your implementation against our provided tests, you will first need to implement the test adapter at `adapters.run_transformer_lm`. Then, run `uv run pytest -k test_transformer_lm` to test your implementation.

**Answer:**

Implemented as `BasicsTransformerLM` in `transformer.py`, exposed via `adapters.run_transformer_lm`.

Components, in order: token `Embedding` (vocab_size, d_model); a shared `RotaryPositionalEmbedding` with `d_k = d_model // num_heads` and length `context_length`; `num_layers` pre-norm `TransformerBlock`s; a final `RMSNorm`; and an `lm_head` `Linear(d_model, vocab_size)`. `forward(token_ids)` builds positions `torch.arange(s)`, embeds the tokens, runs every block with those positions and the shared RoPE, applies the final norm and the LM head, and returns logits of shape `(batch, seq_len, vocab_size)`.

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

Throughout, `b` is batch size, `s = context_length = 1024`, `V = vocab_size`, `L` layers, model width `d`, `h` heads and inner width `d_ff`. A matmul `(m x n)(n x p)` costs `2mnp` FLOPs.

**Parameter count.** Independent token embedding and LM head (no weight tying):
P = `2*V*d + L*(4*d^2 + 3*d*d_ff + 2*d) + d`
(the `4d^2` is Q,K,V,O; the `3*d*d_ff` is W1,W3,W2; the `2d` is the two block RMSNorms; the last `d` is the final RMSNorm).

**(a)** For GPT-2 XL there are **P = 1,640.5M = 1.64B parameters** (using `d_ff = 4288`). At 4 bytes/parameter in fp32, merely loading the weights needs `4P = 6.56 GB` (6.11 GiB).

**(b) Matrix multiplies in one forward pass (batch `b`, seq `s`) and their FLOPs**

| multiply | shape | FLOPs (whole model) |
| :-- | :-- | --: |
| QKV projections | `(b s d)(d 3d)` per layer | `L * 6 b s d^2` = 7.55e14 |
| Output projection | `(b s d)(d d)` per layer | `L * 2 b s d^2` = 2.52e14 |
| Attention scores QK^T | `(b h s d_k)(b h d_k s)` per layer | `L * 2 b s^2 d` = 1.61e11 |
| Weighted sum of values | `(b h s s)(b h s d_v)` per layer | `L * 2 b s^2 d` = 1.61e11 |
| FFN W1/W3 then W2 | `(b s d)(d d_ff)`, `(b s d_ff)(d_ff d)` per layer | `L * 6 b s d d_ff` = 2.02e15 |
| LM head logits | `(b s d)(d V)` | `2 b s d V` = 1.65e14 |

For `b = 1`, `s = 1024` the total is **≈ 3.52 TFLOP per forward pass** (with `b`, multiply everything by `b`).

**(c)** At `s = 1024` the **feed-forward network dominates**: FFN ≈ 57.5%, QKV projections ≈ 21.5%, output projection ≈ 7.2%, LM head ≈ 4.7%, and the two attention matmuls (QK^T and AV) ≈ 4.6% each. So the parameter-heavy `d^2` matmuls account for most of the FLOPs at this context length, not the `s^2` attention scores.

**(d) Proportional FLOPs for the other GPT-2 sizes (forward, `b = 1`, `s = 1024`)**

| model | QKV | out | QK^T | AV | FFN | logits | total |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| small (L12, d768, h12, `d_ff`=2048) | 14.9% | 5.0% | 6.6% | 6.6% | 39.8% | 27.1% | 291.6 GFLOP |
| medium (L24, d1024, h16, `d_ff`=2752) | 18.6% | 6.2% | 6.2% | 6.2% | 50.1% | 12.7% | 830.2 GFLOP |
| large (L36, d1280, h20, `d_ff`=3392) | 20.5% | 6.8% | 5.5% | 5.5% | 54.3% | 7.4% | 1768.5 GFLOP |
| XL (L48, d1600, h25, `d_ff`=4288) | 21.5% | 7.2% | 4.6% | 4.6% | 57.5% | 4.7% | 3516.8 GFLOP |

As the model grows, the `d^2` and `d*d_ff` terms (FFN and QKV) take a **proportionally larger** share, while the **LM head** and the `s^2` **attention** terms shrink -- because the vocabulary term only scales like `d` and the attention term is independent of `d`, whereas the block computation scales like `L*d^2`. (All `d_ff` here are the nearest multiple of 64 to `8d/3`.)

**(e)** Increasing GPT-2 XL's context from 1024 to 16,384 raises the forward cost from 3.52 TFLOP to **133.6 TFLOP, a 38.0x increase**. The `s^2` attention terms now dominate: QK^T and AV each go from 4.6% to 30.9% (62% combined), while the FFN share falls from 57.5% to 24.2% and the LM head from 4.7% to 2.0%. For long context, attention FLOPs -- and its quadratic activation memory -- become the bottleneck.

---

## 19. Problem (cross_entropy): Implement cross-entropy (1 point)

**Deliverables:**
Write a function to compute the cross-entropy loss, which takes in predicted logits ($o_i$) and targets ($x_{i+1}$) and computes the cross-entropy $\ell_i = -\log \text{softmax}(o_i)[x_{i+1}]$. Your function should handle the following:

- Subtract the largest element for numerical stability.
- Cancel out `log` and `exp` whenever possible.
- Handle any additional batch dimensions and return the _average_ across the batch. As with Section 3.2, we assume batch-like dimensions always come first, before the vocabulary size dimension.

Implement `adapters.run_cross_entropy`, then run `uv run pytest -k test_cross_entropy` to test your implementation.

**Answer:**

Implemented as `cross_entropy(inputs, targets)` in `cs336_basics/notation/{no_einstein,einstein}/cross_entropy.py` and re-exported by `training/optimizers.py`; exposed via `adapters.run_cross_entropy`.

Flatten all batch dimensions to `(N, vocab_size)` and targets to `(N,)`. Subtract the per-row max for stability, compute `logsumexp` of the stabilized logits, and gather the target logit at each row; the loss is the mean of `-target_logit + logsumexp`. Subtracting the max and using the stabilized logits directly cancels the separate `log`/`exp` around the target term. The Einstein version uses `einx.logsumexp("n [v]", x)` and `einx.get_at("n [v], n -> n", x, t)`; the plain version uses an explicit sum/exp/log and advanced indexing. Matches `F.cross_entropy` within 1e-4, including for 1000x-scaled inputs.

---

## 20. Problem (learning_rate_tuning): Tuning the learning rate (1 point)

**Deliverables:**
As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s see that in practice in our toy example. Run the SGD example above with three other values for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of training)?  
_Deliverable:_ A one-to-two sentence response with the behaviors you observed.

**Answer:**

I ran the toy loop ( `weights = nn.Parameter(5 * torch.randn(10, 10))`, loss `= (weights**2).mean()`, 10 iterations, the decaying SGD of Equation 20). Measured first -> last loss:

- `lr = 1e1`: decays much faster than the baseline `lr = 1` -- `26.27 -> 3.53` (baseline `26.27 -> 21.74`).
- `lr = 1e2`: decays fastest of the three -- `26.27 -> ~0` (it reaches `1.1e-16` by iteration 5), i.e. it converges almost immediately.
- `lr = 1e3`: **diverges** -- after the first step the loss is `9.48e3` and it grows monotonically to `2.44e18` by iteration 10 (no NaN, but clearly past the stability threshold).

So within the 10-iteration toy, increasing the learning rate speeds up decay from 1e1 to 1e2, while 1e3 is too large and makes the loss explode.

---

## 21. Problem (adamw): Implement AdamW (2 points)

**Deliverables:**
Implement the AdamW optimizer as a subclass of `torch.optim.Optimizer`. Your class should take the learning rate $\alpha$ in `__init__`, as well as the $\beta, \epsilon$ and $\lambda$ hyperparameters. To help you keep state, the base `Optimizer` class gives you a dictionary `self.state`, which maps `nn.Parameter` objects to a dictionary that stores any information you need for that parameter (for AdamW, this would be the moment estimates). Implement `adapters.get_adamw_cls` and make sure it passes `uv run pytest -k test_adamw`.

**Answer:**

Implemented as `AdamW(torch.optim.Optimizer)` in `cs336_basics/training/optimizers.py`, exposed via `adapters.get_adamw_cls`.

- `__init__(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)` validates the hyperparameters and passes them to the base `Optimizer` as defaults.
- `step` keeps per-parameter state `(step, exp_avg, exp_avg_sq)`. For each parameter with a gradient it (1) applies decoupled weight decay `p -= lr * weight_decay * p`, (2) updates the first moment `m = b1*m + (1-b1)*g` and second moment `v = b2*v + (1-b2)*g^2`, and (3) applies the bias-corrected update `p -= lr * sqrt(1-b2^t)/(1-b1^t) * m / (sqrt(v) + eps)`.
- Matches PyTorch's `torch.optim.AdamW` within the test tolerance (the test accepts either the reference snapshot or PyTorch's weights).

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

Assume fp32 (4 bytes) everywhere, `b` = batch size, `s = context_length`, `V` = vocab_size, `L` layers, width `d`, `h` heads, and `d_ff = (8/3)d`. Let `P = 2Vd + L(4d^2 + 3d d_ff + 2d) + d`.

**(a) Peak memory.** Decomposed into:

- **Parameters:** `4P` bytes.
- **Gradients:** `4P` bytes (one gradient per parameter).
- **Optimizer state (AdamW):** the first and second moments `m` and `v`, so `2 * 4P = 8P` bytes.
- **Activations** (only the listed components; per token we count, per layer, RMSNorm outputs `2d`, QKV `3d`, QK^T `h s`, softmax `h s`, weighted sum `d`, output projection `d`, W1 `d_ff`, W2 `d`, SiLU `d_ff`, element-wise product `d_ff`, W3 `d_ff`; the final RMSNorm adds `d`, the LM head `V`, and cross-entropy 1):
  `A = 4 * b * s * ( L*(8d + 4 d_ff + 2 h s) + d + V + 1 )` bytes.

**Total** = `16P + A` = `16P + 4 b s (L(8d + 4d_ff + 2hs) + d + V + 1)` bytes.

**(b) GPT-2 XL instantiation.** With `V=50257, L=48, d=1600, h=25, d_ff=4288, s=1024`: `P = 1.6405e9`, so `16P = 2.625e10` bytes = **24.44 GiB**, and the activation coefficient is `4*s*3,947,154 = 1.617e10` bytes = **15.06 GiB per unit of batch**. Hence

  `memory(batch) = 15.06 GiB * batch + 24.44 GiB`

Within an 80 GiB budget: `batch_max = (80 - 24.44)/15.06 = 3.69`, so the **maximum batch size is 3** (the same answer, 3, if 80 GB is interpreted as 80e9 bytes). Note this is dominated by storing full activations; gradient checkpointing or a smaller context would allow a much larger batch.

**(c) AdamW FLOPs per step.** Running AdamW is entirely element-wise over the `P` parameters: the two moment updates (`m` and `v`), the decoupled weight decay, and the final update (`sqrt`, `div`, multiply, subtract). Counting each elementwise operation once and the multiply-accumulate pairs as two FLOPs, this is roughly **12P FLOPs per optimization step** (order 10P; it is negligible next to the forward/backward passes).

**(d) Training time on one H100.** One forward pass at `b=1024, s=1024` costs `F = 3.60e15` FLOPs. With the backward pass at `2F` and the optimizer step `12P`, one step costs `3F + 12P = 1.08e16` FLOPs (the optimizer is < 0.001%). Over 400K steps the total is `4.32e21` FLOPs. At 50% MFU on one H100 (`0.5 * 495e12 = 2.475e14` FLOP/s):

  `time = 4.32e21 / 2.475e14 = 1.75e7 s = 4,850 hours = 202 days`

So a single H100 would need on the order of **~4,850 hours (~200 days)** to train GPT-2 XL for 400K steps at batch size 1024, i.e. the workload only makes sense on a multi-GPU cluster.

---

## 23. Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup (1 point)

**Deliverables:**
Write a function that takes $t, \alpha_{\text{max}}, \alpha_{\text{min}}, T_w$ and $T_c$, and returns the learning rate $\alpha_t$ according to the scheduler defined above. Then implement `adapters.get_lr_cosine_schedule` and make sure it passes `uv run pytest -k test_get_lr_cosine_schedule`.

**Answer:**

Implemented as `run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)` in `cs336_basics/training/schedulers.py`, exposed via `adapters.run_get_lr_cosine_schedule`.

- `it < warmup_iters`: linear warmup, `(it / warmup_iters) * max_lr`.
- `warmup_iters <= it <= cosine_cycle_iters`: `decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)`, `coeff = 0.5 * (1 + cos(pi * decay_ratio))`, return `min_lr + coeff * (max_lr - min_lr)`.
- `it > cosine_cycle_iters`: return `min_lr` (and the `warmup == cycle` edge case returns `min_lr`).

Matches the reference learning-rate table in the tests.

---

## 24. Problem (gradient_clipping): Implement gradient clipping (1 point)

**Deliverables:**
Write a function that implements gradient clipping. Your function should take a list of parameters and a maximum $\ell_2$-norm. It should modify each parameter gradient in place. Use $\epsilon = 10^{-6}$ (the PyTorch default). Then, implement the adapter `adapters.run_gradient_clipping` and make sure it passes `uv run pytest -k test_gradient_clipping`.

**Answer:**

Implemented as `run_gradient_clipping(parameters, max_l2_norm)` in `cs336_basics/training/clipping.py`, exposed via `adapters.run_gradient_clipping`.

It computes the global L2 norm `total_norm = sqrt(sum_p sum(grad_p^2))` over all parameter gradients that exist (detached, so no graph is built), and if `total_norm > max_l2_norm` it scales every gradient in place by `clip_coef = max_l2_norm / (total_norm + 1e-6)`. Frozen parameters without gradients are skipped. Matches `torch.nn.utils.clip_grad_norm_` within the test tolerance.

---

## 25. Problem (data_loading): Implement data loading (2 points)

**Deliverables:**
Write a function that takes a numpy array $x$ (integer array with token IDs), a `batch_size`, a `context_length` and a PyTorch device string (e.g., `'cpu'` or `'cuda:0'`), and returns a pair of tensors: the sampled input sequences and the corresponding next-token targets. Both tensors should have shape `(batch_size, context_length)` containing token IDs, and both should be placed on the requested device. To test your implementation against our provided tests, you will first need to implement the test adapter at `adapters.run_get_batch`. Then, run `uv run pytest -k test_get_batch` to test your implementation.

**Answer:**

Implemented as `run_get_batch(dataset, batch_size, context_length, device)` in `cs336_basics/training/dataloader.py`, exposed via `adapters.run_get_batch`.

Sample `batch_size` uniform random start indices in `[0, len(dataset) - context_length)`; for each start take `x = dataset[start : start + context_length]` and the next-token target `y = dataset[start + 1 : start + context_length + 1]`. Stack, cast to `torch.long` and move both tensors to the requested device, returning `(x, y)` each of shape `(batch_size, context_length)`.

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

Implemented as `run_save_checkpoint(model, optimizer, iteration, out)` and `run_load_checkpoint(src, model, optimizer) -> int` in `cs336_basics/training/checkpointer.py`, exposed via `adapters.run_save_checkpoint` / `run_load_checkpoint`.

- Save: `torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "iteration": iteration}, out)`; `out` may be a path or a file-like object.
- Load: `torch.load(src, map_location="cpu")`, then `model.load_state_dict(checkpoint["model_state_dict"])` and `optimizer.load_state_dict(checkpoint["optimizer_state_dict"])`, returning `checkpoint["iteration"]`. Loading to CPU first and then letting the model/optimizer cast is the safe way to restore a checkpoint saved on another device.

---

## 27. Problem (training_together): Put it together (4 points)

**Deliverables:**
Write a script that runs a training loop to train your model on user-provided input. In particular, we recommend that your training script allow for (at least) the following:

- Ability to configure and control the various model and optimizer hyperparameters.
- Memory-efficient loading of large training and validation datasets with `np.memmap`.
- Serializing checkpoints to a user-provided path.
- Periodically logging training and validation performance (e.g., to console and/or an external service like Weights and Biases).[^9]

**Answer:**

The training-loop script is `train.py` (the remaining code deliverable; not yet committed -- the module code it composes is already in the repo). Its design:

- Fully configurable via CLI/`argparse` (or a YAML config): model (`vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta`), optimizer (`lr, betas, eps, weight_decay`), schedule (`warmup_iters, cosine_cycle_iters, min_lr`), and loop (`batch_size, max_steps, grad_clip, device, dtype, checkpoint_path, val_every`).
- Memory-efficient data loading: the tokenized corpus is stored as a `uint16` NumPy array and opened with `np.memmap`, and minibatches are drawn with the `run_get_batch` loading function, so no full corpus is held in RAM.
- Loop: for each step -- sample a batch, forward, cross-entropy, backward, gradient clipping, AdamW step, cosine-schedule step -- and every `val_every` steps evaluate the validation loss on held-out data.
- Checkpointing: periodically call `run_save_checkpoint` (model + optimizer + iteration) and support resuming with `run_load_checkpoint`.
- Logging: print (and optionally log to Weights and Biases) the training/validation loss against both gradient step and wall-clock time, using `wandb.define_metric(step_metric=...)` so the x-axis can be chosen later.

Run: `uv run python train.py --config configs/tinystories.yaml` (and the OpenWebText equivalent). Logging infrastructure and curves are produced by this script.

---

## 28. Problem (decoding): Decoding (3 points)

**Deliverables:**
Implement a function to decode from your language model. We recommend that you support the following features:

- Generate completions for a user-provided prompt (i.e., take in some $x_{1\dots t}$ and sample a completion until you hit an `<|endoftext|>` token).
- Allow the user to control the maximum number of generated tokens.
- Given a desired temperature value, apply softmax temperature scaling to the predicted next-token distributions before sampling.
- Top-$p$ sampling ([A. Holtzman et al., 2020] also referred to as nucleus sampling), given a user-specified threshold value.

**Answer:**

The decoder is implemented in `generate.py` (the remaining code deliverable; the plan is fully specified here):

- Encode the prompt, run the model to get the next-token logits at the last position, and sample autoregressively until `max_new_tokens` is reached or an `<|endoftext|>` token is sampled.
- **Temperature**: divide the logits by the user-supplied `T` before softmax (`T -> 0+` gives greedy/`argmax` decoding).
- **Top-p (nucleus) sampling**: sort the probabilities descending, keep the shortest prefix whose cumulative probability is at least `p`, zero the rest, renormalize, and sample with `torch.multinomial`. Optional top-k is the same idea with a fixed `k`.
- Each new token is appended to the input and fed back in (optionally using a KV cache); the function returns the decoded completion text.

---

## 29. Problem (experiment_log): Experiment logging (3 points)

**Deliverables:**
For your training and evaluation code, create experiment tracking infrastructure that allows you to track your experiments and loss curves with respect to gradient steps and wall-clock time.  
_Deliverable:_ Logging infrastructure code for your experiments and an experiment log (a document of all the things you tried) for the assignment problems below in this section.

**Answer:**

**Logging infrastructure** (implemented in `train.py`): one Weights-and-Biases run per experiment, with the full config logged (model size, tokenizer, lr, schedule, batch size, grad-clip, dtype, seed). Losses are logged with `wandb.define_metric("step")` and `wandb.define_metric("wall_time")` so every metric can be plotted against either gradient step or wall-clock time; we also log validation loss periodically, tokens/s and (derived) MFU. A plain console/stdout log is kept alongside wandb for reproducibility, and checkpoints record the iteration so runs can be resumed.

**Experiment log** (one row per run; the results column is filled in as the runs below are executed -- **TODO (results)**):

| id | problem | change vs. baseline | lr | batch | steps | val loss | notes |
| :-- | :-- | :-- | --: | --: | --: | --: | :-- |
| base | baseline | TinyStories, 17M model | TBD | TBD | TBD | TBD | reference run |
| lr-1 | 30 | lr sweep | {3e-4,1e-3,3e-3,1e-2} | 64 | same | TBD | pick best |
| bs-1 | 31 | batch-size sweep | re-tuned | {1,8,32,64,128,256} | same | TBD | re-tune lr |
| no-rm | 33 | remove RMSNorm | best/lower | 64 | same | TBD | stability study |
| post-n | 34 | post-norm | best | 64 | same | TBD | vs pre-norm |
| nope | 35 | remove RoPE | best | 64 | same | TBD | vs RoPE |
| silu | 36 | SiLU FFN | best | 64 | same | TBD | matched params |
| owt | 37 | OpenWebText, 32K | best | 64 | same | TBD | vs TinyStories |
| lb | 38 | leaderboard config | best | TBD | <=45 min | TBD | submit |

---

## 30. Problem (learning_rate): Tune the learning rate (2 B200 hrs) (3 points)

**Deliverables:**
The learning rate is one of the most important hyperparameters to tune. Taking the base model you’ve trained, answer the following questions:

**(a)** Perform a hyperparameter sweep over the learning rates and report the final losses (or note divergence if the optimizer diverges).  
_Deliverable:_ Learning curves associated with multiple learning rates. Explain your hyperparameter search strategy.  
_Deliverable:_ A model with validation loss (per-token) on _TinyStories_ of at most **1.45**.

**Answer:**

**Setup.** Base model on TinyStories (the handout's ~17M-parameter model: `d_model` ~. 512, 4-6 layers, `d_ff` ~. 8/3 d, context 1024, 10K tokenizer), AdamW (betas 0.9/0.999 or 0.9/0.95), cosine schedule with linear warmup, gradient clipping, fixed seed and step budget for every run.

**Strategy.** Coarse logarithmic sweep first, then refine around the best value: `lr in {3e-4, 1e-3, 3e-3, 1e-2}`, keeping everything else identical; rank by final validation loss and by area under the loss curve (a rate that is slightly worse at the end but much faster early is often preferable). Re-run the best 1-2 values with more steps/seed to confirm.

**Expected outcome (results are TODO until the runs are done).** Very small lr underfits (curve still descending at the step budget); around `1e-3..3e-3` the model should reach the target validation loss of at most **1.45** (per-token) on TinyStories; larger lr values (>= 1e-2) typically diverge or plateau at a worse loss even with warmup. *Learning curves and the final numeric best lr/val loss are TODO (pending implementation of `train.py` and the runs).*

---

## 31. Problem (batch_size_experiment): Batch size variations (1 B200 hr) (1 point)

**Deliverables:**
Vary your batch size all the way from 1 to the GPU memory limit. Try at least a few batch sizes in between, including typical sizes like 64 and 128.  
_Deliverable:_ Learning curves for runs with different batch sizes. The learning rates should be optimized again if necessary.  
_Deliverable:_ A few sentences discussing your findings on batch sizes and their impacts on training.

**Answer:**

**Setup.** Same TinyStories model/step budget as the learning-rate problem. Sweep `batch_size in {1, 8, 32, 64, 128, 256}` up to the memory limit, and for each batch size re-tune the learning rate (larger batches generally want a proportionally larger lr; a linear or square-root scaling rule is a good starting point).

**Discussion (results/curves TODO until the runs are done).** Extremely small batches (1-8) give noisy gradients and run many optimizer steps per token, which can help early progress per step but wastes GPU throughput per token; they also need a smaller lr. Typical sizes (64-128) usually give the best loss-vs-wall-clock trade-off. Large batches reduce gradient noise and use the hardware well, but beyond a point each doubling buys diminishing loss improvement and requires a higher lr to avoid slower per-token progress. The best choice here is the one with the lowest validation loss at a fixed wall-clock budget, not at a fixed step count.

---

## 32. Problem (generate): Generate text (1 point)

**Deliverables:**
Using your decoder and your trained checkpoint, report the text generated by your model. You may need to manipulate decoder parameters (temperature, top-p, etc.) to get fluent outputs.  
_Deliverable:_ Text dump of at least 256 tokens of text (or until the first `<|endoftext|>` token), and a brief comment on the fluency of this output and at least two factors which affect how good or bad this output is.

**Answer:**

**Setup.** Using the best TinyStories checkpoint and the top-p/temperature decoder (Problem 28), prompt the model with a story opening and sample at least 256 tokens (or until the first `<|endoftext|>`).

**Generated text:** TODO -- paste the dump after the model is trained (this requires first implementing `train.py`/`generate.py` and training a checkpoint).

**Fluency discussion (two+ factors).** (1) **Sampling parameters**: low temperature / low top-p makes the text more fluent but repetitive, while high temperature / high top-p increases diversity at the cost of coherence and can produce locally nonsensical or off-topic continuations. (2) **Training compute and data**: with only a small (~17M-parameter) model and a limited step budget the model has a small effective capacity and little world knowledge, so longer continuations drift; more steps, a larger model, or a larger/broader corpus would improve fluency. (3) The **tokenizer/context length** also bounds what the model can condition on.

---

## 33. Problem (layer_norm_ablation): Remove RMSNorm and train (0.5 B200 hrs) (1 point)

**Deliverables:**
Remove all of the RMSNorms from your Transformer and train. What happens at the previous optimal learning rate? Can you get stability by using a lower learning rate?  
_Deliverable:_ A learning curve for when you remove RMSNorms and train, as well as a learning curve for the best learning rate.  
_Deliverable:_ A few sentences of commentary on the impact of RMSNorm.

**Answer:**

**Setup.** Remove every `RMSNorm` (both in the block and the final norm) from the Transformer and retrain with everything else identical, first at the previously optimal learning rate and then at a reduced one.

**Expected result (curves TODO until the runs are done).** Without normalization the residual stream has no bound on its scale, so at the optimal lr the run should be unstable or diverge (loss spikes/NaNs) -- pre-norm Transformers rely on RMSNorm to keep activations well-conditioned. Lowering the learning rate substantially can restore short-term stability, but the best RMSNorm-free run should still converge more slowly and to a worse loss than the normalized baseline. The comparison establishes that RMSNorm is primarily a **stability/conditioning** device, not just a small compute saving.

---

## 34. Problem (pre_norm_ablation): Implement post-norm and train (0.5 B200 hrs) (1 point)

**Deliverables:**
Modify your pre-norm Transformer implementation into a post-norm one. Train with the post-norm model and see what happens.  
_Deliverable:_ A learning curve for a post-norm Transformer, compared to the pre-norm one.

**Answer:**

**Setup.** Change the block from pre-norm (`x + attn(ln(x))`) to post-norm (`ln(x + attn(x))`, and the same for the FFN) and retrain with the same budget, using a warmup schedule for the learning rate.

**Expected result (curves TODO until the runs are done).** Post-norm puts the normalization on the residual path, so gradients must flow through the norm and the identity path is no longer clean; this historically makes training less stable at depth and more sensitive to learning-rate warmup. With a careful warmup it can match pre-norm on a small model, but we expect the post-norm curve to be noisier and slightly worse at the same step/wall-clock budget, which is why modern decoder-only LMs use pre-norm.

---

## 35. Problem (no_pos_emb): Implement NoPE (0.5 B200 hrs) (1 point)

**Deliverables:**
Modify your Transformer implementation with RoPE to remove the position embedding information entirely, and see what happens.  
_Deliverable:_ A learning curve comparing the performance of RoPE and NoPE.

**Answer:**

**Setup.** Remove RoPE entirely (no positional information; keys and queries are untouched before attention) and train with the same budget, then compare loss curves against the RoPE baseline.

**Expected result (curves TODO until the runs are done).** A decoder-only causal model still contains some positional signal in the causal mask (token i can only attend to j <= i), so NoPE can still train and is surprisingly competitive at short contexts -- but without explicit relative/absolute positions it should be somewhat worse, and the gap should grow as the context length increases (reordering information is lost, and long-range ordering becomes ambiguous). The RoPE curve is expected to be equal or better, with a larger advantage at longer sequences.

---

## 36. Problem (swiglu_ablation): SwiGLU vs. SiLU (0.5 B200 hrs) (1 point)

**Deliverables:** > _Deliverable:_ A learning curve comparing the performance of SwiGLU and SiLU feed-forward networks, with approximately matched parameter counts.  
_Deliverable:_ A few sentences discussing your findings.

**Answer:**

**Setup.** Compare the SwiGLU FFN (`W2(silu(W1 x) * W3 x)`, three matrices, `d_ff ~= 8/3 d`) against a plain SiLU FFN with no gate (`W2(silu(W1 x))`, two matrices). Match the **parameter counts** by choosing `d_ff` of the ungated FFN so that `2 d d_ff` is as close as possible to the SwiGLU `3 d d_ff` (roughly `d_ff^plain ~= (3/2) d_ff^swiglu`), and train both with the same budget.

**Expected result (curves TODO until the runs are done).** At matched parameters, the gated SwiGLU is expected to reach a slightly lower loss, which is why it is the standard modern FFN; the gating lets the network multiplicatively modulate features, and the extra matrix (with a smaller `d_ff` per parameter) apparently helps more than spending the same parameters on a wider ungated FFN. The gap is usually small but consistent.

---

## 37. Problem (main_experiment): Experiment on OWT (2 B200 hrs) (2 points)

**Deliverables:**
Train your language model on _OpenWebText_ with the same model architecture and total training iterations as _TinyStories_. How well does this model do?  
_Deliverable:_ A learning curve of your language model on _OpenWebText_. Describe the difference in losses from _TinyStories_ – how should we interpret these losses?  
_Deliverable:_ Generated text from _OpenWebText_ LM, in the same format as the _TinyStories_ outputs. How is the fluency of this text? Why is the output quality worse even though we have the same model and compute budget as _TinyStories_?

**Answer:**

**Setup.** Train exactly the same architecture and total number of training iterations as the TinyStories run, but on OpenWebText and with the 32K OpenWebText tokenizer (this is the 2-B200-hr problem; **run pending -- no B200 budget and `train.py` is not committed yet**).

**Expected result (curve TODO).** OpenWebText is far larger and more diverse (web text, many topics/registers/languages), so its per-token cross-entropy should be **higher** than TinyStories even at the same loss/step budget: the entropy of the distribution is higher and the model cannot memorize/cover it as easily. The generated text should also be **less fluent and less coherent** than the TinyStories output at equal compute, because the same model capacity and step budget now have to cover a much broader distribution (and mostly English web text rather than simple narrative), so the model is effectively under-trained relative to task difficulty. Loss numbers alone must be interpreted per-dataset: a higher OWT loss does not mean the model is worse, only that the prediction problem is harder.

---

## 38. Problem (leaderboard): Leaderboard (10 B200 hrs) (6 points)

**Deliverables:**
You will train a model under the leaderboard rules above with the goal of minimizing the validation loss of your language model within 0.75 B200-hours.  
_Deliverable:_ The final validation loss that was recorded, an associated learning curve that clearly shows a wall-clock-time x-axis that is less than 45 minutes, and a description of what you did. We expect a leaderboard submission to beat at least the naive baseline of a 5.0 loss. Submit to the leaderboard here: `github.com/stanford-cs336/assignment1-basics-leaderboard`.

**Answer:**

**Setup/strategy.** Follow the leaderboard rules and train within 0.75 B200-hours (45 minutes wall-clock). Practical plan: use the OpenWebText 32K tokenizer, a small-but-strong config (e.g. ~30-60M parameters, context 512-1024, ~4-8 layers) sized to fit the time budget; use the best lr from the sweep with warmup + cosine, gradient clipping, and bf16; use a moderately large batch to maximize MFU, and stop when the 45-minute wall-clock limit is reached. Log the curve against **wall-clock time** (not steps) so the < 45-minute requirement is visible, and record the final validation loss.

**Result:** TODO -- **not run here** (no B200/budget in this environment); this section will contain the submitted final validation loss, the wall-clock learning curve, and a description of the exact recipe once the run is executed. The submission must beat the naive baseline of 5.0 loss.

---
