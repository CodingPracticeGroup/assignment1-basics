"""把原始文本语料分词成 uint16 token-id 二进制流，供 np.memmap 训练。

用法示例：
  uv run python scripts/tokenize_dataset.py \
      --input data/TinyStoriesV2-GPT4-train.txt \
      --tokenizer-dir artifacts/tinystories_10k_stream \
      --output artifacts/tinystories_10k_stream/train.bin
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

from cs336_basics.bpe_tokenizer.base import Tokenizer


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tokenizer(args) -> Tokenizer:
    if args.tokenizer_dir:
        vocab = _load_pickle(os.path.join(args.tokenizer_dir, "vocab.pkl"))
        merges = _load_pickle(os.path.join(args.tokenizer_dir, "merges.pkl"))
    elif args.vocab_json and args.merges_txt:
        return Tokenizer.from_files(args.vocab_json, args.merges_txt, [args.special_token])
    else:
        vocab = _load_pickle(args.vocab)
        merges = _load_pickle(args.merges)
    return Tokenizer(vocab, merges, [args.special_token])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="原始文本文件")
    ap.add_argument("--output", required=True, help="输出 .bin（uint16 token id 流）")
    ap.add_argument("--tokenizer-dir", help="含 vocab.pkl / merges.pkl 的目录")
    ap.add_argument("--vocab", help="vocab.pkl 路径")
    ap.add_argument("--merges", help="merges.pkl 路径")
    ap.add_argument("--vocab-json", help="GPT-2 格式 vocab.json")
    ap.add_argument("--merges-txt", help="GPT-2 格式 merges.txt")
    ap.add_argument("--special-token", default="<|endoftext|>")
    ap.add_argument("--chunk-bytes", type=int, default=1 << 26)
    ap.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    args = ap.parse_args()

    tokenizer = load_tokenizer(args)
    largest_id = max(tokenizer.vocab)
    dtype = np.uint16 if (args.dtype == "uint16" and largest_id < 65536) else np.uint32

    total = 0
    special = args.special_token
    with open(args.input, encoding="utf-8", errors="ignore") as fin, open(args.output, "wb") as fout:
        buf = ""
        while True:
            chunk = fin.read(args.chunk_bytes)
            if not chunk:
                break
            buf += chunk
            # 只处理「最后一个特殊 token 之前」的内容，尾巴留到下一轮，避免把文档切成两半
            cut = buf.rfind(special)
            if cut == -1:
                continue
            process, buf = buf[: cut + len(special)], buf[cut + len(special):]
            arr = np.asarray(tokenizer.encode(process), dtype=dtype)
            fout.write(arr.tobytes())
            total += arr.size
        if buf.strip():
            arr = np.asarray(tokenizer.encode(buf), dtype=dtype)
            fout.write(arr.tobytes())
            total += arr.size

    print(f"wrote {total} tokens ({np.dtype(dtype).name}) to {args.output}")


if __name__ == "__main__":
    main()
