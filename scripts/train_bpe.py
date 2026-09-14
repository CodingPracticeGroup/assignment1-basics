from __future__ import annotations

import argparse
import json
import os
import pickle
import threading
import time

import psutil

from cs336_basics.bpe_tokenizer.base import _gpt2_bytes_to_unicode
from cs336_basics.bpe_tokenizer.trainer import run_train_bpe


def _peak_tree_rss_mb(pid: int, stop: threading.Event, out: list[int]) -> None:
    peak = 0
    start = time.time()
    last_log = start
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        out.append(0)
        return
    while not stop.is_set():
        try:
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, total)
            now = time.time()
            if now - last_log >= 30.0:
                last_log = now
                print(
                    f"  [heartbeat] elapsed={now - start:.0f}s tree_rss={total / 1024 ** 3:.2f} GiB",
                    flush=True,
                )
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.2)
    out.append(peak)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--special-token", default="<|endoftext|>")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stop = threading.Event()
    peak_mb: list[int] = []
    mon = threading.Thread(target=_peak_tree_rss_mb, args=(os.getpid(), stop, peak_mb), daemon=True)
    mon.start()

    t0 = time.time()
    vocab, merges = run_train_bpe(args.input, args.vocab_size, [args.special_token])
    elapsed = time.time() - t0

    stop.set()
    mon.join(timeout=2)
    peak = (peak_mb[0] if peak_mb else 0) / (1024 ** 3)  # bytes -> GiB

    with open(os.path.join(args.out_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(args.out_dir, "merges.pkl"), "wb") as f:
        pickle.dump(merges, f)

    # 同时导出 GPT-2 文本格式，便于 Tokenizer.from_files / 其他工具加载
    byte_to_unicode = _gpt2_bytes_to_unicode()
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"".join(byte_to_unicode[b] for b in token): tid for tid, token in vocab.items()},
            f,
            ensure_ascii=False,
        )
    with open(os.path.join(args.out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for first, second in merges:
            f.write(
                "".join(byte_to_unicode[b] for b in first)
                + " "
                + "".join(byte_to_unicode[b] for b in second)
                + "\n"
            )

    print(f"input={args.input}")
    print(f"vocab_size_target={args.vocab_size} actual_vocab={len(vocab)} merges={len(merges)}")
    print(f"elapsed={elapsed:.1f}s  peak_tree_rss={peak:.2f} GiB")
    longest = sorted(vocab.items(), key=lambda kv: len(kv[1]), reverse=True)[:8]
    print("longest tokens:")
    for tid, tok in longest:
        print(f"  len={len(tok):4d} id={tid} {tok!r}")


if __name__ == "__main__":
    main()
