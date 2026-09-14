"""从训练好的 checkpoint 自回归生成文本（temperature + top-p / nucleus 采样）。

用法示例：
  uv run python scripts/generate.py \
      --checkpoint artifacts/runs/tinystories/ckpt.pt \
      --tokenizer-dir artifacts/tinystories_10k_stream \
      --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
      --prompt "Once upon a time" --max-new-tokens 256 --temperature 0.8 --top-p 0.9
"""

from __future__ import annotations

import argparse
import os
import pickle

import torch

from cs336_basics.bpe_tokenizer.base import Tokenizer
from cs336_basics.model.transformer import BasicsTransformerLM


def load_tokenizer(args) -> Tokenizer:
    if args.tokenizer_dir:
        with open(os.path.join(args.tokenizer_dir, "vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
        with open(os.path.join(args.tokenizer_dir, "merges.pkl"), "rb") as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, [args.special_token])
    return Tokenizer.from_files(args.vocab_json, args.merges_txt, [args.special_token])


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus 采样：只保留累积概率首次达到 top_p 的最短前缀，其余置 -inf。"""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = (cumulative - probs) > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.empty_like(logits)
    filtered.scatter_(0, sorted_idx, sorted_logits)
    return filtered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    # 模型结构（须与训练一致）
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--context-length", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=1344)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    # tokenizer
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--vocab-json", default=None)
    ap.add_argument("--merges-txt", default=None)
    ap.add_argument("--special-token", default="<|endoftext|>")
    # 解码超参
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    tokenizer = load_tokenizer(args)

    model = BasicsTransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("prompt 编码为空")
    eot_id = tokenizer.byte_to_id.get(args.special_token.encode("utf-8"))

    ids = list(prompt_ids)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            x = torch.tensor([ids[-args.context_length:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            logits = logits / max(args.temperature, 1e-8)
            if args.top_p < 1.0:
                logits = top_p_filter(logits, args.top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            ids.append(next_id)
            if eot_id is not None and next_id == eot_id:
                break
            generated.append(next_id)

    print("=== prompt ===")
    print(args.prompt)
    print("=== completion ===")
    print(tokenizer.decode(generated))
    print("=== full ===")
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
