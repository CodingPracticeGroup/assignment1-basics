"""CS336 Assignment 1 Transformer LM 训练循环。

配合 scripts/tokenize_dataset.py 产出的 uint16 .bin（用 np.memmap 省内存读取）。

handout §7.2.1 的 baseline（**从这里开始 tune**）：
    vocab=10000, context=256, d_model=512, d_ff=1344, RoPE θ=10000,
    4 layers / 16 heads, 总 token ≈ 327,680,000（= batch × steps × context）。
    其余（learning rate / warmup / AdamW 的 β/ε / weight decay）需要自己 tune。

用法示例（TinyStories baseline）：
  uv run python scripts/train.py \
      --train-bin artifacts/tinystories_10k_stream/train.bin \
      --val-bin   artifacts/tinystories_10k_stream/valid.bin \
      --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
      --num-heads 16 --d-ff 1344 --rope-theta 10000 \
      --batch-size 64 --max-steps 20000 \
      --lr 1e-3 --warmup-iters 200 \
      --checkpoint artifacts/runs/tinystories_base/ckpt.pt \
      --tensorboard artifacts/runs/tinystories_base/tb
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from cs336_basics.model.transformer import BasicsTransformerLM
from cs336_basics.training.checkpointer import load_checkpoint, save_checkpoint
from cs336_basics.training.clipping import run_gradient_clipping
from cs336_basics.training.dataloader import run_get_batch
from cs336_basics.training.metrics import perplexity_from_loss
from cs336_basics.training.optimizers import AdamW, cross_entropy
from cs336_basics.training.schedulers import run_get_lr_cosine_schedule


def build_argparser() -> argparse.ArgumentParser:
    # 默认值即 handout §7.2.1 的 baseline（vocab=10000, ctx=256, d_model=512,
    # 4 layers / 16 heads, d_ff=1344, θ=10000；batch 64 × steps 20000 × ctx 256
    # = 327,680,000 tokens）。lr / warmup / AdamW / weight decay 需自己 tune。
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--train-bin", required=True, help="token id 数组：原始 uint16 .bin（np.memmap）或 .npy（np.load mmap）")
    ap.add_argument("--val-bin", default=None)
    # 模型
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--context-length", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=16, help="baseline: 4 层 16 头")
    ap.add_argument("--d-ff", type=int, default=1344)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    # 优化器
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-iters", type=int, default=200)
    ap.add_argument("--cosine-cycle-iters", type=int, default=20000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # 循环
    ap.add_argument("--batch-size", type=int, default=64, help="baseline: 64*20000*256 = 327.68M tokens")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--checkpoint", default=None, help="checkpoint 输出路径")
    ap.add_argument("--resume", default=None, help="从该 checkpoint 恢复")
    ap.add_argument("--tensorboard", default=None, help="TensorBoard logdir（本地可视化、无需账号）")
    ap.add_argument("--sdpa", action="store_true", help="用 PyTorch 融合 SDPA 替代手写注意力（更快、数学等价；训练不再走我们自己的 attention 实现）")
    # 运行时
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="cs336-assignment1")
    ap.add_argument("--wandb-run-name", default=None)
    return ap


def _open_dataset(path: str):
    """内存映射打开分词后的 token id 数组（handout P25/P27 推荐 np.memmap）。

    - .npy  -> np.load(..., mmap_mode="r")（若用 np.save 保存）
    - 其他  -> np.memmap(..., dtype=uint16)（原始 uint16 二进制，tokenize_dataset.py 的产物）
    """
    if str(path).endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.uint16, mode="r")


def _amp_settings(device: str, dtype: str):
    amp_device = device.split(":")[0]
    amp_dtype = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    return amp_device, amp_dtype


def evaluate(model, dataset, args, device, amp_device, amp_dtype) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(args.val_batches):
            x, y = run_get_batch(dataset, args.batch_size, args.context_length, device)
            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    amp_device, amp_dtype = _amp_settings(device, args.dtype)

    if args.sdpa:
        # 用 torch 的融合 SDPA（flash / mem-efficient）替换当前 notation 的
        # 手写 scaled_dot_product_attention；因果 mask 交给 is_causal。
        import importlib

        import torch.nn.functional as F

        from cs336_basics import notation

        attn_mod = importlib.import_module(f"cs336_basics.notation.{notation.ACTIVE}.attention")

        def _fused_sdpa(Q, K, V, mask=None):
            return F.scaled_dot_product_attention(Q, K, V, is_causal=(mask is not None))

        attn_mod.scaled_dot_product_attention = _fused_sdpa
        print("using fused torch SDPA for attention", flush=True)

    train_data = _open_dataset(args.train_bin)
    val_data = _open_dataset(args.val_bin) if args.val_bin else None

    model = BasicsTransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M | device={device} dtype={args.dtype}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed from {args.resume} at step {start_step}")

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=vars(args)
        )
        # 每个指标既可按「梯度步」(step) 也可按「墙钟时间」(wall_time) 作图
        wandb_run.define_metric("step")
        wandb_run.define_metric("wall_time")
        wandb_run.define_metric("*", step_metric="step")

    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(args.tensorboard)
        except ImportError:
            print(
                "warning: --tensorboard 需要 tensorboard 包，已跳过"
                "（安装：.venv/bin/pip install tensorboard 或 uv add tensorboard）",
                flush=True,
            )

    model.train()
    t0 = time.time()
    last_log_time = t0
    tokens_since_log = 0
    for step in range(start_step, args.max_steps):
        lr = run_get_lr_cosine_schedule(
            step, args.lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = run_get_batch(train_data, args.batch_size, args.context_length, device)
        tokens_since_log += x.numel()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        run_gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0:
            now = time.time()
            elapsed = now - t0
            tokens_per_sec = tokens_since_log / max(now - last_log_time, 1e-9)
            train_loss = loss.item()
            print(
                f"step {step:6d} | lr {lr:.3e} | loss {train_loss:.4f} | "
                f"{elapsed:.1f}s | {tokens_per_sec / 1e3:.1f}k tok/s",
                flush=True,
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "train/loss": train_loss,
                        "train/perplexity": perplexity_from_loss(train_loss),
                        "lr": lr,
                        "tokens_per_sec": tokens_per_sec,
                        "wall_time": elapsed,
                        "step": step,
                    },
                    step=step,
                )
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", train_loss, step)
                tb_writer.add_scalar("train/perplexity", perplexity_from_loss(train_loss), step)
                tb_writer.add_scalar("train/learning_rate", lr, step)
                tb_writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)
                # 想按墙钟时间看曲线时，用 TensorBoard UI 左上角的 x 轴切到 "Wall"
            last_log_time = now
            tokens_since_log = 0

        if val_data is not None and args.val_every > 0 and (step + 1) % args.val_every == 0:
            val_loss = evaluate(model, val_data, args, device, amp_device, amp_dtype)
            val_ppl = perplexity_from_loss(val_loss)
            print(f"step {step:6d} | val loss {val_loss:.4f} | val ppl {val_ppl:.2f}", flush=True)
            if tb_writer is not None:
                tb_writer.add_scalar("val/loss", val_loss, step)
                tb_writer.add_scalar("val/perplexity", val_ppl, step)
            if wandb_run:
                wandb_run.log(
                    {
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                        "wall_time": time.time() - t0,
                        "step": step,
                    },
                    step=step,
                )

        if args.checkpoint and args.save_every > 0 and (step + 1) % args.save_every == 0:
            os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
            save_checkpoint(model, optimizer, step + 1, args.checkpoint)

    if args.checkpoint:
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        save_checkpoint(model, optimizer, args.max_steps, args.checkpoint)
        print(f"final checkpoint -> {args.checkpoint}")
    if tb_writer is not None:
        tb_writer.close()
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
