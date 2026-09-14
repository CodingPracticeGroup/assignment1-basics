"""CS336 Assignment 1 Transformer LM 训练循环。

配合 scripts/tokenize_dataset.py 产出的 uint16 .bin（用 np.memmap 省内存读取）。

用法示例：
  uv run python scripts/train.py \
      --train-bin artifacts/tinystories_10k_stream/train.bin \
      --val-bin   artifacts/tinystories_10k_stream/valid.bin \
      --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 --num-heads 16 \
      --d-ff 1344 --batch-size 32 --max-steps 20000 --lr 1e-3 --warmup-iters 200 \
      --cosine-cycle-iters 20000 --checkpoint artifacts/runs/tinystories/ckpt.pt
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
    ap = argparse.ArgumentParser()
    # 数据
    ap.add_argument("--train-bin", required=True, help="uint16 token id 二进制（np.memmap）")
    ap.add_argument("--val-bin", default=None)
    # 模型
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--context-length", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=8)
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
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--checkpoint", default=None, help="checkpoint 输出路径")
    ap.add_argument("--resume", default=None, help="从该 checkpoint 恢复")
    ap.add_argument("--tensorboard", default=None, help="TensorBoard logdir（本地可视化、无需账号）")
    # 运行时
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="cs336-assignment1")
    ap.add_argument("--wandb-run-name", default=None)
    return ap


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

    train_data = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_bin, dtype=np.uint16, mode="r") if args.val_bin else None

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
