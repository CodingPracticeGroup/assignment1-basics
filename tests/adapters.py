from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.
    """
    from cs336_basics.model.layers import Linear
    layer = Linear(d_in, d_out)
    layer.weight.data = weights
    return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.
    """
    from cs336_basics.model.layers import Embedding
    layer = Embedding(vocab_size, d_model)
    layer.weight.data = weights
    return layer(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.
    """
    from cs336_basics.model.layers import SwiGLU
    layer = SwiGLU(d_model, d_ff)
    layer.w1.weight.data = w1_weight
    layer.w2.weight.data = w2_weight
    layer.w3.weight.data = w3_weight
    return layer(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.
    """
    from cs336_basics.model.attention import scaled_dot_product_attention
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    multi-head attention, return the output of an optimized batched implementation.
    Does not use RoPE.
    """
    from cs336_basics.model.attention import CausalMultiHeadSelfAttention
    layer = CausalMultiHeadSelfAttention(d_model, num_heads)
    layer.q_proj.weight.data = q_proj_weight
    layer.k_proj.weight.data = k_proj_weight
    layer.v_proj.weight.data = v_proj_weight
    layer.output_proj.weight.data = o_proj_weight
    return layer(in_features, rope=None, token_positions=None)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Causal Multihead Attention with RoPE support.
    """
    from cs336_basics.model.attention import CausalMultiHeadSelfAttention, RotaryPositionalEmbedding
    layer = CausalMultiHeadSelfAttention(d_model, num_heads)
    layer.q_proj.weight.data = q_proj_weight
    layer.k_proj.weight.data = k_proj_weight
    layer.v_proj.weight.data = v_proj_weight
    layer.output_proj.weight.data = o_proj_weight
    
    d_k = d_model // num_heads
    rope_layer = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return layer(in_features, rope=rope_layer, token_positions=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE positional rotation.
    """
    from cs336_basics.model.attention import RotaryPositionalEmbedding
    rope_layer = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rope_layer(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Transformer Block evaluation with weight loading.
    """
    from cs336_basics.model.transformer import TransformerBlock
    from cs336_basics.model.attention import RotaryPositionalEmbedding
    layer = TransformerBlock(d_model, num_heads, d_ff)
    
    layer.attn.q_proj.weight.data = weights["attn.q_proj.weight"]
    layer.attn.k_proj.weight.data = weights["attn.k_proj.weight"]
    layer.attn.v_proj.weight.data = weights["attn.v_proj.weight"]
    layer.attn.output_proj.weight.data = weights["attn.output_proj.weight"]
    layer.ln1.weight.data = weights["ln1.weight"]
    layer.ffn.w1.weight.data = weights["ffn.w1.weight"]
    layer.ffn.w2.weight.data = weights["ffn.w2.weight"]
    layer.ffn.w3.weight.data = weights["ffn.w3.weight"]
    layer.ln2.weight.data = weights["ln2.weight"]
    
    d_k = d_model // num_heads
    rope_layer = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return layer(in_features, rope=rope_layer)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """
    Total Transformer LM evaluation with state dict reconstruction.
    """
    from cs336_basics.model.transformer import BasicsTransformerLM
    layer = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    
    state_dict = {}
    state_dict["token_embeddings.weight"] = weights["token_embeddings.weight"]
    state_dict["ln_final.weight"] = weights["ln_final.weight"]
    state_dict["lm_head.weight"] = weights["lm_head.weight"]
    
    for l in range(num_layers):
        state_dict[f"layers.{l}.attn.q_proj.weight"] = weights[f"layers.{l}.attn.q_proj.weight"]
        state_dict[f"layers.{l}.attn.k_proj.weight"] = weights[f"layers.{l}.attn.k_proj.weight"]
        state_dict[f"layers.{l}.attn.v_proj.weight"] = weights[f"layers.{l}.attn.v_proj.weight"]
        state_dict[f"layers.{l}.attn.output_proj.weight"] = weights[f"layers.{l}.attn.output_proj.weight"]
        state_dict[f"layers.{l}.ln1.weight"] = weights[f"layers.{l}.ln1.weight"]
        state_dict[f"layers.{l}.ffn.w1.weight"] = weights[f"layers.{l}.ffn.w1.weight"]
        state_dict[f"layers.{l}.ffn.w2.weight"] = weights[f"layers.{l}.ffn.w2.weight"]
        state_dict[f"layers.{l}.ffn.w3.weight"] = weights[f"layers.{l}.ffn.w3.weight"]
        state_dict[f"layers.{l}.ln2.weight"] = weights[f"layers.{l}.ln2.weight"]
        
    layer.load_state_dict(state_dict)
    return layer(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    RMSNorm evaluation.
    """
    from cs336_basics.model.layers import RMSNorm
    layer = RMSNorm(d_model, eps)
    layer.weight.data = weights
    return layer(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """
    SiLU (Swish) evaluation.
    """
    from cs336_basics.model.layers import silu
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dataset batch sampling loader.
    """
    from cs336_basics.training.dataloader import run_get_batch as get_batch
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Numerical stable Softmax.
    """
    from cs336_basics.model.attention import softmax
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    Log-sum-exp Cross Entropy.
    """
    from cs336_basics.training.optimizers import cross_entropy
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Norm-based Gradient Clipping.
    """
    from cs336_basics.training.clipping import run_gradient_clipping as clip_grad
    return clip_grad(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Return custom stateful AdamW Optimizer class.
    """
    from cs336_basics.training.optimizers import AdamW
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Cosine learning rate with warmup.
    """
    from cs336_basics.training.schedulers import run_get_lr_cosine_schedule as lr_schedule
    return lr_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Save state checkpoint.
    """
    from cs336_basics.training.checkpointer import run_save_checkpoint as save_cp
    return save_cp(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load state checkpoint.
    """
    from cs336_basics.training.checkpointer import run_load_checkpoint as load_cp
    return load_cp(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """
    Get custom Tokenizer instance.
    """
    from cs336_basics.bpe_tokenizer.base import Tokenizer
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    BPE training.
    """
    from cs336_basics.bpe_tokenizer.trainer import run_train_bpe as train_bpe
    return train_bpe(input_path, vocab_size, special_tokens, **kwargs)
