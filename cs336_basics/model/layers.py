from __future__ import annotations

import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        手写无 Bias 线性变换层。
        权重矩阵 W 的形状为 (out_features, in_features)。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 定义权重参数 W
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        
        # 权重初始化：截断正态分布
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入 x 应用线性变换：y = x @ W^T
        """
        return x @ self.weight.t()


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        手写嵌入查找层。
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 定义嵌入参数矩阵
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        
        # 初始权重设为 N(0, 1) 的截断正态分布
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        根据给定的 token_ids 查表返回词嵌入嵌入向量
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        均方根层归一化 (RMSNorm)。
        """
        super().__init__()
        self.eps = eps
        # 初始化 gain 参数为全 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算均方根层归一化，支持上投 float32 运算。
        """
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        
        # 计算均方根
        variance = torch.mean(x_f32 ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt(variance + self.eps)
        
        # 归一化并恢复原数据精度后，乘以 gain (self.weight)
        normalized = (x_f32 / rms).to(in_dtype)
        return self.weight * normalized


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Swish) 激活函数：f(x) = x * sigmoid(x)
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        基于 Gated Linear Unit (GLU) 与 SiLU 激活实现的 SwiGLU FFN 层。
        """
        super().__init__()
        # 定义 W1, W2, W3 线性变换
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN_SwiGLU(x) = W2(SiLU(W1 x) * W3 x)
        """
        # 门控分支的激活运算 SiLU(W1 x)
        gate = silu(self.w1(x))
        # 乘以特征分支的输入并由 W2 线性输出投影
        return self.w2(gate * self.w3(x))
