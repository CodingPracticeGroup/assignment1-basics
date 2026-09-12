# 作业 1 的两套实现：不用 / 尽量用 爱因斯坦 notation

作业 1 现在同时包含**两份数值等价、可独立运行**的实现，通过环境变量选择：

1. **`no_einstein`（版本 1）**：不使用爱因斯坦 notation。矩阵乘用 `@` / `torch.matmul`，
   维度整理用 `reshape` / `view` / `transpose` / `permute`，归一化用 `torch.mean` /
   `torch.sum`，取值用高级索引。
2. **`einstein`（版本 2）**：使用 `einx`。缩合用 `einx.dot`，结构变换用 `einx.id`，
   归一化与索引用 `einx.softmax` / `einx.logsumexp` / `einx.get_at` / `einx.mean` /
   `einx.multiply`。

两份实现共享同一套公开 API，官方测试套件（`tests/`）对两者都通过。

> 概念/原理（爱因斯坦 notation、einops / einx / einsum 的选型）不在本作业 repo，
> 见概念仓库 `Stanford-CS336/Spring2026/assignments/assignment1/einsum_and_elegant_matrix_ops.md`。

## 目录结构

```
cs336_basics/
├── notation/
│   ├── __init__.py              # CS336_NOTATION 选择器
│   ├── no_einstein/             # 版本 1：不用爱因斯坦 notation
│   │   ├── layers.py            # Linear / Embedding / RMSNorm / silu / SwiGLU
│   │   ├── attention.py         # softmax / SDPA / RoPE / Causal MHA
│   │   ├── transformer.py       # TransformerBlock / BasicsTransformerLM
│   │   └── cross_entropy.py
│   └── einstein/                # 版本 2：统一 einx API
│       ├── layers.py
│       ├── attention.py
│       ├── transformer.py
│       └── cross_entropy.py
├── model/
│   ├── layers.py                # 按 CS336_NOTATION 转发到对应版本
│   ├── attention.py             # （tests/adapters.py 无需任何改动）
│   └── transformer.py
└── training/
    └── optimizers.py            # AdamW 两版共用 + 转发 cross_entropy
```

**不涉及张量 notation 的模块**（BPE 分词器、dataloader、scheduler、gradient clipping、
checkpointer）保持单一实现，两套版本共用。

## 如何选择版本

`CS336_NOTATION` 环境变量必须在 **首次 import `cs336_basics` 之前** 设置：

```sh
# 版本 1：不用爱因斯坦 notation
CS336_NOTATION=no_einstein uv run pytest

# 版本 2：使用 einx（默认值）
CS336_NOTATION=einstein uv run pytest
```

别名：`plain` / `explicit` / `none` -> `no_einstein`；
`einsum` / `einops` / `elegant` -> `einstein`。

也可以直接、显式地 import 某一版（完全绕过选择器）：

```python
from cs336_basics.notation.no_einstein.attention import CausalMultiHeadSelfAttention
from cs336_basics.notation.einstein.attention import CausalMultiHeadSelfAttention
```

## 两版代码逐操作对照

| 操作 | `no_einstein`（版本 1） | `einstein`（版本 2，einx） |
| :-- | :-- | :-- |
| Linear | `x @ self.weight.t()` | `einx.dot("... d_in, d_out d_in -> ... d_out", x, W)` |
| Embedding | `self.weight[token_ids]` | `einx.get_at("[v] d, ... -> ... d", W, token_ids)` |
| RMSNorm 归一化 | `torch.mean(x**2, dim=-1, keepdim=True)` | `einx.mean("... ([d])", x**2)` |
| RMSNorm gain | `self.weight * normalized` | `einx.multiply("... d, d -> ... d", normalized, w)` |
| SwiGLU 门控 | `gate * self.w3(x)` | `einx.multiply("... f, ... f -> ... f", gate, w3x)` |
| Softmax | `exp(x - max) / sum(exp)` | `einx.softmax("... [d] ...", x)` |
| Attention scores | `Q @ K.transpose(-2, -1)` | `einx.dot("... q d, ... k d -> ... q k", Q, K)` |
| Attention context | `probs @ V` | `einx.dot("... q k, ... k v -> ... q v", probs, V)` |
| MHA 分头 | `q.reshape(b, s, h, d_k).transpose(1, 2)` | `einx.id("b s (h d) -> b h s d", q, h=num_heads)` |
| MHA 合头 | `out.transpose(1, 2).contiguous().reshape(b, s, d)` | `einx.id("b h s d -> b s (h d)", out)` |
| RoPE 配对 | `x[..., 0::2]` / `x[..., 1::2]` + `torch.empty_like` | `einx.id("... s (d pair) -> ... s d pair", x, pair=2)` |
| RoPE 旋转 | 4 行逐元素 `cos` / `sin` 组合 | `einx.dot("... p i j, ... p j -> ... p i", rotation, x_pairs)` |
| Cross-entropy target | `stable_inputs[torch.arange(n), targets]` | `einx.get_at("n [v], n -> n", x, t)` |
| Cross-entropy log-sum-exp | `log(sum(exp(x - max)))` | `einx.logsumexp("n [v]", x)` |

## 实现说明

* **纯 einx**：版本 2 只 import `einx`，不依赖 `einops`，也不用 `torch.einsum`
  （`einx.dot` / `einx.id` 即 einsum / rearrange 对应物）。
* **rank 对齐**：`einx.dot` **不会**跨秩隐式广播，所以 RoPE 里用
  `_broadcast_leading` 配合 `einx.id` 把位置角度的前置维度显式对齐到 q/k。
* **数值等价**：两版都对 Softmax / log-sum-exp 先减最大值，输出在官方容差内一致，
  都能通过快照测试。
* **转发层**：`cs336_basics.model.layers` 等只是 import 时的转发，选择在进程启动时
  一次性完成，不会引入运行时开销。
* **默认版本**：如果未设置 `CS336_NOTATION`，默认使用 `einstein`。
