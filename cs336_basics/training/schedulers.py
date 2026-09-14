from __future__ import annotations

import math


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    计算基于线性 Warmup 和 Cosine 退火算法的自适应学习率。
    """
    # 1. Warm-up 阶段：线性增加学习率
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    # 2. Cosine 退火降维阶段
    if warmup_iters <= it <= cosine_cycle_iters:
        # 如果 warmup_iters == cosine_cycle_iters，退化至常数极值
        if cosine_cycle_iters == warmup_iters:
            return min_learning_rate
        decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(decay_ratio * math.pi))
        return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)

    # 3. 超出退火循环范围：保持最低学习率恒定不变
    return min_learning_rate
