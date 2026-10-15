"""Causal attention utilities for cube operators."""


def causal_factor(causal: bool, q_len: int, kv_len: int) -> float:
    """Causal attention 的实际计算量 / full attention 计算量。

    考虑 prefix cache 的场景（cache_len > 0）：
      causal: query i 只 attend 到 position <= cache_len + i
      总和 = sum_{i=0}^{q_len-1} (cache_len + i + 1)
           = q_len * cache_len + q_len * (q_len + 1) / 2

    当 cache_len == 0 且 q_len > 1（prefill 无 prefix cache）时：
      causal_elts = q_len * (q_len + 1) / 2，约等于 q_len^2 / 2

    当 q_len == 1（decode）时：
      causal_elts = cache_len + 1 = kv_len，ratio=1.0

    当 kv_len < q_len（如 sliding window attention）时：
      cache_len 为负，退化为全量计算，返回 1.0。

    当 causal=False 时恒为 1.0。
    """
    if not causal or kv_len < q_len:
        return 1.0
    cache_len = kv_len - q_len
    causal_elts = q_len * cache_len + q_len * (q_len + 1) / 2
    full_elts = q_len * kv_len
    return causal_elts / full_elts
