"""Readable causal self-attention shared by CPU, MPS, and later CUDA."""

import math

import torch


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Return attention output with shape [batch, heads, tokens, head_dim].

    Contract for this first implementation:
      * q, k, v have the same shape [B, H, T, D]; all dimensions are positive.
      * Inputs are finite float32 tensors on the same device; views may be
        non-contiguous. CPU and MPS are tested; CUDA validation comes later.
      * Position i may attend only to positions j <= i, including itself.
      * No dropout, padding, grouped-query heads, or KV cache yet.
      * Leave input tensors unchanged.

    This intentionally materializes the full score matrix for clarity. Later
    attention kernels will avoid this intermediate while keeping the interface.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same four-dimensional [B, H, T, D] shape")
    if any(size == 0 for size in q.shape):
        raise ValueError("All attention dimensions must be positive")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")
    if any(tensor.dtype != torch.float32 for tensor in (q, k, v)):
        raise ValueError("This learning implementation supports float32 inputs")

    # [B,H,T,D] @ [B,H,D,T] -> [B,H,T,T]. Scale before softmax.
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    tokens = q.shape[-2]
    positions = torch.arange(tokens, device=q.device)
    # Row i is a query; column j is a key. Block keys with j > i.
    future = positions[None, :] > positions[:, None]
    scores = scores.masked_fill(future, -torch.inf)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v
