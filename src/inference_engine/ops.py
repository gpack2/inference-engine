"""Portable PyTorch operations; future custom kernels can replace these."""

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


def rope_frequencies(positions: torch.Tensor, dim: int, theta: float,
                     dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Build broadcastable cos/sin values once for a shared position range."""
    inverse_frequency = theta ** (-torch.arange(0, dim, 2, device=positions.device, dtype=dtype) / dim)
    angles = positions.to(dtype)[:, None] * inverse_frequency[None, :]
    return angles.cos()[None, None], angles.sin()[None, None]


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float,
               frequencies: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
    """Rotate paired features (split-half convention) at absolute positions.

    x: [batch, heads, tokens, even head dimension]; positions: [tokens].
    """
    cosine, sine = (rope_frequencies(positions, x.shape[-1], theta, x.dtype)
                    if frequencies is None else frequencies)
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, query_start: int = 0) -> torch.Tensor:
    """Causal GQA with explicit absolute-position masking.

    Q is [B, query_heads, query_tokens, D]. K/V are the full valid prefix
    [B, kv_heads, cached_tokens, D]. This includes both prefill and decode.
    The caller validates compatible shapes. K/V heads divide query heads.
    """
    repeats = q.shape[1] // k.shape[1]
    # Straightforward reference layout; optimized kernels should avoid copies.
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    query_positions = torch.arange(q.shape[-2], device=q.device) + query_start
    key_positions = torch.arange(k.shape[-2], device=q.device)
    future = key_positions[None, :] > query_positions[:, None]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return scores.masked_fill(future, -torch.inf).softmax(dim=-1) @ v
