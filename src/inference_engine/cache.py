"""Fixed-capacity contiguous KV storage for one lockstep batch."""

import torch


class KVCache:
    def __init__(self, *, layers, batch_size, kv_heads, capacity, head_dim, device, dtype, owner):
        self.keys = torch.empty(layers, batch_size, kv_heads, capacity, head_dim, device=device, dtype=dtype)
        self.values = torch.empty_like(self.keys)
        self.length = 0
        self.capacity = capacity
        self.owner = owner

    @property
    def nbytes(self) -> int:
        return (self.keys.numel() + self.values.numel()) * self.keys.element_size()

    def reset(self) -> None:
        """Reuse storage. Old values are unreachable until overwritten."""
        self.length = 0

    def write(self, layer: int, key: torch.Tensor, value: torch.Tensor):
        """Write at the current offset; the model commits length once per call."""
        end = self.length + key.shape[-2]
        self.keys[layer, :, :, self.length:end, :].copy_(key)
        self.values[layer, :, :, self.length:end, :].copy_(value)
        return self.keys[layer, :, :, :end, :], self.values[layer, :, :, :end, :]
