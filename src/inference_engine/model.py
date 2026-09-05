"""A tiny inference-only decoder with RMSNorm, RoPE, GQA, and SwiGLU.

The default configuration is a random-weight toy. The checkpoint loader can
configure the same decoder for the supported pretrained Qwen2.5 model.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .cache import KVCache
from .ops import RMSNorm, apply_rope, attention, rope_frequencies


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    def __post_init__(self):
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_layers",
                     "num_heads", "num_kv_heads", "max_seq_len"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads or self.num_heads % self.num_kv_heads:
            raise ValueError("hidden_size must divide into heads; query heads must divide into KV groups")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        width = config.hidden_size
        kv_width = config.num_kv_heads * config.head_dim
        self.attention_norm = RMSNorm(width, config.norm_eps)
        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, kv_width, bias=True)
        self.v_proj = nn.Linear(width, kv_width, bias=True)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.mlp_norm = RMSNorm(width, config.norm_eps)
        self.gate_proj = nn.Linear(width, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(width, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, width, bias=False)

    def forward(self, x, positions, cache, layer, frequencies=None):
        batch, tokens, _ = x.shape
        cfg = self.config
        normalized = self.attention_norm(x)
        q = self.q_proj(normalized).view(batch, tokens, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(normalized).view(batch, tokens, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(normalized).view(batch, tokens, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q = apply_rope(q, positions, cfg.rope_theta, frequencies)
        k = apply_rope(k, positions, cfg.rope_theta, frequencies)
        start = 0 if cache is None else cache.length
        if cache is not None:
            k, v = cache.write(layer, k, v)
        attended = attention(q, k, v, query_start=start)
        x = x + self.o_proj(attended.transpose(1, 2).reshape(batch, tokens, cfg.hidden_size))
        normalized = self.mlp_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized))


class TinyDecoder(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._cache_owner = object()
        self.apply(self._initialize)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.eval()

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def device(self):
        return self.embedding.weight.device

    def new_cache(self, batch_size=1, capacity=None) -> KVCache:
        capacity = self.config.max_seq_len if capacity is None else capacity
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(capacity) is not int or not 1 <= capacity <= self.config.max_seq_len:
            raise ValueError("cache capacity must be within the model context limit")
        return KVCache(layers=self.config.num_layers, batch_size=batch_size,
                       kv_heads=self.config.num_kv_heads, capacity=capacity,
                       head_dim=self.config.head_dim, device=self.device,
                       dtype=self.embedding.weight.dtype, owner=self._cache_owner)

    def validate_tokens(self, token_ids):
        if token_ids.ndim != 2 or 0 in token_ids.shape:
            raise ValueError("token_ids must be a nonempty [batch, tokens] tensor")
        if token_ids.dtype != torch.long or token_ids.device != self.device:
            raise ValueError("token_ids must be int64 on the model device")
        if self.embedding.weight.dtype != torch.float32:
            raise ValueError("This initial model supports float32 weights")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.config.vocab_size:
            raise ValueError("token id is outside the vocabulary")

    @torch.inference_mode()
    def forward(self, token_ids: torch.Tensor, cache: KVCache | None = None,
                *, last_token_only: bool = False, reuse_rope: bool = False) -> torch.Tensor:
        """Return [batch, tokens, vocab] logits; optionally append to cache.

        All requests in a batch share one sequence length. No padding or ragged
        batching yet. Only new tokens should be passed when reusing a cache.
        Cache offsets advance after every layer and the output projection succeed.
        Reset/discard the cache if model weights change; it is not thread-safe.
        last_token_only projects just the final position to vocabulary logits,
        returning [batch, 1, vocab]. All input positions still update the cache.
        reuse_rope=True shares per-forward cos/sin across Q/K and every layer.
        This is opt-in: measured MPS regressions keep recomputation the default.
        """
        self.validate_tokens(token_ids)
        start = 0
        if cache is not None:
            if cache.owner is not self._cache_owner:
                raise ValueError("cache belongs to a different model")
            if cache.keys.device != self.device or cache.keys.dtype != self.embedding.weight.dtype:
                raise ValueError("cache device/dtype must match the model; create a new cache after moving it")
            if cache.keys.shape[1] != token_ids.shape[0]:
                raise ValueError("cache batch size mismatch")
            start = cache.length
            if start + token_ids.shape[1] > cache.capacity:
                raise ValueError("cache capacity exceeded")
        end = start + token_ids.shape[1]
        if end > self.config.max_seq_len:
            raise ValueError("model context limit exceeded")
        positions = torch.arange(start, end, device=self.device)
        x = self.embedding(token_ids)
        # Every layer uses the same positions, head dimension and RoPE theta.
        # Share only this call's frequencies: no persistent buffer or stale
        # device/position state, and checkpoint loading remains unchanged.
        frequencies = (rope_frequencies(positions, self.config.head_dim,
                                       self.config.rope_theta, x.dtype)
                       if reuse_rope else None)
        for index, layer in enumerate(self.layers):
            x = layer(x, positions, cache, index, frequencies)
        normalized = self.norm(x)
        # Generation consumes only the final position's logits. Slice after all
        # decoder/cache work, before the expensive hidden-to-vocabulary matmul.
        # Keep normalization unchanged to isolate the projection optimization.
        if last_token_only:
            normalized = normalized[:, -1:, :]
        logits = self.lm_head(normalized)
        if cache is not None:
            cache.length = end
        return logits
