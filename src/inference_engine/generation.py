"""Greedy generation for a single request, with cached and uncached paths."""

import torch

from .model import TinyDecoder


@torch.inference_mode()
def generate(model: TinyDecoder, prompt: torch.Tensor, max_new_tokens: int = 16,
             *, use_cache: bool = True, eos_token_id: int | None = None,
             last_token_only: bool = True, reuse_rope: bool = False) -> torch.Tensor:
    """Return prompt + generated tokens. EOS, when emitted, is included.

    This API keeps the full returned sequence within the declared context limit.
    A fresh cache is allocated per request and released after generation.
    last_token_only=False retains the full-logits baseline for comparisons.
    reuse_rope=True opts into experimental shared RoPE setup; recomputation
    remains the default because the shared path regressed on tested MPS workloads.
    """
    model.validate_tokens(prompt)
    if prompt.shape[0] != 1:
        raise ValueError("generate currently accepts one request at a time")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if prompt.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus requested output exceeds the model context limit")
    if eos_token_id is not None and (type(eos_token_id) is not int or not 0 <= eos_token_id < model.config.vocab_size):
        raise ValueError("eos_token_id must be within the vocabulary")
    sequence = prompt.clone()
    if max_new_tokens == 0:
        return sequence
    cache = model.new_cache(capacity=prompt.shape[1] + max_new_tokens) if use_cache else None
    current = prompt
    for _ in range(max_new_tokens):
        logits = model(current, cache=cache, last_token_only=last_token_only, reuse_rope=reuse_rope)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sequence = torch.cat((sequence, next_token), dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
        # Prefill consumes the prompt; subsequent cached calls consume one token.
        current = next_token if use_cache else sequence
    return sequence
