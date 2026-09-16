# Architecture

The engine runs Qwen2.5-0.5B using PyTorch operations on CPU, Apple MPS, or NVIDIA CUDA.

```text
Prompt → Tokenizer → Decoder → Logits → Greedy token selection
                       ↕                         │
                    KV cache ←── next token ─────┘
```

## Request flow

1. Load the checkpoint and tokenize the prompt.
2. Run the full prompt through the decoder to fill the KV cache and select the first output token.
3. Feed each selected token back through the decoder, reusing cached keys and values.
4. Stop at the output limit or an end-of-sequence token.

Each decoder block contains RMSNorm, rotary position embeddings, grouped-query attention, and a SwiGLU MLP, with residual connections. Generation computes vocabulary logits only for the final position.

## Code

- [`model.py`](../src/inference_engine/model.py): decoder blocks and forward pass.
- [`ops.py`](../src/inference_engine/ops.py): normalization, rotary positions, and attention.
- [`cache.py`](../src/inference_engine/cache.py): preallocated per-layer KV storage.
- [`generation.py`](../src/inference_engine/generation.py): prefill and greedy decoding.
- [`pretrained.py`](../src/inference_engine/pretrained.py): checkpoint loading and weight mapping.
- [`cli.py`](../src/inference_engine/cli.py): command-line interface.

The cache tracks its valid prefix and belongs to one model and batch. Generation handles one request at a time in float32. Custom GPU kernels and request scheduling are future work.
