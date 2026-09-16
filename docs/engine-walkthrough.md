# First engine slice

The engine runs either a small randomly initialized decoder or pretrained Qwen2.5-0.5B on CPU/Apple MPS. The tiny default has two layers, hidden size 128, four query heads, two KV heads, 256 byte tokens, and a 256-token context limit. The Qwen loader supplies the real tokenizer, model dimensions, tied embeddings, and pretrained weights, with a default local context cap of 512 tokens. See the [pretrained guide](pretrained.md) for commands and validation reports.

**Read the code in this order**

| File | Purpose | What to inspect |
| --- | --- | --- |
| `src/inference_engine/ops.py` | RMSNorm, rotary positions, causal grouped-query attention | Tensor dimensions, reductions, and intermediate allocations |
| `src/inference_engine/model.py` | Decoder blocks, embedding, final projection | Attention and MLP residual paths; absolute position construction |
| `src/inference_engine/cache.py` | Preallocated per-layer K/V storage | Valid prefix length, capacity, and buffer reuse |
| `src/inference_engine/generation.py` | Prefill and greedy decode | Why the first call receives the prompt and later calls receive one token |
| `src/inference_engine/pretrained.py` | Pinned snapshot and strict weight mapping | How each checkpoint tensor maps into our decoder |
| `tests/test_engine.py` | Numerical and runtime checks | Cached versus full computation, chunked prefill, and request isolation |
| `benchmarks/generation.py` | Initial timing harness | Synchronization boundaries and what the measurement includes |

These components follow common decoder conventions. The [Transformers Qwen2 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py) is a reference for the architecture vocabulary, and [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) supplies an independent attention oracle in the tests. The tiny configuration has random weights and an untied output projection. The pretrained loader configures the same decoder for the supported Qwen checkpoint, ties the output projection to the embeddings, and maps weights explicitly. Full-checkpoint verification compares layer outputs, logits, and generated tokens with Transformers.

**One request through the engine**

1. The CLI tokenizes plain text with Qwen's tokenizer in pretrained mode. In tiny mode it uses UTF-8 bytes as toy token IDs from 0 to 255.
2. Prefill sends all prompt tokens through the embedding and each decoder block. Each block normalizes its input, creates Q/K/V, applies RoPE to Q/K, computes causal attention, adds the residual, then runs a normalized SwiGLU MLP and adds its residual.
3. Each layer saves its rotated keys and values into the cache. The final norm processes all positions; generation projects only the final position into vocabulary logits. Greedy selection takes the highest-scoring token at the last prompt position.
4. Decode sends that one selected token into the model at the next absolute position. Its query attends to the entire valid prefix, including its own newly written key and value. Previously cached keys and values are reused.
5. Repeat until the requested output length or an optional emitted EOS token. The returned tensor includes the prompt and any emitted EOS.

Example: a four-token prompt fills positions 0–3 and produces the first new token. The next model call consumes that new token at position 4. Its query tensor has length one, but it can attend to five cached keys. A causal mask based only on local row zero would be incorrect. The implementation compares absolute query positions with key positions.

**Cache contract**

Storage has shape `[layers, batch, KV heads, capacity, head dimension]` for each of K and V. Only positions before `cache.length` contain a committed prefix. Every layer writes the same new range; the length advances once after the full forward pass succeeds. If a layer raises, partial writes beyond the old length are not committed and a retry overwrites them.

The cache belongs to one model and a fixed lockstep batch. It checks capacity, device, dtype, and batch size before writes. Resetting sets length to zero and reuses the allocated storage; stale entries remain inaccessible until overwritten. Reset or discard a cache after changing model weights, and allocate a new cache after moving the model. Do not share a cache between concurrent requests. Ragged batches, padding, request admission, and continuous batching are not implemented yet.

Generation allocates a fresh cache per call. Its output sequence, including the prompt, must fit the declared context limit. The last generated token need not be inserted into the cache because no subsequent logits are requested. The generation API does not return a resumable cache in this version.

**What is verified**

The 30 offline unittest cases cover pretrained weight mapping and Transformers parity on tiny configurations, plus operation references, absolute rotary positions, full versus chunked and single-token cached logits, future-token isolation, batched versus individual forwards, cache reset/capacity/ownership, failed-forward position handling, cached/uncached generation, EOS, and invalid inputs. Device-dependent numerical checks run on CPU and any locally available MPS/CUDA device. CPU, MPS and RTX 2080 CUDA checks passed; see the [NVIDIA guide](nvidia.md). Separate full-checkpoint scripts validate the actual pretrained weights on short prompts; see the [pretrained guide](pretrained.md). Last-token projection preserves the full-logits forward API and can be disabled for an ablation; see the [performance case study](performance-measurement.md).

**How to interpret the benchmark**

The benchmark uses either a seeded, untrained model or the pinned Qwen checkpoint and a fixed number of output tokens with no EOS stopping. Prompts can be supplied as text or generated as seeded random token IDs; the result records which was used. It excludes model initialization and initial prompt transfer. It includes prefill, decode, greedy selection, output concatenation, per-request cache allocation, and Python validation. It measures synchronized wall time around a complete generation request; it does not report client-visible time to first token or inter-token latency. The projection benchmark adds separate, synchronized prefill/first-token and decode-step diagnostics; these are not streaming latencies. Results contain individual samples and a median, not unsupported tail-latency claims.

Several deliberate baseline costs remain: materialized attention score matrices, repeated KV heads for grouped-query attention, and token range validation that reads GPU scalars on the host. At this scale, launch and host overhead may outweigh saved arithmetic. Keep these costs visible when choosing future optimizations. Checkpoint parity is established for the tested short prompts; longer contexts and broader workloads need additional validation.

**Next milestone**

The [decode case study](decode-performance.md) now measures normal decode latency, synchronized components, host operator counts and KV storage across context lengths. RoPE setup can be shared once per forward; the benchmark retains the recomputed baseline. Next, add request scheduling and bounded batching. NVIDIA work follows the same correctness checks before custom CUDA/Triton operations replace selected PyTorch operations.
