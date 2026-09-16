# LLM Inference Engine

A Qwen2.5-0.5B inference runtime with a custom decoder, contiguous KV cache, and CPU/MPS/CUDA validation and performance experiments.

I’m building this to study where inference time goes: model operations, memory management, and the generation loop. The runtime implements the forward pass and greedy decoding in PyTorch. The pretrained model is [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B); Hugging Face tools supply checkpoint access, tokenization, and the Transformers reference implementation used for validation.

**Working today:** float32 inference on CPU, Apple MPS and NVIDIA CUDA (validated on an RTX 2080), cached decoding, last-token vocabulary projection, and reproducible benchmarks. **Next:** custom CUDA/Triton kernels and request scheduling/batching.

[Engineering overview](docs/portfolio.md) · [Code walkthrough](docs/engine-walkthrough.md) · [Performance experiments](results/README.md)

## Architecture

```text
Text → Tokenizer → Prefill → Greedy token selection → Decode → …
                     │                                ↕
                     └──────── Per-layer KV cache ────┘
```

The decoder includes RMSNorm, split-half RoPE, grouped-query attention, and SwiGLU. Each layer writes keys and values into preallocated storage; subsequent decode calls process one new token against the valid cached prefix. Cache ownership, capacity, and position checks are explicit.

Generation projects only the final hidden-state position into vocabulary logits. Ordinary model forwards retain the full-logits interface. The Python API supports equal-length batches for forward passes; generation currently handles one request at a time.

## Measured results

These are comparisons within this engine, using Qwen2.5-0.5B in float32. Each primary case uses five timed runs after two warmups; the linked reports retain raw samples and measurement boundaries.

| Experiment | Workload and device | Result |
| --- | --- | --- |
| KV caching vs. full-prefix recomputation | 10 input / 16 output tokens, Apple MPS | **33.8% lower median request latency:** 649.1 → 429.4 ms |
| Last-token vocabulary projection | 256 input / 16 output tokens, Apple M5 CPU, one thread | **6.1% lower median request latency:** 688.6 → 646.6 ms |
| NVIDIA bring-up | 10 input / 16 output tokens, RTX 2080, PyTorch 2.6.0/CUDA 12.4 | Cached median **374.1 ms / 42.8 output tokens/s**; similar to uncached on this short workload |
| Shared RoPE setup | 16–448 initial context tokens, 32 decode steps, Apple M5 CPU/MPS | Small CPU gains in some cases; **5.8–25.3% slower MPS decoding**. Kept opt-in. |

Last-token projection also reduces the returned 256-token prefill logits tensor from **148.4 MiB to 0.58 MiB**. That is one tensor’s storage, not peak memory saved.

![KV caching comparison with individual measurements](docs/assets/kv-cache-baseline.png)

The RoPE experiment is a useful counterexample: reducing repeated setup from 48 times to once per forward did not reliably improve execution. Separate host profiles and synchronized component diagnostics help explain the limits of the measurements; the device-level cause of the MPS regression remains open.

Read the [projection study](docs/performance-measurement.md) and [decode profiling study](docs/decode-performance.md) for methods, correctness checks, results across all shapes, and limitations. These Mac experiments do not claim production serving throughput or predict NVIDIA performance. A separate [RTX 2080 baseline and validation](docs/nvidia.md) is now available.

## Run locally

From a checkout of this repository, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --frozen --extra cpu --python 3.12

# Pretrained completion through this engine
uv run --extra cpu tiny-infer --model qwen --device cpu \
  --prompt "The capital of France is" --max-new-tokens 16

# Tests use tiny local models; no checkpoint download is needed
uv run --extra cpu python -m unittest discover -s tests -v
```

The first Qwen run downloads approximately 1 GB into the Hugging Face cache. Add `--offline` on later runs. On a supported Mac, use `--device mps` for the Apple GPU. For a quick mechanics-only demo without downloading weights, use `--model tiny`; its random weights do not produce meaningful language.

For the NVIDIA/Linux setup, use the CUDA extra:

```bash
uv sync --frozen --extra cuda124 --python 3.12
uv run --extra cuda124 tiny-infer --model qwen --device cuda \
  --prompt "The capital of France is" --max-new-tokens 16
```

The [NVIDIA setup guide](docs/nvidia.md) covers the RTX 2080/driver 550.67 environment, preflight checks and validation. Keep the extra on each `uv run` command.

Useful comparison flags:

- `--no-cache`: recompute the full prefix at every step.
- `--full-logits`: project every input position into the vocabulary.
- `--share-rope`: opt into experimental shared RoPE setup; disabled by default because of measured regressions.

## Validation and layout

**30 tests** cover operation references, checkpoint mapping, causal masking, chunked prefill, cache ownership/reuse/overflow, generation, and benchmark accounting. Separate full-checkpoint checks compare layer outputs, logits, and greedy tokens against Hugging Face Transformers on CPU, MPS and the RTX 2080. Tolerances and fixture scope are documented in the [validation guide](docs/pretrained.md).

| Path | Contents |
| --- | --- |
| [`src/inference_engine/`](src/inference_engine/) | Decoder, tensor operations, KV cache, generation, checkpoint loader, CLI |
| [`tests/`](tests/) | Numerical and runtime invariants |
| [`benchmarks/`](benchmarks/) | Reference validation, request timing, projection comparisons, decode profiling |
| [`results/`](results/) | Raw measurements, model revision, environment and source hashes |
| [`docs/`](docs/) | Design explanations and performance case studies |
| [`learning/`](learning/) | Small tensor and attention exercises |

The [roadmap](docs/project-plan.md) tracks remaining work. The engine currently uses library matrix multiplication and explicit attention scores; it has no custom GPU kernels, continuous batching, or HTTP serving layer.
