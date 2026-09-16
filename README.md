# LLM Inference Engine

A small inference engine for [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), built to learn GPU programming and inference performance.

The decoder and generation loop are implemented in PyTorch, with KV caching, RoPE, grouped-query attention, and SwiGLU. It runs in float32 on CPU, Apple MPS, and NVIDIA CUDA. Hugging Face provides the pretrained weights, tokenizer, and reference model used for validation.

Custom CUDA/Triton kernels and request batching are planned.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync --frozen --extra cpu --python 3.12
uv run --frozen --extra cpu tiny-infer --model qwen --device cpu \
  --prompt "The capital of France is" --max-new-tokens 16
```

On Apple Silicon, replace `--device cpu` with `--device mps`.

For NVIDIA GPUs on Linux:

```bash
uv sync --frozen --extra cuda124 --python 3.12
uv run --frozen --extra cuda124 tiny-infer --model qwen --device cuda \
  --prompt "The capital of France is" --max-new-tokens 16
```

Keep the matching extra on each `uv run` command. The CUDA setup has been tested on an RTX 2080 with driver 550.67.

The first run downloads roughly 1 GB of model weights. Use `--offline` after downloading, or `--model tiny` to try the engine with random weights and no download.

## Tests

```bash
uv run --frozen --extra cpu python -m unittest discover -s tests -v
```

Tests use small local models. On the NVIDIA setup, replace `--extra cpu` with `--extra cuda124`.

## Project layout

- `src/inference_engine/` — decoder, KV cache, generation, and CLI
- `tests/` — correctness tests
- `benchmarks/` — timing and reference checks
- `results/` — saved benchmark data
- `docs/architecture.md` — engine structure
- `learning/` — tensor and attention exercises

See [architecture](docs/architecture.md) for the request flow and [results](results/README.md) for saved measurements.
