# GPU-Optimized LLM Inference Engine

A learning project connecting custom GPU kernels, autoregressive inference, and system-level performance analysis.

Build a small, correct inference engine for one decoder-only transformer on one GPU. Optimize selected operations, integrate them into generation, and explain the measured effects on latency, throughput, and memory use.

**Status:** the engine runs pretrained Qwen2.5-0.5B on CPU and Apple MPS, using its own decoder, contiguous KV cache, and greedy generation loop. A tiny random-weight model remains available for fast offline experiments. Continuous batching and CUDA/Triton kernels are still ahead. CUDA is selectable but untested on hardware.

Run from the repository root:

```bash
uv sync --python 3.12
uv run tiny-infer --model qwen --prompt "The capital of France is" --device mps --max-new-tokens 16
uv run python -m unittest discover -s tests -v
```

The first Qwen run downloads approximately 1 GB into the Hugging Face cache outside the repository. Add `--offline` after downloading. Use `--model tiny` for the untrained 361,600-parameter byte-vocabulary demo. Use `--device cpu` for CPU or omit the flag for automatic device selection. Add `--no-cache` to recompute the full prefix at every generation step.

The model includes RMSNorm, RoPE, grouped-query attention, and SwiGLU. The forward pass supports equal-length batches; generation currently accepts one request. All computation is float32. The [engine walkthrough](docs/engine-walkthrough.md) explains the code; the [pretrained-model guide](docs/pretrained.md) covers weight mapping, validation, and numerical tolerances.

Run the initial generation benchmark:

```bash
uv run python benchmarks/generation.py --model qwen --device mps --offline --prompt "Explain GPU memory access:" --output results/local-generation.json
```

It checks cached/uncached token agreement, warms up both paths, synchronizes timing boundaries, and records raw samples and environment metadata. A [pretrained Mac smoke result](results/qwen-mps-generation-smoke.json) is included. These short single-request runs are not serving or NVIDIA performance claims. The original [tiny-model result](results/mac-mps-generation-smoke.json) is retained as an earlier development baseline.

Designed for 6–12 hours per week, with Python/PyTorch orchestration and future CUDA C++ and Triton kernels. NVIDIA GPU access is pending; prefer available CMU hardware, with approximately $50 total cloud spending as a fallback.

Start with the [Mac-first learning track](learning/README.md): a tensor walkthrough and implemented causal attention with CPU/MPS correctness checks.

The proposed progression is:

1. Learn the required PyTorch and transformer fundamentals; establish a reference model and reproducible performance baseline.
2. Build prefill, cached decoding, and a generation loop on Mac using PyTorch.
3. Establish an NVIDIA baseline, then implement and profile CUDA/Triton kernels.
4. Integrate custom kernels and measure their end-to-end impact.
5. Add batching and study latency, throughput, and memory tradeoffs.
6. Publish reproducible experiments and a technical report.

The first complete version should generate text correctly, use at least two custom kernels, and include an evidence-based performance analysis. Continuous batching, paged KV storage, and advanced optimizations follow that checkpoint.

The portfolio target adds continuous batching and three reproducible case studies covering kernels, decoding, and request scheduling. Budget approximately 120–180 focused hours for that target; an advanced attention or memory-management extension is additional scope.

See [the initial project plan](docs/project-plan.md) for scope, milestones, architecture, and the benchmark methodology.
