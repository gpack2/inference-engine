# Running on NVIDIA

The engine uses the same PyTorch decoder, KV cache and generation loop on CUDA. Custom CUDA/Triton kernels are a separate next step.

The initial target is an **RTX 2080 (8 GiB)** on **Linux x86_64**, with **driver 550.67** and **Python 3.12**. The `cuda124` extra selects **PyTorch 2.6.0 + CUDA 12.4**. The Mac environment retains PyTorch 2.14.0 and MPS support.

## Install and run

From the repository root on the NVIDIA machine:

```bash
nvidia-smi
uv sync --frozen --extra cuda124 --python 3.12

# Small numerical check; fails if CUDA is unavailable instead of using CPU
uv run --frozen --extra cuda124 python benchmarks/check_cuda.py \
  --output results/local-cuda-preflight.json

uv run --frozen --extra cuda124 tiny-infer --model qwen --device cuda \
  --prompt "The capital of France is" --max-new-tokens 16
```

Keep `--extra cuda124` on subsequent `uv run` commands. A plain `uv run` can synchronize a different PyTorch build. Alternatively, after syncing, use `.venv/bin/python` and `.venv/bin/tiny-infer` directly on Linux. CPU-only installations can select `--extra cpu`; the two extras are mutually exclusive. The documented CUDA target uses Python 3.12; do not assume its older PyTorch wheel supports newer Python releases.

The first pretrained run downloads approximately 1 GB of checkpoint data. Subsequent commands can add `--offline`. Weights, package caches and virtual environments belong outside Git. On machines with small home-directory quotas, set `UV_CACHE_DIR` and `HF_HOME` to appropriate private scratch directories before installing/downloading. Scratch storage may be removed by the host; preserve code and result files elsewhere.

## Why this CUDA build

PyTorch publishes an official CUDA 12.4 build of version 2.6.0. NVIDIA lists Linux driver 550.54.14 for CUDA 12.4 GA; the target's 550.67 driver exceeds it. This avoids requiring a system-driver upgrade. [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v260) · [NVIDIA CUDA 12.4 release notes](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html)

`nvidia-smi` reports driver capability, not the CUDA runtime bundled with PyTorch. The preflight records both, plus GPU name, compute capability, VRAM, compiled architectures and float32 matmul settings. Its actual model execution checks that the installed stack works together. Passing that check does not establish compatibility for every NVIDIA GPU or driver.

Inference uses prebuilt PyTorch operations and float32 weights. There is no custom-kernel compilation step. Keep this baseline before introducing mixed precision, Tensor Core optimizations, or new kernels.

## Validate before benchmarking

```bash
uv run --frozen --extra cuda124 python -m unittest discover -s tests -v

uv run --frozen --extra cuda124 python benchmarks/verify_pretrained.py \
  --device cuda --offline --output results/local-cuda-verification.json

uv run --frozen --extra cuda124 python benchmarks/generation.py \
  --model qwen --device cuda --offline \
  --prompt "Explain how a GPU executes a matrix multiplication." \
  --new-tokens 16 --output results/local-cuda-generation.json
```

The unit suite includes CUDA whenever it is available. The independent checkpoint check compares layer outputs, full logits, chunked prefill and cached/uncached greedy generation against Transformers on the same device; CPU drift is reported separately. The verifier collects reference outputs on CPU and CUDA, releases the reference model, then loads the engine. Only one model is alive at a time; this avoids retaining two sets of weights under the target host’s 16 GiB per-process address-space limit. It still compares the same layers, logits and generated tokens.

Generation timing synchronizes at request boundaries and includes prefill, decode, selection, validation and cache allocation. Model loading and initial transfer are excluded. The first CUDA results use a different GPU, CPU, PyTorch version and runtime from the Mac studies; comparing those numbers does not isolate a hardware speedup.

Shared RoPE remains opt-in with `--share-rope`. The CUDA baseline retains recomputation; its MPS behavior does not predict its CUDA behavior.

## Recorded RTX 2080 validation

- [CUDA preflight](../results/rtx2080-preflight.json): real CUDA operations, CPU/CUDA tiny-model logits, cached/uncached tokens and chunked prefill passed.
- [30-test suite](../results/rtx2080-tests.txt): CPU and available CUDA paths passed.
- [Full-checkpoint comparison](../results/rtx2080-verification.json): all three prose/code fixtures passed; cached and uncached greedy tokens matched Transformers exactly. Maximum prefill logit error was approximately 2.34e-5.
- [Actual CLI output](../results/rtx2080-demo.txt): Qwen generated a completion on `device=cuda`.
- [Initial request benchmark](../results/rtx2080-generation.json): 10 input / 16 output tokens, five runs after two warmups, cache vs. full-prefix recomputation. This short workload is a baseline, not evidence that caching always helps.

The original verifier failed allocations while keeping multiple models alive and when creating CPU offload copies. The host reports a hard 16 GiB address-space limit. Keeping one model alive at a time let the unchanged numerical checks pass without changing host limits or relaxing tolerances. This is a validation-memory change, not an inference-speed optimization.

For the recorded 10-input/16-output-token request, cached generation had a median **374.09 ms (42.77 output tokens/s)**, versus **372.64 ms** without caching. That is a 0.4% latency regression on this short workload, not an established caching benefit. Both paths produced identical tokens. Longer contexts and an NVIDIA-specific profile are needed before choosing the next optimization.

The staged verifier was also rerun on [Mac CPU](../results/cuda-port-cpu-verification.json) and [MPS](../results/cuda-port-mps-verification.json); both passed the same three checkpoint fixtures.

## Troubleshooting

- **“No CUDA support”**: the selected PyTorch build is CPU/MPS-only. Re-sync with `--extra cuda124` and retain that option when invoking uv.
- **“No usable NVIDIA GPU”**: check the driver, host allocation and `CUDA_VISIBLE_DEVICES`. The preflight deliberately fails rather than silently benchmarking CPU.
- **Out of memory**: inspect other GPU processes and reduce the context cap/output length. Do not terminate another user's work. Qwen generation and reference validation have different host-memory requirements; inspect process limits as well as free VRAM.
- **Different host/driver**: validate the wheel and driver combination on that host before publishing measurements. This setup is validated only on the recorded target, not all Linux, Windows or NVIDIA configurations.
