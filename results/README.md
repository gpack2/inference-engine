# Performance experiments

All measurements use float32 and one request at a time. CPU runs use one thread. Check each JSON file for its exact model, workload, environment, source hashes, and timing scope.

| Experiment | Reports | What it establishes |
| --- | --- | --- |
| NVIDIA bring-up | [Preflight](rtx2080-preflight.json), [checkpoint verification](rtx2080-verification.json), [tests](rtx2080-tests.txt), [request baseline](rtx2080-generation.json) | RTX 2080, PyTorch 2.6.0/CUDA 12.4; same engine on NVIDIA |
| KV caching | [MPS smoke run](qwen-mps-generation-smoke.json) | Cached vs. uncached request latency for 10 input / 16 output tokens; an initial five-run experiment |
| Last-token projection | [CPU matrix](projection-cpu-matrix.json), [MPS matrix](projection-mps-matrix.json) | 16–256 input tokens × 16/64 output tokens, normal request timing and separate phase diagnostics |
| Decode and shared RoPE | [CPU sweep](decode-rope-cpu.json), [MPS sweep](decode-rope-mps.json) | 16–448 initial tokens, 32 decode steps, request timing, component diagnostics and host operator counts |
| RoPE regression repeat | [MPS confirmation](decode-rope-mps-confirmation.json) | A second 16-token-context run confirmed the slowdown; its magnitude differed |
| Latest checkpoint validation | [CPU](rope-cpu-verification.json), [MPS](rope-mps-verification.json) | Shared-RoPE candidate compared against Transformers on three prose/code fixtures; not a speed test |

The `cuda-port-cpu-verification.json` and `cuda-port-mps-verification.json` reports recheck the staged verifier on Mac after NVIDIA bring-up.

The earlier `qwen-*-verification.json`, `projection-*-verification.json`, and `mac-mps-generation-smoke.json` files preserve previous validation/development checkpoints.

## Reading the evidence

- [Projection methodology and reproduction](../docs/performance-measurement.md) · [All projection results](../docs/projection-results.md)
- [Decode methodology and reproduction](../docs/decode-performance.md) · [All decode results](../docs/decode-results.md)
- [Numerical validation and tolerances](../docs/pretrained.md)

Primary comparisons alternate execution order and retain individual samples. Synchronized component diagnostics perturb execution and are not a decomposition of ordinary latency. CPU profiler events on MPS describe host dispatch and waits, not Metal kernel time. Tensor-byte calculations are not peak-memory measurements.

Source hashes identify the files used for each experiment, including runs made before a commit. Historical reports are immutable evidence; later documentation or default-setting changes can make those hashes differ from the current checkout. The RTX 2080 file establishes only its recorded single-request CUDA baseline. No results here establish serving throughput or tail latency. See the [NVIDIA guide](../docs/nvidia.md) for environment differences from the Mac studies.
