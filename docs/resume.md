# Resume entry

**LLM Inference Engine** · Python, PyTorch, Apple MPS

[github.com/gpack2/inference-engine](https://github.com/gpack2/inference-engine)

- Built a Qwen2.5-0.5B inference runtime with a custom decoder, preallocated KV cache, and greedy generation; validated numerical behavior with 25 tests and CPU/MPS checks against Hugging Face Transformers.
- Reduced median request latency by 33.8% with KV caching on a fixed 10-input/16-output-token Apple MPS workload, preserving exact generated-token agreement.
- Reduced the 256-token prefill logits tensor from 148.4 MiB to 0.58 MiB through last-token projection; benchmarked request latency and separate prefill/decode diagnostics across CPU/MPS workloads.

## Evidence and interview preparation

The project runs pretrained weights; it does not train Qwen or implement custom CUDA/Triton kernels yet. The memory claim concerns one output tensor, not peak device memory. The latency claim comes from five runs after two warmups, not a serving workload.

[KV-cache measurements](../results/qwen-mps-generation-smoke.json) · [Projection study](performance-measurement.md) · [Validation](pretrained.md)

Be ready to explain:

- Why prefill and decode have different tensor shapes and performance characteristics.
- How cache positions, capacity, ownership, and reset prevent incorrect reuse.
- Why computing fewer vocabulary rows saves prefill work but barely changes cached decoding.
- Why CPU profiler waits do not directly identify GPU kernel costs.
- Why shared RoPE remains opt-in after a measured MPS regression.

Use the entry when you can explain the implementation and reproduce the cited results. The repository should be public before using its link on a resume; otherwise reviewers need explicit access.
