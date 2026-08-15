# Mac-first learning track

Start here on macOS. These are small learning examples with random tensors; no model download or paid GPU is needed. Causal attention is implemented using ordinary PyTorch operations and validated against a CPU reference. This is the first building block, not yet the full inference engine.

**This week's work (6–12 hours)**

| Time | Activity | Completion check |
| --- | --- | --- |
| 1–2 h | Run the tensor walkthrough; learn shape, stride, views, broadcasting, and batched matmul. | Predict the printed shapes/strides and explain which tensors share storage. |
| 2–4 h | Read and explain the attention implementation, then experiment with it. | All attention checks pass on CPU and on MPS when available. |
| 1–2 h | Explain the implementation and inspect intermediate tensors. | Draw the causal mask; explain scaling, softmax dimension, and the output shape. |
| 2–4 h, optional | Design the next exercise: incremental attention using cached K/V. | Write expected cache shapes, positions, and a cached-versus-full-prefix comparison before coding it. |

**Run the starter**

The repository uses `uv` and an isolated environment. From the repository root:

```bash
uv sync --python 3.12
uv run python learning/tensor_basics.py
uv run python learning/check_attention.py --device cpu
uv run python learning/check_attention.py --device mps
```

Both check commands should report six PASS groups. On a Mac without MPS support, use CPU. The default `--device auto` selects MPS when available, then CUDA, then CPU. Explicit requests for an unavailable device fail with a clear error. `--reference` checks the selected device's PyTorch built-in attention against the CPU oracle. Test inputs originate from a seeded CPU generator so devices receive the same data.

These scripts use float32 tensors and allocate intermediates on the input device. CPU and MPS have been verified locally; CUDA selection exists but has not been tested on NVIDIA hardware. The dependency configuration selects CPU PyTorch wheels on Linux/Windows and the standard macOS package. The lockfile records resolved versions; a future NVIDIA environment will need a separate dependency-source decision. See [PyTorch installation guidance](https://pytorch.org/get-started/locally/) for platform details.

**Mental model for the exercise**

For each batch item and attention head, every token has a query Q, key K, and value V. A query scores keys, softmax converts those scores into weights, and the output is a weighted sum of values. A causal mask prevents a position from reading later positions.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Q, K, V | `[B, H, T, D]` | Batch, attention heads, tokens, features per head |
| K transposed | `[B, H, D, T]` | Swap only the final two dimensions |
| Scores/weights | `[B, H, T, T]` | One row per query, one column per key |
| Output | `[B, H, T, D]` | Weighted values for each query |

The computation is `softmax((Q @ K^T) / sqrt(D) + mask) @ V`, with softmax over keys. Allowed mask entries add zero; future entries add negative infinity before softmax. This walkthrough covers equal-length self-attention with the same number of Q/K/V heads. Grouped-query attention and position embeddings come later. For the exact reference semantics, see [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

Read the [PyTorch tensor tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) alongside the walkthrough. Relate tensor strides to C++ pointer indexing: the offset for an index is the storage offset plus the sum of each index times its stride, measured in elements. A changed logical layout does not necessarily mean the underlying data moved.

**Questions to answer in your own notes**

1. Why does a transpose often produce a non-contiguous view, and when might a later operation need a copy?
2. Why do attention scores have two token dimensions? How does their storage grow with sequence length?
3. Why scale scores by `sqrt(D)`? Why is masking applied before softmax?
4. Why must the first token's output equal its value vector in this exercise?
5. What computation and data could be reused when one new token arrives?

**Mac now, NVIDIA later**

The tiny decoder, KV cache, and generation loop are now implemented in `src/inference_engine/`. Follow the [engine walkthrough](../docs/engine-walkthrough.md) after the attention example, and run `uv run python -m unittest discover -s tests -v` to check cached versus uncached computation on available devices. Pretrained checkpoint loading and parity checks are also implemented; see the [pretrained guide](../docs/pretrained.md). Request scheduling follows. Keep device and dtype choices at entry points, and allocate masks and caches from the input or model device. CPU remains the correctness oracle. PyTorch's [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html) executes supported operations through Apple's Metal framework.

When NVIDIA access is ready, install a compatible CUDA-enabled PyTorch environment, run the same correctness checks with `--device cuda`, and establish a new GPU baseline. Add CUDA/Triton kernels behind the same operation interfaces. Backend-specific timing and profiling will be separate: Mac results do not predict NVIDIA speedups. Custom Metal kernels are outside the current scope.

For the KV-cache exercise, distinguish absolute query position from its local tensor row. A decode query of length one may attend to the whole cached prefix. Do not blindly apply a square-prefill causal mask to a rectangular decode input.
