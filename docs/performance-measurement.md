# Performance case study: project only the logits generation needs

The engine originally projected every prompt position into the vocabulary, then discarded all but the final position's logits when choosing the next token. This experiment changes that projection and measures prefill, decode, and whole-request effects separately. It compares two paths in the same engine, with KV caching enabled in both.

**The change**

For a prompt of length `T`, Qwen2.5-0.5B produces hidden states shaped `[1, T, 896]`. The output projection maps them to `[1, T, 151936]`. Generation only needs `[1, 1, 151936]` to choose the next token.

```python
# Before
logits = self.lm_head(normalized)

# During generation now
logits = self.lm_head(normalized[:, -1:, :])
```

Every prompt token still passes through all decoder layers, populates every layer's KV cache, and receives final normalization. Only the vocabulary projection is reduced. Keeping normalization unchanged makes this an isolated optimization. Ordinary forward calls still return every position's logits by default; `last_token_only=True` opts into the smaller output. Generation enables it by default, and `--full-logits` selects the baseline from the CLI.

At 256 prompt tokens, the returned float32 logits tensor falls from **148.375 MiB to about 0.580 MiB**, a 256-fold reduction in this one tensor's storage. The projected row count also falls from 256 to one. This is not a 256-fold model speedup or a measurement of peak device-memory savings: decoder work, weights, and the KV cache are unchanged.

**Measurement design**

The matrix covers 16, 64, 128, and 256 input tokens, each with 16 and 64 output tokens, on CPU and Apple MPS. It uses one pinned Qwen2.5-0.5B checkpoint, float32, one request at a time, and one CPU thread. Prompts are deterministic repetitions of a tokenized paragraph truncated to exact lengths. Token IDs and source hashes are saved with each result. EOS stopping is disabled so output counts stay fixed.

| Measurement | Included | Interpretation |
| --- | --- | --- |
| Normal generation latency | One complete `generate` call, including host validation, fresh cache allocation, prefill, decode, token selection and output concatenation | Synchronized only at the outer benchmark boundaries; excludes model loading, tokenization and initial prompt transfer |
| Diagnostic prefill / first token | Prompt forward, first-token selection and append | A separate synchronized phase run; cache allocation excluded; not client-visible time to first token |
| Diagnostic decode step | One cached token forward, selection and append | Every step is synchronized for diagnosis; not production inter-token latency or pure GPU execution time |
| Tensor storage | Allocated KV buffer bytes and returned prefill logits bytes | Exact tensor sizes, not peak memory or allocator reservations |

Normal generation uses five timed runs after two warmups per variant, with alternating order within paired repeats and across cases. Instrumented diagnostics run separately, three times per variant, after normal timing. They retain every decode-step sample and summarize each diagnostic run's mean step time. Do not sum diagnostic times to predict ordinary request latency: the synchronization schedule and allocation boundary differ.

These small samples support an initial comparison, not confidence intervals or tail-latency claims. A single prompt per shape does not characterize a real workload distribution. CPU/MPS results are separate device baselines; the single-thread CPU setting is for control, not a claim about the fastest CPU configuration.

**Correctness gate**

The optimized projection must retain the final-position logits within declared float32 tolerances and produce exactly the same greedy token IDs as the full-logits path on every matrix case. Diagnostic runs must also return those exact IDs. Unit tests additionally check that the cache contains all prompt positions, later decoding remains correct, both output contracts work, and EOS behavior is preserved with and without caching.

At this experiment’s checkpoint, the 21-test offline suite passed. Full-checkpoint validation against Transformers is repeated on CPU and MPS with optimized generation enabled, using the existing three prose/code prompts. [CPU reference checks](../results/projection-cpu-verification.json) · [MPS reference checks](../results/projection-mps-verification.json)

**Measured result**

![Whole-request latency changes across prompt and output lengths](assets/last-token-projection.png)

All 16 cases completed with exact baseline/optimized generated-token agreement, including the separately instrumented paths. Diagnostic prefill/first-token medians were lower in every case: **8.7–26.2% on MPS** and **7.4–14.9% on CPU**. Whole-request improvements were smaller and sometimes absent.

| Device | Input / output tokens | Median generation: full → last | Change in latency |
| --- | --- | ---: | ---: |
| Apple M5 MPS | 256 / 16 | 567.01 → 552.25 ms | 2.6% lower |
| Apple M5 MPS | 256 / 64 | 1951.39 → 1925.01 ms | 1.4% lower |
| Apple M5 CPU, one thread | 256 / 16 | 688.56 → 646.59 ms | 6.1% lower |
| Apple M5 CPU, one thread | 256 / 64 | 2189.72 → 2122.65 ms | 3.1% lower |

The strongest CPU case above, 256 input / 16 output tokens, improved median generation latency by **6.1%**; all five paired measurements improved, by 4.7–7.5%. Across all CPU cases, median changes ranged from a 0.6% regression to a 6.1% improvement.

MPS request medians ranged from a **5.0% regression to a 6.9% improvement** across the full matrix. Samples overlap and some paired comparisons change sign. For 256 input / 64 output tokens, diagnostic prefill fell from 113.11 to 83.44 ms, while normal generation improved only 1.4%. Five samples on one Mac are insufficient to promise a consistent end-to-end MPS speedup.

The interpretation follows the scope of the code change: prefill no longer computes discarded vocabulary rows, but each cached decode call already has just one row. Longer outputs add many decode steps that this optimization does not materially change. This is an example of Amdahl's law: improving one phase has limited effect when the rest of the request dominates. Phase measurements also include host work; they do not prove a particular GPU hardware bottleneck.

The optimization remains the generation default because it removes unnecessary projection work, shrinks the logits tensor, preserves the cache/output contracts, and improved measured prefill medians. The full-logits option remains available; the result is not a universal latency guarantee. No second optimization was bundled into this comparison.

[All phase timings and tensor sizes](projection-results.md) · [Raw MPS matrix](../results/projection-mps-matrix.json) · [Raw CPU matrix](../results/projection-cpu-matrix.json)

**Reproduce**

```bash
uv run python -m unittest discover -s tests -v
uv run python benchmarks/projection.py --device mps --offline --output results/projection-mps-matrix.json
uv run python benchmarks/projection.py --device cpu --offline --output results/projection-cpu-matrix.json
```

Each matrix loads its checkpoint once. Run device benchmarks sequentially to avoid competition for the Mac's shared resources. Reports are saved after each case and marked `complete` only after the matrix finishes. RoPE sharing is explicitly disabled in both projection variants to preserve this original experiment after the later decode work. The older generation benchmark accepts `--full-logits --recompute-rope` to reproduce both original computation modes. See the [subsequent decode study](decode-performance.md) for that separate optimization.

Regenerate the figures and tables without rerunning inference:

```bash
uv run python docs/assets/summarize_projection.py
uv run --no-project --python 3.12 docs/assets/render_projection.py
```
