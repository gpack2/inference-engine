# Projection experiment: measured results

Generated from saved measurements. All times are milliseconds; arrows show full-position → last-position logits. Both variants use KV caching. Positive reduction means faster normal generation.

Generation values are medians of five ordinary timed runs after two warmups. Phase columns come from three separate synchronized diagnostic runs. Decode reports the median of each run's mean step time. Phase values do not add up to ordinary generation time.

All cases use Qwen2.5-0.5B, float32, batch size one and one CPU thread; generated token IDs matched exactly. See the [method and interpretation](performance-measurement.md) for scope and limitations.

**Apple M5 · Apple GPU / MPS**

| Input / output tokens | Generation ms | Reduction | Diagnostic prefill + first token ms | Diagnostic decode step ms |
| --- | ---: | ---: | ---: | ---: |
| 16 / 16 | 433.04 → 434.03 | -0.2% | 38.09 → 29.54 | 26.85 → 28.51 |
| 16 / 64 | 1683.58 → 1707.71 | -1.4% | 39.44 → 36.03 | 27.01 → 27.14 |
| 64 / 16 | 433.19 → 454.71 | -5.0% | 40.79 → 30.88 | 27.25 → 26.80 |
| 64 / 64 | 1785.66 → 1768.10 | +1.0% | 49.26 → 38.67 | 29.32 → 29.77 |
| 128 / 16 | 527.40 → 491.16 | +6.9% | 69.23 → 53.27 | 30.85 → 30.83 |
| 128 / 64 | 1943.33 → 1891.59 | +2.7% | 68.14 → 52.14 | 29.84 → 29.02 |
| 256 / 16 | 567.01 → 552.25 | +2.6% | 111.64 → 86.13 | 31.43 → 31.84 |
| 256 / 64 | 1951.39 → 1925.01 | +1.4% | 113.11 → 83.44 | 32.61 → 30.62 |

[Raw MPS samples, token IDs, tensor sizes, and environment](../results/projection-mps-matrix.json)

**Apple M5 · CPU / one thread**

| Input / output tokens | Generation ms | Reduction | Diagnostic prefill + first token ms | Diagnostic decode step ms |
| --- | ---: | ---: | ---: | ---: |
| 16 / 16 | 474.96 → 470.39 | +1.0% | 52.57 → 48.70 | 28.34 → 28.17 |
| 16 / 64 | 1849.62 → 1859.93 | -0.6% | 51.93 → 48.00 | 28.58 → 28.33 |
| 64 / 16 | 511.96 → 503.44 | +1.7% | 79.50 → 69.80 | 29.51 → 30.08 |
| 64 / 64 | 1903.69 → 1893.02 | +0.6% | 78.64 → 69.83 | 28.97 → 28.97 |
| 128 / 16 | 572.92 → 559.92 | +2.3% | 131.21 → 116.41 | 30.34 → 30.15 |
| 128 / 64 | 1970.77 → 1956.43 | +0.7% | 127.25 → 109.91 | 29.63 → 29.45 |
| 256 / 16 | 688.56 → 646.59 | +6.1% | 233.36 → 198.54 | 30.39 → 30.05 |
| 256 / 64 | 2189.72 → 2122.65 | +3.1% | 231.68 → 197.34 | 30.04 → 30.07 |

[Raw CPU samples, token IDs, tensor sizes, and environment](../results/projection-cpu-matrix.json)

**Storage example**

For 256 input / 16 output tokens, the allocated KV buffers remain 6.375 MiB in both variants. The prefill logits tensor shrinks from 148.375 MiB to 0.580 MiB. These are tensor storage sizes, not measured peak memory.

Regenerate this file with `uv run python docs/assets/summarize_projection.py`.
