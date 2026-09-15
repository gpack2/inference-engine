# Decode and RoPE measurements

Generated from completed raw reports by `docs/assets/render_decode.py`. Lower latency is better; positive reductions mean faster. Each request generates 33 tokens: one from prefill, then 32 decode steps. Five paired runs per variant after two warmups; float32, one request, one CPU thread, EOS disabled.

**Apple M5 · MPS**

[Raw samples, source hashes and environment](../results/decode-rope-mps.json)

| Initial context | Decode ms/step, recompute → shared | Decode reduction | Request ms, recompute → shared | Request reduction | Positive paired decode / request runs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 25.68 → 32.15 | -25.2% | 848.77 → 1052.81 | -24.0% | 0/5 · 0/5 |
| 64 | 26.10 → 32.71 | -25.3% | 855.56 → 1061.00 | -24.0% | 0/5 · 0/5 |
| 128 | 29.56 → 33.24 | -12.4% | 1027.70 → 1116.40 | -8.6% | 0/5 · 0/5 |
| 256 | 31.33 → 33.18 | -5.9% | 1031.40 → 1134.62 | -10.0% | 1/5 · 0/5 |
| 448 | 28.61 → 30.26 | -5.8% | 1095.75 → 1092.44 | 0.3% | 1/5 · 3/5 |

**Separate component diagnostics: 448 initial tokens**

Mean of two diagnostic run means, milliseconds per decode step. Synchronized components include host overhead; these are not normal execution percentages or pure GPU kernel times.

| Component | Recompute | Shared |
| --- | ---: | ---: |
| Attention projections | 19.309 | 19.826 |
| MLP projections | 25.702 | 27.715 |
| Vocabulary projection | 4.975 | 5.325 |
| Normalization | 10.846 | 12.309 |
| RoPE (including shared setup) | 11.475 | 11.441 |
| Attention scores / softmax / values | 11.429 | 13.890 |
| Cache, embedding, selection and residual | 26.163 | 26.830 |

**Host profiler: first four decode steps at this context**

| Operator | Calls, recompute → shared | Exclusive CPU self time ms, recompute → shared |
| --- | ---: | ---: |
| `aten::cos` | 192 → 4 | 0.238 → 0.010 |
| `aten::sin` | 192 → 4 | 0.263 → 0.015 |
| `aten::_local_scalar_dense` | 8 → 8 | 55.518 → 61.171 |
| `aten::linear` | 676 → 676 | 9.988 → 8.159 |
| `aten::bmm` | 192 → 192 | 16.652 → 3.871 |

Scalar-read time can include previous queued GPU work. It must not be interpreted as wholly removable validation cost.

**Apple M5 · CPU**

[Raw samples, source hashes and environment](../results/decode-rope-cpu.json)

| Initial context | Decode ms/step, recompute → shared | Decode reduction | Request ms, recompute → shared | Request reduction | Positive paired decode / request runs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 30.17 → 28.86 | 4.3% | 994.50 → 966.23 | 2.8% | 5/5 · 4/5 |
| 64 | 28.46 → 28.01 | 1.6% | 975.48 → 967.59 | 0.8% | 3/5 · 4/5 |
| 128 | 29.15 → 29.25 | -0.3% | 1039.06 → 1037.55 | 0.1% | 3/5 · 3/5 |
| 256 | 29.35 → 28.91 | 1.5% | 1134.13 → 1122.12 | 1.1% | 5/5 · 4/5 |
| 448 | 30.64 → 29.99 | 2.1% | 1335.39 → 1331.46 | 0.3% | 5/5 · 5/5 |

**Separate component diagnostics: 448 initial tokens**

Mean of two diagnostic run means, milliseconds per decode step. Synchronized components include host overhead; these are not normal execution percentages or pure GPU kernel times.

| Component | Recompute | Shared |
| --- | ---: | ---: |
| Attention projections | 2.511 | 2.496 |
| MLP projections | 16.171 | 16.120 |
| Vocabulary projection | 6.702 | 6.695 |
| Normalization | 0.330 | 0.320 |
| RoPE (including shared setup) | 0.771 | 0.343 |
| Attention scores / softmax / values | 3.466 | 3.461 |
| Cache, embedding, selection and residual | 1.143 | 1.141 |

**Host profiler: first four decode steps at this context**

| Operator | Calls, recompute → shared | Exclusive CPU self time ms, recompute → shared |
| --- | ---: | ---: |
| `aten::cos` | 192 → 4 | 0.072 → 0.006 |
| `aten::sin` | 192 → 4 | 0.060 → 0.004 |
| `aten::_local_scalar_dense` | 8 → 8 | 0.001 → 0.001 |
| `aten::linear` | 676 → 676 | 0.292 → 0.299 |
| `aten::bmm` | 192 → 192 | 6.553 → 6.576 |

Scalar-read time can include previous queued GPU work. It must not be interpreted as wholly removable validation cost.

**KV storage is unchanged by RoPE sharing**

| Initial context | Allocated capacity (tokens) | Valid tokens after decode | Allocated MiB | Valid MiB |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 49 | 48 | 1.148 | 1.125 |
| 64 | 97 | 96 | 2.273 | 2.250 |
| 128 | 161 | 160 | 3.773 | 3.750 |
| 256 | 289 | 288 | 6.773 | 6.750 |
| 448 | 481 | 480 | 11.273 | 11.250 |

One output token has not yet been inserted into the cache. The engine allocates one extra position for that token; valid storage is therefore one token smaller than capacity at the end. These are exact KV tensor sizes, not peak memory.

[Method, implementation and limitations](decode-performance.md)
