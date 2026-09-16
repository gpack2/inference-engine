# Pretrained model support

The engine loads [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) at revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. Its own PyTorch decoder performs every forward pass, cache update, and greedy generation step. Hugging Face supplies checkpoint download/tokenization and a separate reference implementation for validation.

**Run on Mac**

```bash
uv sync --python 3.12
uv run tiny-infer --model qwen --device mps --prompt "The capital of France is" --max-new-tokens 16
```

The first run caches approximately 1 GB of checkpoint data outside Git. Later runs can add `--offline`. The source weights are converted to float32 for the current engine. CPU is also supported with `--device cpu`. CUDA is validated on the RTX 2080; use the `cuda124` environment extra and `--device cuda` as described in the [NVIDIA guide](nvidia.md). The model is a base completion model: use plain text prompts; no chat template or instruction-tuned behavior is assumed.

The default engine context cap is 512 tokens. `--context-length` can change it within the checkpoint's declared context limit, but large contexts have not been validated here and the reference attention implementation materializes quadratic score matrices. Generation requires prompt length plus requested output length to fit the configured cap. The tiny model remains available with `--model tiny` and requires no downloaded checkpoint.

**Loading contract**

`src/inference_engine/pretrained.py` reads only the pinned snapshot's config, tokenizer files, and safetensors. It constructs a model on the meta device, validates every tensor name and shape, converts weights to CPU float32, assigns them, and moves the model to the selected device. This avoids initializing another full set of random weights. Remote Python code is not enabled.

The mapping includes Q/K/V biases, both normalization weights per layer, all attention and MLP projections, the final norm, and token embeddings. When embeddings are tied, the output projection shares the exact embedding parameter. Missing weights, unexpected weights, wrong shapes, and inconsistent declared ties are rejected. Support is intentionally limited to full-attention Qwen2 with default rotary positions and SwiGLU; sliding-window and scaled/multimodal rotary variants are rejected.

**Validation**

The 30 fast tests include tiny random Hugging Face configurations and do not download any models:

```bash
uv run python -m unittest discover -s tests -v
```

Full-checkpoint checks are explicit, separate runs:

```bash
uv run python benchmarks/verify_pretrained.py --share-rope --device cpu --offline --output results/rope-cpu-verification.json
uv run python benchmarks/verify_pretrained.py --share-rope --device mps --offline --output results/rope-mps-verification.json
```

With last-token projection and experimental shared RoPE enabled, the [CPU report](../results/rope-cpu-verification.json) and [MPS report](../results/rope-mps-verification.json) cover three short prose/code prompts. They compare the embedding output, every decoder layer, the final normalization, all prefill logits, chunked prefill, and each cached decode step's full vocabulary. Cached and uncached greedy generation must both exactly match Transformers' greedy token IDs. The reference uses float32 and eager attention. This validates these cases; it is not a language-quality evaluation or evidence of correctness for every context length.

Same-device, same-shape comparisons use `rtol=2e-4, atol=2e-4`. The reports record maximum and mean absolute errors. CPU/MPS arithmetic drift is recorded separately for both our model and the Hugging Face implementation rather than treating it as a weight-mapping error.

Full versus chunked prefill changes matrix shapes. In a diagnostic MPS run, Hugging Face itself showed approximately `5.6e-4` maximum logit drift between those paths, while our chunked logits matched its chunked logits within about `1.3e-5`. Therefore comparisons across different shapes use `atol=1e-3` with the same relative tolerance. Both implementations' shape-dependent errors are recorded and bounded. Exact greedy-token agreement remains required on these fixtures. Tolerances are not a substitute for investigating a new failure.

**Initial performance measurement**

```bash
uv run python benchmarks/generation.py --model qwen --device mps --offline --prompt "Explain how a GPU executes a matrix multiplication." --new-tokens 16 --full-logits --recompute-rope --output results/qwen-mps-generation-smoke.json
```

The [saved run](../results/qwen-mps-generation-smoke.json) measures complete single-request generation, including Python orchestration, prefill, decode, cache allocation, and token selection. Model loading and prompt transfer are excluded. It records five samples per path after two warmups, alternates cached/uncached order, and synchronizes at timing boundaries. Output length is fixed and EOS stopping is disabled for this benchmark. This saved run predates the last-token projection optimization; `--full-logits --recompute-rope` reproduces its original projection mode. The [projection case study](performance-measurement.md) adds a CPU/MPS matrix and separate diagnostic prefill/decode timings. The [decode case study](decode-performance.md) adds a context sweep, synchronized component diagnostics and CPU operator profiles. Client-visible time-to-first-token, inter-token latency, load sweeps and GPU kernel traces remain ahead.

Reports identify the model revision, dependency versions, local commit/dirty state, and source hashes. Weight files stay in the Hugging Face cache. NVIDIA execution and performance remain untested; the CUDA environment must be configured and validated when hardware is available.
