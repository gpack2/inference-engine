# Initial project plan

This plan incorporates the project owner's background and goals: strong Python/C++, moderate CUDA, no prior PyTorch or transformer-kernel experience, and 6–12 hours per week. The objective is learning and a portfolio for GPU, ML/AI systems, and performance engineering internships and full-time roles. A tiny random-weight decoder, contiguous KV cache, greedy generation, CPU/MPS correctness tests, and a synchronized smoke benchmark are implemented. See the [engine walkthrough](engine-walkthrough.md). Pretrained Qwen2.5-0.5B loading and short-prompt reference-model parity on CPU/MPS are now implemented as well; see the [pretrained guide](pretrained.md). The larger plan is not complete.

**Objective**

Trace performance from a GPU operation to a model forward pass to a running inference system. The deliverable is a readable engine plus experiments that explain which optimizations help, for which workloads, and why. A fast isolated kernel is useful evidence; its effect on generation must be measured separately.

**Scope and technical choices**

| Decision | Provisional choice |
| --- | --- |
| Runtime | macOS CPU/MPS for initial implementation; one NVIDIA GPU on Linux for CUDA/Triton optimization later |
| Host code | Python and PyTorch for tensor storage, reference operations, and orchestration |
| Kernel code | CUDA C++ first for RMSNorm, then implement the same operation in Triton; use Triton for a second operation and select later work based on profiling |
| Model | Proposed starting checkpoint: Qwen/Qwen2.5-0.5B; confirm fit and pin its revision during setup |
| Precision | One supported 16-bit format initially; higher-precision references where useful |
| Generation | Greedy decoding, causal attention, explicit context limits |
| First interface | Python API and command-line generation; HTTP serving is a later option |

Implement the model forward path, KV cache, generation loop, and selected kernels. Reuse tokenization, checkpoint loading utilities, and library matrix multiplication. Keep a reference implementation available for comparison. Implementing a tiled matrix multiplication is a useful exercise, but replacing a mature GEMM library is not a prerequisite.

Defer training, multi-GPU parallelism, quantization, speculative decoding, prefix sharing, and support for many model families. Avoid dependencies or a framework layout that require these features in the first version.

The proposed [Qwen2.5-0.5B checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B) includes RMSNorm, rotary position embeddings, SwiGLU, and grouped-query attention, giving the project several relevant kernel targets at a small model size. Account for its QKV biases and tied embeddings when mapping weights. Use plain text completion for this base checkpoint; chat behavior is outside the initial scope. Measurements on this model describe this workload, and should not be generalized to much larger models without experiments.

**Development order and GPU access**

Build the portable PyTorch model and runtime on the current Mac first. MPS is available on this machine, and the initial attention checks pass on both CPU and MPS. Keep the CPU reference, select devices at entry points, and create temporary tensors on the model/input device. Defer custom kernels until the model and cached generation work. When NVIDIA access arrives, validate the same operations on CUDA before changing implementations. Mac timings and NVIDIA timings are separate baselines. The NVIDIA dependency source, precision choices, and profiling tools will be configured on that machine; CUDA execution has not been verified locally.

**GPU access and budget**

First investigate the NVIDIA machines available through CMU: GPU model/VRAM, access schedule, environment restrictions, and profiler permissions. Use one primary GPU for comparable results. NVIDIA documents that [Nsight Compute hardware counters can require administrator-enabled access](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute); confirm this before depending on a machine for the report. Timings and available traces can support initial progress if counters are unavailable, but mark that analysis limitation and arrange a suitable profiling session later.

Continue PyTorch examples and tiny-model correctness work on Mac CPU/MPS immediately. Keep the existing AMD PC available for editing and CPU development; an additional AMD GPU backend is outside this project's scope. Select a compatible NVIDIA/Linux software stack when the actual machine is known, using the [Triton compatibility information](https://github.com/triton-lang/triton). A GPU purchase can wait until these access options and the workload are clear.

Treat the approximately $50 cloud budget as a total project cap. Proposed allocation: $5 for environment/profiler validation, $30 for targeted development and experiments, $10 for final reruns, and $5 for storage or other charges. These are spending envelopes, not provider price quotes or guaranteed compute hours. Verify current rates and profiler support before renting. Compute affordable hours as remaining compute dollars divided by the all-in hourly rate; include setup and compilation time. Download results and terminate instances after sessions, checking separately billed storage. Prepare code and workloads locally so rented time is used for GPU execution. No purchase or rental is part of this planning task.

**Learning sequence and language split**

Start with the [PyTorch tensor tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) and relevant parts of the [quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html): shapes, strides, broadcasting, matrix multiplication, device/dtype handling, modules, loading weights, and inference mode. Training loops and optimizers can wait.

Next, implement a tiny causal attention operation and decoder block with random weights. Write down tensor shapes for Q/K/V, attention scores, and grouped-query head mapping. Explain tokenization, embeddings, RMSNorm, RoPE, SwiGLU, residual connections, and why prefill differs from decode. The completion check is a causal-mask test and agreement between cached and full-prefix computation on tiny inputs.

Use CUDA C++ RMSNorm to practice warp/block reductions, memory access, launch choices, and integration with PyTorch through a [custom operator](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html). Reimplement RMSNorm in Triton for a controlled comparison of implementation effort and generated performance. Then implement fused SiLU-and-multiply for the SwiGLU activation in Triton; retain library GEMMs. Add RoPE only if time and profiling justify it. This yields experience with both languages without duplicating every kernel.

**Milestones and completion criteria**

Budget 120–180 focused hours including PyTorch/transformer foundations, batching, and the portfolio report. At an average of 8 hours per week, that is approximately 15–23 weeks; allow roughly 4–6 months with interruptions. At the full 6–12 hour range, the arithmetic spans 10–30 weeks. These are planning estimates, and GPU availability can extend elapsed time. The first complete checkpoint is approximately 80–115 hours.

| Stage | Work and learning focus | Completion evidence | Effort |
| --- | --- | --- | --- |
| 0. Foundations and baseline | Learn PyTorch and transformer concepts on Mac CPU/MPS. Build the tiny decoder exercise and establish a reference workload. Investigate NVIDIA access independently. | Tiny attention/cache checks, environment manifest, reference generation, and a Mac baseline. NVIDIA profiling follows when hardware is available. | 15–20 h |
| 1. Correct inference engine | On Mac CPU/MPS, implement the selected model's forward pass with library operations, prefill, a preallocated contiguous KV cache, and token-by-token decoding. | Logit comparisons against the reference; cached decoding agrees with full-context recomputation within justified tolerances; CLI generates text. | 25–35 h |
| 2. NVIDIA baseline and kernels | Validate the PyTorch engine on NVIDIA and capture a baseline trace. Implement CUDA RMSNorm, Triton RMSNorm, and Triton fused SiLU-and-multiply. Connect reduction/fusion choices to GPU execution and numerical stability. Bound optional softmax/GEMM exercises. | Two distinct model-relevant operations pass numerical checks; RMSNorm has a CUDA/Triton comparison; latency sweeps include awkward shapes. | 20–30 h |
| 3. Integrated optimization | Replace operations one at a time. Profile prefill and decode separately. Investigate fusion, allocation removal, and host launch overhead where traces justify them. | Before/after kernel and generation measurements, ablations, and explanations of wins or regressions. This is the first complete project checkpoint. | 20–30 h |
| 4. Batching and resource control | Add fixed batching, then iteration-level continuous batching with a bounded request queue and token/memory budgets. Begin with contiguous cache slots. | Requests can enter and finish independently; cache slots are reclaimed; mixed request lengths remain correct; load sweeps show latency/throughput tradeoffs. | 25–40 h |
| 5. Comparative report | Reproduce selected experiments, compare against a mature engine where configurations align, and explain remaining bottlenecks. | Scripts, raw results, plots, profiler evidence, and three case studies someone else can reproduce. | 15–25 h |

Profiling and correctness start in stage 0 and continue throughout. The final stage consolidates evidence already collected.

For an 8-hour week, allocate roughly 2 hours to concepts/design, 4 to implementation, and 2 to correctness, measurement, and notes. Use extra hours for the current milestone. At 6 hours, extend the schedule rather than dropping validation.

**Proposed architecture**

```text
CLI / Python caller / optional HTTP endpoint
                    |
           request queue + scheduler
                    |
        model runner: prefill / decode
             |                 |
         KV cache         operator backend
                          /              \
                   PyTorch reference   custom kernels
```

Prefill processes prompt tokens and initializes cached keys and values. Decode consumes the next token while reusing that cache. Initially the scheduler runs one request to completion; later it selects a batch on each iteration.

Keep kernel selection behind a small operator interface so the same weights, inputs, cache layout, and generation loop can run with reference or custom operations. The cache manager owns capacity, positions, and request-to-slot mappings. The scheduler owns admission and completion. The model runner executes the chosen work.

Create directories as they acquire implementations; this is a proposed layout:

```text
src/inference_engine/
  models/       # model definition and checkpoint mapping
  kernels/      # custom operations and dispatch
  runtime/      # generation, KV cache, scheduler
benchmarks/     # kernel, model, and request workload runners
tests/          # numerical and runtime correctness
configs/        # model and experiment configurations
results/        # small raw measurements and environment manifests
docs/           # design decisions and performance report
```

Keep weights and large profiler traces outside Git; record their locations and identities in experiment metadata.

**Performance experiments**

Start with a small matrix: batch sizes 1, 4, and 8; prompt lengths 128, 512, and 2,048; output lengths 32 and 128. Reduce these to fit memory. Treat fixed output lengths as benchmark settings and record how EOS stopping is handled. Later add mixed-length requests and arrival-rate sweeps to expose queueing and saturation.

| Layer | Measurements | Question |
| --- | --- | --- |
| Kernel | GPU latency, estimated or measured memory traffic, achieved bandwidth, occupancy, relevant compute counters | Is this operation limited by memory traffic, computation, or inefficient execution? |
| Model | Prefill latency, decode step latency versus context and batch size, peak allocated/reserved GPU memory | Did the change improve the part of inference it targeted? |
| System | Time to first token, inter-token latency, request latency, generated tokens/s, requests/s, queue time | What happens to responsiveness and throughput as load increases? |

Define time to first token from request submission to first returned token, including queueing at that boundary. Define inter-token latency from consecutive returned tokens; report its distribution separately from average time per output token. Throughput is completed output tokens divided by a declared observation interval. Report p50 and p95 latency with sample counts; reserve p99 claims for sufficiently large runs. Keep client-observed and engine-only timings distinct. Metric naming differs across tools, so align definitions before comparing with [vLLM's benchmark tooling](https://github.com/vllm-project/vllm/blob/main/benchmarks/README.md).

For trustworthy measurements:

1. Record commit, GPU/VRAM, driver, software versions, model/tokenizer revision, dtype, attention backend, workload, and seed.
2. Separate initialization, compilation, and warmup from steady-state measurements; report cold-start cost separately when relevant.
3. Use GPU events for isolated device work and synchronized boundaries for end-to-end GPU completion. Avoid synchronizing every production decode step solely for measurement.
4. Repeat measurements and preserve raw samples. Keep correctness checks and intrusive profiler collection outside the primary timing run.
5. Compare one change at a time against the same engine using reference operators. Add compiled PyTorch or vLLM comparisons later, with matching model, precision, workload, and generation settings; disclose remaining differences.
6. Show both improvements and regressions across shapes. Set numerical speed targets after the baseline exists; do not assume every custom kernel will beat optimized library code.

Use [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) for CPU/GPU timelines and NVTX annotations around scheduling, prefill, and decode. Use [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) on a few selected kernels for memory behavior, occupancy, and roofline analysis. Check profiler access early, especially on shared GPUs.

**Correctness and memory discipline**

Test kernels against trusted references across actual model shapes, non-tile-aligned sizes, supported layouts, and relevant numerical extremes. State supported layouts explicitly. Choose tolerances by operation and dtype; inspect error distributions rather than relying only on matching generated text.

For the engine, check causal masking, position offsets, cached versus uncached logits, prompt/context boundaries, different request lengths, EOS, and cache slot reuse. Continuous batching must preserve request isolation and reject or queue work that exceeds its declared budget. Add cancellation and failure cleanup if a serving interface is introduced.

Before loading a checkpoint, estimate weight memory as parameter count times bytes per parameter. For a uniform full-attention model, estimate KV memory as:

```text
2 × layers × total cached tokens × KV heads × head dimension × bytes per element
```

The factor of two represents keys and values. Budget against allocated cache capacity, not just currently used tokens, and leave measured headroom for activations, temporary workspaces, and allocator overhead. Adapt the estimate to the selected architecture.

**Optional depth after the first complete version**

Choose one substantial extension based on the measured bottleneck:

- Implement tiled attention with online softmax, validate causal prefill and single-query decode separately, and study intermediate memory traffic. Use the [FlashAttention paper](https://arxiv.org/abs/2205.14135) and [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) as references.
- Add paged KV storage and a compatible attention path, then compare memory waste and admitted concurrency against contiguous storage. The [PagedAttention paper](https://arxiv.org/abs/2309.06180) provides the underlying design reference.
- Explore CUDA Graphs for repeated decode work if host launch overhead matters; account for stable buffers, shape buckets, and capture constraints.

The preferred advanced kernel extension is a single-query decode-attention kernel that reads the contiguous KV cache and uses a stable softmax reduction. Budget another 20–35 hours as an initial estimate and reassess after a prototype; full prefill FlashAttention and paged attention are separate expansions. Select this extension after the main checkpoints if it best supports the measured bottleneck and learning goals.

**Portfolio evidence**

Build three case studies, each stating a hypothesis, workload, baseline, correctness evidence, measurements, explanation, and limitations:

1. CUDA versus Triton RMSNorm: how reduction and launch choices affect bandwidth and latency across shapes, including comparison to the reference operation.
2. Kernel-to-engine impact: how much time an operation originally consumed, its isolated improvement, and the resulting prefill/decode change. Use Amdahl's law to explain limited end-to-end gains where applicable.
3. Scheduling under load: fixed versus continuous batching, throughput versus p50/p95 latency, queueing, and cache capacity under mixed request lengths. Specify the policy for admitting new prefills; a simple whole-prompt prefill can delay ongoing decode, which should be measured and documented.

Package the result with a short demo, architecture explanation, reproducible commands, raw result files, and a technical report. Credit tutorials and reused code, and explain original changes. Document an unsuccessful optimization if it reveals a useful bottleneck. Resume claims should use measured results with the GPU, model, and workload identified; broad claims about outperforming production engines are not a project success criterion.

**First implementation session**

Work through the Mac tensor and attention examples, then inspect the implemented tiny decoder and cache using the engine walkthrough. The local environment is pinned, 17 offline tests pass, and separate full-checkpoint verification passes on CPU/MPS. The next implementation step is broader workload measurement with separate prefill/decode timing, then request scheduling. Record candidate NVIDIA GPU model/VRAM and profiler permissions independently; capture a separate CUDA baseline when available.

The NVIDIA machine and access conditions remain open for the optimization phase. They do not block Mac implementation. NVIDIA dtype, dependency versions, and experiment sizes follow from that choice; the current Mac examples use float32.
