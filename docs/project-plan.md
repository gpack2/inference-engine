# Roadmap

The goal is to connect operator performance to model execution and request-level behavior. Each optimization needs a correctness comparison, a controlled benchmark, and an explanation of its effect on generation.

## Completed

- Qwen2.5-0.5B checkpoint loading into a custom PyTorch decoder, with strict weight-name and shape checks.
- Float32 CPU/MPS/CUDA inference, prompt prefill, greedy decoding, EOS handling, and context limits.
- Preallocated contiguous KV storage with ownership, capacity, and reset/reuse checks.
- Last-token vocabulary projection, preserving the full-logits forward interface.
- CPU/MPS reference validation, 30 offline tests, and saved performance measurements.
- Separate request/decode timing, component diagnostics, host operator profiles, and context-dependent KV storage accounting.
- A shared-RoPE experiment retained behind a flag after MPS regressions.

See the [engineering overview](portfolio.md) and [experiment index](../results/README.md) for implementation details and evidence.

## Next: scheduling and batching

Start with fixed, equal-length batches, then introduce a bounded request queue and iteration-level scheduling. Keep contiguous cache slots initially; paged storage is a separate extension.

Completion criteria:

- Requests enter and finish independently without sharing another request’s cache entries.
- Finished or cancelled requests release their slots.
- Admission respects explicit token and memory budgets.
- Mixed request lengths remain numerically correct.
- Load sweeps report queueing, throughput, and client-observed latency with declared measurement boundaries.

## NVIDIA baseline and custom kernels

The float32 implementation, checkpoint checks and initial request baseline now run on the RTX 2080 using the pinned CUDA 12.4 extra; see the [NVIDIA guide](nvidia.md). Next, profile this baseline before changing precision or replacing operators. Mac measurements cannot predict CUDA performance.

Candidate work:

1. Implement RMSNorm in CUDA C++ and Triton; compare reduction and launch choices across model-relevant shapes.
2. Implement fused SiLU-and-multiply in Triton while retaining library matrix multiplication.
3. Integrate one operation at a time, keeping reference implementations selectable.
4. Measure isolated device work and whole-request effects separately.

Use device timelines to investigate dispatch and synchronization; use kernel counters where available to support claims about bandwidth or occupancy. Device/kernel counter studies and custom kernels are planned, not implemented.

## Further investigations

- Explain the shared-RoPE MPS regression with a Metal device trace.
- Expand prompt coverage and validate longer contexts against the independent reference.
- Study single-query decode attention if attention becomes a significant measured cost.
- Consider paged KV storage or CUDA Graphs only after the baseline scheduler and memory behavior are understood.
- Compare against another engine only with aligned model, precision, workload, and metric definitions.

Training, multi-GPU execution, quantization, speculative decoding, and support for additional model families are outside the current scope.

## Experiment requirements

Record the checkpoint revision, input tokens, dtype, device, source hashes, software versions, warmups, and individual samples. Keep intrusive profiling outside primary timing runs, compare one change at a time, and preserve regressions alongside improvements.

For serving measurements, distinguish request submission to first returned token, consecutive-token latency, queue time, and completed output tokens per observation interval. Report percentile latency only when supported by sufficient samples; current five-run microbenchmarks do not establish tail latency.
