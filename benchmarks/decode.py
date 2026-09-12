"""Context sweep with uninstrumented decode timing and separate diagnostics.

CPU profiler events on MPS describe host dispatch/waits, not Metal kernel time.
Synchronized component timings deliberately perturb execution and are diagnostic.
"""

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from unittest.mock import patch

import torch
from torch import nn

import inference_engine.model as model_module
from inference_engine import generate
from inference_engine.device import resolve_device, synchronize
from inference_engine.ops import RMSNorm
from inference_engine.pretrained import load_qwen
from metadata import source_metadata
from projection import PROMPT_TEXT, summary


class Components:
    """Benchmark-only, non-nested scopes; restore every patched method on exit."""

    def __init__(self, model, cache):
        self.model, self.cache = model, cache
        self.ms = defaultdict(float)
        self.calls = defaultdict(int)
        self.stack = ExitStack()

    def measure(self, name, function, *args, **kwargs):
        synchronize(self.model.device)
        start = time.perf_counter()
        result = function(*args, **kwargs)
        synchronize(self.model.device)
        self.ms[name] += (time.perf_counter() - start) * 1000
        self.calls[name] += 1
        return result

    def wrap(self, obj, attribute, category):
        original = getattr(obj, attribute)
        self.stack.enter_context(patch.object(
            obj, attribute, lambda *a, **kw: self.measure(category, original, *a, **kw)))

    def __enter__(self):
        self.wrap(self.model, "validate_tokens", "validation")
        self.wrap(model_module, "apply_rope", "rope")
        self.wrap(model_module, "rope_frequencies", "shared_rope_setup")
        self.wrap(model_module, "attention", "attention")
        self.wrap(self.cache, "write", "cache_write")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                category = "vocabulary_projection" if name == "lm_head" else (
                    "mlp_linear" if name.endswith(("gate_proj", "up_proj", "down_proj"))
                    else "attention_linear")
            elif isinstance(module, RMSNorm):
                category = "normalization"
            elif isinstance(module, nn.Embedding):
                category = "embedding"
            else:
                continue
            self.wrap(module, "forward", category)
        return self

    def __exit__(self, *exception):
        return self.stack.__exit__(*exception)


def select_and_append(logits, sequence):
    token = logits[:, -1].argmax(dim=-1, keepdim=True)
    return token, torch.cat((sequence, token), dim=1)


@torch.inference_mode()
def prepare(model, prompt, steps, reuse_rope=True):
    cache = model.new_cache(capacity=prompt.shape[1] + steps + 1)
    logits = model(prompt, cache, last_token_only=True, reuse_rope=reuse_rope)
    token, sequence = select_and_append(logits, prompt)
    synchronize(model.device)
    return cache, token, sequence


@torch.inference_mode()
def decode_trial(model, prompt, steps, *, diagnostic=False, host_profile=False, reuse_rope=True):
    """Exclude prefill/allocation; time steps decode calls and token selection.

    First call sees prompt_length cached tokens, then the cache grows each step.
    Returned sequence includes the untimed first token produced during prefill.
    """
    cache, token, sequence = prepare(model, prompt, steps, reuse_rope)
    with ExitStack() as stack:
        components = stack.enter_context(Components(model, cache)) if diagnostic else None
        profiler = stack.enter_context(torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False,
            with_stack=False)) if host_profile else None
        synchronize(model.device)
        start = time.perf_counter()
        for _ in range(steps):
            logits = model(token, cache, last_token_only=True, reuse_rope=reuse_rope)
            token, sequence = (components.measure("select_append", select_and_append, logits, sequence)
                               if components else select_and_append(logits, sequence))
        synchronize(model.device)
        elapsed = (time.perf_counter() - start) * 1000
    result = {"decode_total_ms": elapsed, "decode_mean_step_ms": elapsed / steps,
              "decode_steps": steps, "initial_cache_length": prompt.shape[1],
              "final_cache_length": cache.length, "cache_allocated_bytes": cache.nbytes,
              "cache_valid_bytes": cache.nbytes * cache.length // cache.capacity}
    if components:
        result["component_ms_per_step"] = {k: v / steps for k, v in components.ms.items()}
        result["component_calls_per_step"] = {k: v / steps for k, v in components.calls.items()}
        result["unattributed_ms_per_step"] = (elapsed - sum(components.ms.values())) / steps
    if profiler:
        events = sorted(profiler.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
        result["host_operator_events"] = [
            {"name": e.key, "calls": e.count, "self_cpu_ms": e.self_cpu_time_total / 1000,
             "inclusive_cpu_ms": e.cpu_time_total / 1000} for e in events]
    return sequence, result


def run_case(model, prompt, steps, *, warmup, repeats, diagnostic_repeats, order_index=0):
    modes = {"recompute_rope": False, "shared_rope": True}
    expected = generate(model, prompt, steps + 1, reuse_rope=False).cpu()
    actual = generate(model, prompt, steps + 1, reuse_rope=True).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Check every vocabulary logit and valid cache entry along a teacher-forced
    # decode path, not only the winning token. This includes the full context.
    caches = [model.new_cache(capacity=prompt.shape[1] + steps + 1) for _ in modes]
    max_error = 0.0
    chunks = [prompt] + [expected[:, i:i+1].to(model.device)
                         for i in range(prompt.shape[1], prompt.shape[1] + steps)]
    for chunk in chunks:
        logits = [model(chunk, cache, last_token_only=True, reuse_rope=mode).cpu()
                  for cache, mode in zip(caches, modes.values())]
        torch.testing.assert_close(logits[0], logits[1], rtol=2e-4, atol=2e-4)
        max_error = max(max_error, (logits[0] - logits[1]).abs().max().item())
    for field in ("keys", "values"):
        valid = [getattr(c, field)[:, :, :, :c.length].cpu() for c in caches]
        torch.testing.assert_close(valid[0], valid[1], rtol=2e-4, atol=2e-4)
    del caches, chunks, logits, valid
    for _ in range(warmup):
        for mode in modes.values():
            decode_trial(model, prompt, steps, reuse_rope=mode)
            generate(model, prompt, steps + 1, reuse_rope=mode)
    normal, diagnostics, requests = ({name: [] for name in modes} for _ in range(3))
    order = []
    for repeat in range(repeats):
        names = list(modes)
        if (order_index + repeat) % 2:
            names.reverse()
        order.append(names)
        for name in names:
            actual, trace = decode_trial(model, prompt, steps, reuse_rope=modes[name])
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            normal[name].append(trace)
        for name in names:
            synchronize(model.device)
            start = time.perf_counter()
            actual = generate(model, prompt, steps + 1, reuse_rope=modes[name])
            synchronize(model.device)
            requests[name].append((time.perf_counter() - start) * 1000)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    for repeat in range(diagnostic_repeats):
        names = list(modes)
        if (order_index + repeat) % 2:
            names.reverse()
        for name in names:
            actual, trace = decode_trial(model, prompt, steps, diagnostic=True, reuse_rope=modes[name])
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            diagnostics[name].append(trace)
    variants = {}
    for name, mode in modes.items():
        actual, host = decode_trial(model, prompt, min(steps, 4), host_profile=True, reuse_rope=mode)
        torch.testing.assert_close(actual.cpu(), expected[:, :prompt.shape[1] + min(steps, 4) + 1], rtol=0, atol=0)
        variants[name] = {
            "normal_mean_decode_step_ms": summary([t["decode_mean_step_ms"] for t in normal[name]]),
            "generation_ms": summary(requests[name]),
            "normal_runs": normal[name], "diagnostic_runs": diagnostics[name],
            "component_median_ms_per_step": {
                key: statistics.median(t["component_ms_per_step"][key] for t in diagnostics[name])
                for key in diagnostics[name][0]["component_ms_per_step"]},
            "host_profile": host}
    baseline, optimized = variants.values()
    return {"context_tokens": prompt.shape[1], "decode_steps": steps,
            "prompt_ids": prompt.cpu()[0].tolist(), "generated_ids": expected[0, prompt.shape[1]:].tolist(),
            "all_trial_token_ids_equal": True, "valid_cache_entries_close": True,
            "all_decode_logit_max_abs_error": max_error, "logit_rtol": 2e-4, "logit_atol": 2e-4,
            "paired_execution_order": order, "variants": variants,
            "median_decode_latency_reduction_percent": (1 - optimized["normal_mean_decode_step_ms"]["median"] / baseline["normal_mean_decode_step_ms"]["median"]) * 100,
            "median_generation_latency_reduction_percent": (1 - optimized["generation_ms"]["median"] / baseline["generation_ms"]["median"]) * 100,
            "paired_decode_latency_reduction_percent": [
                (1 - new["decode_total_ms"] / old["decode_total_ms"]) * 100
                for old, new in zip(normal["recompute_rope"], normal["shared_rope"])],
            "paired_generation_latency_reduction_percent": [
                (1 - new / old) * 100 for old, new in zip(requests["recompute_rope"], requests["shared_rope"])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="auto")
    parser.add_argument("--contexts", type=int, nargs="+", default=[16, 64, 128, 256, 448])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--diagnostic-repeats", type=int, default=2)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.contexts) < 1 or args.steps < 1 or max(args.contexts) + args.steps + 1 > 512:
        parser.error("Positive context/steps and context + steps + 1 <= 512 are required")
    if args.warmup < 0 or min(args.repeats, args.diagnostic_repeats) < 1:
        parser.error("Repeats must be positive and warmup nonnegative")
    device = resolve_device(args.device)
    torch.set_num_threads(1)
    checkpoint = load_qwen(device=device, local_files_only=args.offline)
    phrase = checkpoint.tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    chip = (subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                           capture_output=True, text=True).stdout.strip()
            if platform.system() == "Darwin" else platform.processor())
    report = {"kind": "cached_decode_profile_and_rope_ablation", "complete": False,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "device": str(device), "hardware": chip, "platform": platform.platform(),
              "torch": torch.__version__, "python": platform.python_version(),
              "dtype": "float32", "cpu_threads": 1,
              "model_id": checkpoint.model_id, "model_revision": checkpoint.revision,
              "model_config": asdict(checkpoint.model.config), **source_metadata(),
              "settings": {k: v for k, v in vars(args).items() if k != "output"},
              "prompt_construction": {"text": PROMPT_TEXT, "method": "Repeat encoded phrase then truncate; one prompt per shape"},
              "measurement_scope": {
                  "normal": "Outer synchronized wall time for 32 (or --steps) growing-cache decode calls, argmax and append. Prefill, first output token, cache allocation, model loading, tokenization and initial transfer excluded; last-token projection enabled, EOS disabled. No per-component instrumentation.",
                  "generation": "Separate outer synchronized generate calls; includes fresh cache allocation, prefill, all output token selections and decode; steps + 1 output tokens, EOS disabled. Excludes load/tokenization/initial transfer. Both variants use last-token projection and identical validation.",
                  "comparison": "Only per-forward RoPE frequency sharing differs. Paired order alternates by repeat and context. Small sample, fixed prompt per shape; no confidence intervals or tail-latency claims.",
                  "components": "Separate intrusive runs synchronize before/after non-nested components. Includes Python dispatch and device completion waits. Residual includes SiLU, residual adds, views, Python loop and instrumentation. Not pure GPU kernel timing, and not a decomposition of normal unsynchronized execution.",
                  "host_profiler": "Separate first four decode steps: PyTorch CPU activity only, including host dispatch/waits on MPS. Self times are exclusive; inclusive times overlap and must not be summed. No Metal kernel timing inferred.",
                  "memory": "Exact allocated and valid KV tensor bytes; not allocator-reserved or peak process/device memory.",
              }, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for context in args.contexts:
        ids = (phrase * ((context + len(phrase) - 1) // len(phrase)))[:context]
        prompt = torch.tensor([ids], dtype=torch.long, device=device)
        case = run_case(checkpoint.model, prompt, args.steps, warmup=args.warmup,
                        repeats=args.repeats, diagnostic_repeats=args.diagnostic_repeats,
                        order_index=len(report["cases"]))
        report["cases"].append(case)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{device}: context={context}, decode reduction={case['median_decode_latency_reduction_percent']:.1f}%, "
              f"request reduction={case['median_generation_latency_reduction_percent']:.1f}%; parity PASS", flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
