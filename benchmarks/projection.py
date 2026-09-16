"""Compare all-position and last-position logits with cached generation.

Whole-request timing is separate from intrusive synchronized phase diagnostics.
RoPE sharing is disabled in both variants to retain the original experiment.
The default matrix stays within the pretrained loader's 512-token context cap.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

from inference_engine import generate
from inference_engine.device import resolve_device, synchronize
from inference_engine.pretrained import load_qwen
from metadata import source_metadata, runtime_metadata

PROMPT_TEXT = (
    "A GPU executes many threads in parallel. Threads read data from memory, "
    "perform arithmetic, and write their results. Efficient inference reuses "
    "cached keys and values while generating one new token at a time. "
)
MODES = {"full_logits": False, "last_token_logits": True}


def summary(values):
    return {"samples": values, "median": statistics.median(values),
            "min": min(values), "max": max(values)}


@torch.inference_mode()
def phase_trial(model, prompt, output_tokens, last_token_only):
    """Diagnostic wall times. Includes host work; these are not GPU event times.

    Cache allocation is outside these phase intervals. Each step ends with a
    device synchronization, so this trace is not a production inter-token trace.
    """
    cache = model.new_cache(capacity=prompt.shape[1] + output_tokens)
    sequence = prompt.clone()
    current = prompt
    phases = []
    first_logits_bytes = 0
    for step in range(output_tokens):
        synchronize(model.device)
        start = time.perf_counter()
        logits = model(current, cache, last_token_only=last_token_only, reuse_rope=False)
        next_token = logits[:, -1].argmax(-1, keepdim=True)
        sequence = torch.cat((sequence, next_token), dim=1)
        synchronize(model.device)
        phases.append((time.perf_counter() - start) * 1000)
        if step == 0:
            first_logits_bytes = logits.numel() * logits.element_size()
        current = next_token
    return sequence, {
        "prefill_first_token_ms": phases[0],
        "decode_step_ms": phases[1:],
        "cache_allocated_bytes": cache.nbytes,
        "prefill_logits_tensor_bytes": first_logits_bytes,
    }


def run_case(model, prompt, output_tokens, *, warmup, repeats, diagnostic_repeats, order_index):
    # Untimed output equivalence checks use the unchanged all-position path.
    full_logits = model(prompt, reuse_rope=False)[:, -1:].cpu()
    last_logits = model(prompt, last_token_only=True, reuse_rope=False).cpu()
    torch.testing.assert_close(last_logits, full_logits, rtol=2e-4, atol=1e-3)
    error = (full_logits - last_logits).abs()
    expected = generate(model, prompt, output_tokens, last_token_only=False, reuse_rope=False)
    actual = generate(model, prompt, output_tokens, last_token_only=True, reuse_rope=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected = expected.cpu()
    del full_logits, last_logits, actual

    for _ in range(warmup):
        for mode in MODES.values():
            generate(model, prompt, output_tokens, last_token_only=mode, reuse_rope=False)
    e2e = {name: [] for name in MODES}
    execution_order = []
    for repeat in range(repeats):
        names = list(MODES)
        if (order_index + repeat) % 2:
            names.reverse()
        execution_order.append(names)
        for name in names:
            synchronize(model.device)
            start = time.perf_counter()
            generate(model, prompt, output_tokens, last_token_only=MODES[name], reuse_rope=False)
            synchronize(model.device)
            e2e[name].append((time.perf_counter() - start) * 1000)

    # Run instrumented phases after normal generation, never inside its timer.
    diagnostic = {name: [] for name in MODES}
    for repeat in range(diagnostic_repeats):
        names = list(MODES)
        if (order_index + repeat) % 2:
            names.reverse()
        for name in names:
            sequence, measured = phase_trial(model, prompt, output_tokens, MODES[name])
            torch.testing.assert_close(sequence.cpu(), expected, rtol=0, atol=0)
            diagnostic[name].append(measured)

    variants = {}
    for name in MODES:
        phases = diagnostic[name]
        # Equal-length runs: retain full traces and summarize each run's mean
        # decode time before taking a median across diagnostic repetitions.
        decode_means = [statistics.mean(p["decode_step_ms"]) for p in phases]
        variants[name] = {
            "generation_ms": summary(e2e[name]),
            "output_tokens_per_second_at_median": output_tokens / (statistics.median(e2e[name]) / 1000),
            "diagnostic_prefill_first_token_ms": summary([p["prefill_first_token_ms"] for p in phases]),
            "diagnostic_mean_decode_step_ms": summary(decode_means),
            "diagnostic_runs": phases,
        }
    full, last = (variants[name] for name in MODES)
    baseline_ms, optimized_ms = full["generation_ms"]["median"], last["generation_ms"]["median"]
    paired_reductions = [(1 - new / old) * 100 for old, new in zip(e2e["full_logits"], e2e["last_token_logits"])]
    return {
        "prompt_tokens": prompt.shape[1], "output_tokens": output_tokens,
        "prompt_ids": prompt.cpu()[0].tolist(),
        "generated_ids": expected[0, prompt.shape[1]:].tolist(),
        "parity": {"generated_token_ids_equal": True, "diagnostic_token_ids_equal": True,
                   "prefill_last_logit_max_abs_error": error.max().item(),
                   "prefill_last_logit_mean_abs_error": error.mean().item(),
                   "logit_rtol": 2e-4, "logit_atol": 1e-3},
        "normal_generation_execution_order": execution_order,
        "variants": variants,
        "median_generation_latency_reduction_percent": (1 - optimized_ms / baseline_ms) * 100,
        "paired_generation_latency_reduction_percent": paired_reductions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="auto")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[16, 64, 128, 256])
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--diagnostic-repeats", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.prompt_lengths) < 1 or min(args.output_lengths) < 2:
        parser.error("Prompt lengths must be positive; output lengths must be >= 2 to measure decode")
    if args.warmup < 0 or min(args.repeats, args.diagnostic_repeats) < 1:
        parser.error("Repeats must be positive and warmup nonnegative")
    if max(args.prompt_lengths) + max(args.output_lengths) > 512:
        parser.error("This matrix must fit the 512-token context cap")
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1)
    checkpoint = load_qwen(device=device, local_files_only=args.offline)
    phrase = checkpoint.tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    chip = None
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        chip = result.stdout.strip() if result.returncode == 0 else None
    elif device.type == "cuda":
        chip = torch.cuda.get_device_name(device)
    report = {
        "kind": "last_token_projection_ablation", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False, "device": str(device), "hardware": chip, "platform": platform.platform(),
        "torch": torch.__version__, "python": platform.python_version(), "dtype": "float32", "cpu_threads": 1,
        "model_id": checkpoint.model_id, "model_revision": checkpoint.revision,
        "model_config": asdict(checkpoint.model.config), **source_metadata(), "runtime": runtime_metadata(device),
        "prompt_construction": {"text": PROMPT_TEXT, "method": "Repeat tokenized phrase, truncate to exact requested length; single fixed prompt per shape"},
        "settings": {"warmup_per_variant": args.warmup, "reuse_rope": False, "repeats": args.repeats,
                     "diagnostic_repeats": args.diagnostic_repeats,
                     "prompt_lengths": args.prompt_lengths, "output_lengths": args.output_lengths},
        "measurement_scope": {
            "normal_generation": "Synchronized wall time around generate; no added per-step instrumentation. Includes host validation, prefill, decode, argmax, concatenation and fresh cache allocation. Excludes model loading, tokenization and initial prompt transfer. EOS disabled; cache enabled for both variants.",
            "diagnostic": "Separate runs synchronize each step. Prefill includes model forward, first-token argmax and append. Decode includes forward, argmax and append per subsequent token. Cache allocation is outside phase timers. These are diagnostic phase latencies, not streaming TTFT/ITL or pure GPU kernel times; they must not be summed to estimate normal generation latency.",
            "memory": "Exact tensor storage bytes for KV buffers and returned prefill logits, not peak device memory or measured total-memory savings.",
            "comparison": "Same checkpoint, device, dtype, prompt, output length and cache; only last_token_only differs. Order alternates within paired repeats and across cases. Small samples, no confidence-interval or tail-latency claims.",
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for prompt_length in args.prompt_lengths:
        ids = (phrase * ((prompt_length + len(phrase) - 1) // len(phrase)))[:prompt_length]
        prompt = torch.tensor([ids], dtype=torch.long, device=device)
        for output_length in args.output_lengths:
            case = run_case(checkpoint.model, prompt, output_length, warmup=args.warmup,
                            repeats=args.repeats, diagnostic_repeats=args.diagnostic_repeats,
                            order_index=len(report["cases"]))
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{device}: prompt={prompt_length}, output={output_length}; "
                  f"generation reduction={case['median_generation_latency_reduction_percent']:.1f}%; parity PASS", flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved complete report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
