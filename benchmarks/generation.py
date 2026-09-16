"""Small synchronized generation benchmark, including Python and allocation costs."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from inference_engine import ModelConfig, TinyDecoder, generate
from inference_engine.device import resolve_device, synchronize
from metadata import source_metadata, runtime_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--model", choices=("tiny", "qwen"), default="tiny")
    parser.add_argument("--prompt", help="Text prompt; otherwise use seeded random token IDs")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--full-logits", action="store_true", help="Use the original all-position projection")
    rope = parser.add_mutually_exclusive_group()
    rope.add_argument("--share-rope", dest="reuse_rope", action="store_true", help="Opt into shared RoPE setup")
    rope.add_argument("--recompute-rope", dest="reuse_rope", action="store_false", help="Use default per-application RoPE setup")
    parser.set_defaults(reuse_rope=False)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.prompt_length, args.new_tokens, args.repeats) < 1 or args.warmup < 0:
        parser.error("Lengths and repeats must be positive; warmup must be nonnegative")
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    torch.manual_seed(0)
    torch.set_num_threads(1)
    checkpoint = None
    if args.model == "qwen":
        from inference_engine.pretrained import load_qwen
        checkpoint = load_qwen(device=device, local_files_only=args.offline)
        model = checkpoint.model
    else:
        model = TinyDecoder(ModelConfig()).to(device)
    cfg = model.config
    if args.prompt is not None:
        ids = (checkpoint.tokenizer.encode(args.prompt, add_special_tokens=False)
               if checkpoint else list(args.prompt.encode("utf-8")))
        if not ids:
            parser.error("Prompt must contain at least one token")
        prompt = torch.tensor([ids], dtype=torch.long, device=device)
    else:
        prompt = torch.randint(0, cfg.vocab_size, (1, args.prompt_length)).to(device)
    if prompt.shape[1] + args.new_tokens > cfg.max_seq_len:
        parser.error("Requested sequence exceeds context capacity")
    projection = {"last_token_only": not args.full_logits, "reuse_rope": args.reuse_rope}
    cached = generate(model, prompt, args.new_tokens, **projection)
    uncached = generate(model, prompt, args.new_tokens, use_cache=False, **projection)
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
    for _ in range(args.warmup):
        for use_cache in (True, False):
            generate(model, prompt, args.new_tokens, use_cache=use_cache, **projection)
    samples = {"cached": [], "uncached": []}
    for repeat in range(args.repeats):
        # Alternate order to reduce consistent first/second-run bias.
        for use_cache in ((True, False) if repeat % 2 == 0 else (False, True)):
            synchronize(device)
            start = time.perf_counter()
            generate(model, prompt, args.new_tokens, use_cache=use_cache, **projection)
            synchronize(device)
            samples["cached" if use_cache else "uncached"].append(time.perf_counter() - start)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "generation_smoke_benchmark", "model": args.model,
        "model_id": checkpoint.model_id if checkpoint else None,
        "model_revision": checkpoint.revision if checkpoint else None,
        "last_token_only": not args.full_logits,
        "reuse_rope": args.reuse_rope,
        "prompt_text": args.prompt,
        "device": str(device), "platform": platform.platform(), "torch": torch.__version__,
        "python": platform.python_version(), "dtype": "float32", "seed": 0, "cpu_threads": 1,
        **source_metadata(), "runtime": runtime_metadata(device),
        "config": asdict(cfg), "prompt_ids": prompt.cpu()[0].tolist(),
        "new_tokens": args.new_tokens, "warmup_runs_per_mode": args.warmup,
        "repeats": args.repeats, "cached_uncached_token_agreement": True,
        "cache_allocated_bytes": model.new_cache(capacity=prompt.shape[1] + args.new_tokens).nbytes,
        "scope": "Full generation wall time: includes prefill, decode, Python validation, cache allocation, and token selection; excludes model initialization and prompt transfer. No EOS stopping.",
        "limitations": "Single request, short runs. Not serving throughput or a prediction of NVIDIA performance. The tiny model is untrained; Qwen uses the pinned checkpoint. Cache may be slower at small shapes.",
        "results": {
            mode: {"seconds": values, "median_seconds": statistics.median(values),
                   "output_tokens_per_second_at_median": args.new_tokens / statistics.median(values)}
            for mode, values in samples.items()
        },
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
