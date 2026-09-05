"""Full-checkpoint parity checks, separate from timing and offline unit tests."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import torch
import transformers
from transformers import AutoModelForCausalLM

from inference_engine import generate
from inference_engine.device import resolve_device
from inference_engine.pretrained import load_qwen
from metadata import source_metadata

PROMPTS = [
    "The capital of France is",
    "Explain why a GPU benefits from coalesced memory access:",
    "def add(a, b):\n    return",
]


@torch.inference_mode()
def captured_forward(model, tokens, *, reference=False, reuse_rope=False):
    outputs, handles = {}, []
    modules = {"embedding": model.model.embed_tokens if reference else model.embedding}
    layers = model.model.layers if reference else model.layers
    modules.update({f"layer_{i}": layer for i, layer in enumerate(layers)})
    modules["norm"] = model.model.norm if reference else model.norm

    def capture(name):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            outputs[name] = value.detach().cpu()
        return hook

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(capture(name)))
        result = model(tokens, use_cache=False) if reference else model(tokens, reuse_rope=reuse_rope)
        return (result.logits if reference else result).cpu(), outputs
    finally:
        for handle in handles:
            handle.remove()


def error_stats(actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    error = (actual - expected).abs()
    return {"max_abs_error": error.max().item(), "mean_abs_error": error.mean().item()}


def compare(actual, expected, *, atol=2e-4):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-4, atol=atol)
    return error_stats(actual, expected)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--share-rope", action="store_true", help="Validate experimental shared RoPE setup")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.new_tokens <= 64:
        parser.error("new-tokens must be between 1 and 64 for this bounded validation")
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1)
    checkpoint = load_qwen(device=device, local_files_only=args.offline)
    reference = AutoModelForCausalLM.from_pretrained(
        checkpoint.snapshot, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False,
    ).eval()
    # Same-device parity isolates implementation errors from backend arithmetic.
    # Keep a CPU oracle too, to quantify cross-device drift rather than hide it.
    device_reference = reference if device.type == "cpu" else copy.deepcopy(reference).to(device)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **source_metadata(),
        "model_id": checkpoint.model_id, "revision": checkpoint.revision,
        "engine_device": str(device), "reference_device": str(device), "drift_reference_device": "cpu",
        "dtype": "float32", "reference_attention": "eager", "reuse_rope": args.share_rope,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "rtol": 2e-4, "atol": 2e-4, "max_new_tokens": args.new_tokens,
        "changed_shape_atol": 1e-3,
        "changed_shape_tolerance_reason": "HF eager on MPS showed about 5.6e-4 maximum logit drift between full and chunked prefill in a diagnostic run. Same-shape implementation comparisons retain 2e-4 absolute tolerance; both engines' shape-dependent drift is recorded and bounded separately.",
        "scope": "Same-device short-prompt checkpoint parity; CPU drift is reported separately without applying the implementation tolerance to it. Not a quality evaluation, long-context validation, or speed benchmark.",
        "cases": [],
    }
    for text in PROMPTS:
        cpu_tokens = checkpoint.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        tokens = cpu_tokens.to(device)
        cpu_expected, cpu_layers = captured_forward(reference, cpu_tokens, reference=True)
        expected, expected_layers = captured_forward(device_reference, tokens, reference=True)
        actual, actual_layers = captured_forward(checkpoint.model, tokens, reuse_rope=args.share_rope)
        case = {"prompt": text, "prompt_tokens": tokens.shape[1], "logits": compare(actual, expected),
                "layers": {name: compare(actual_layers[name], value) for name, value in expected_layers.items()}}
        case["cross_device_drift"] = {
            "engine_vs_hf_cpu_logits": error_stats(actual, cpu_expected),
            "hf_device_vs_hf_cpu_logits": error_stats(expected, cpu_expected),
            "engine_vs_hf_cpu_layers": {name: error_stats(actual_layers[name], value) for name, value in cpu_layers.items()},
            "hf_device_vs_hf_cpu_layers": {name: error_stats(expected_layers[name], value) for name, value in cpu_layers.items()},
        }

        # A multi-token append after prefill exercises absolute-position masking.
        split = max(1, tokens.shape[1] // 2)
        cache = checkpoint.model.new_cache(capacity=tokens.shape[1] + args.new_tokens)
        prefill = checkpoint.model(tokens[:, :split], cache, reuse_rope=args.share_rope)
        suffix = checkpoint.model(tokens[:, split:], cache, reuse_rope=args.share_rope)
        ref_prefill = device_reference(tokens[:, :split], use_cache=True)
        ref_suffix = device_reference(tokens[:, split:], past_key_values=ref_prefill.past_key_values, use_cache=True)
        ref_chunked = torch.cat((ref_prefill.logits, ref_suffix.logits), dim=1)
        chunked = torch.cat((prefill, suffix), dim=1)
        case["chunked_logits"] = compare(chunked, ref_chunked)
        case["chunked_vs_full_logits"] = compare(chunked, expected, atol=1e-3)
        case["hf_chunked_vs_full_logits"] = compare(ref_chunked, expected, atol=1e-3)

        eos = checkpoint.tokenizer.eos_token_id
        expected_tokens = device_reference.generate(
            tokens, attention_mask=torch.ones_like(tokens), max_new_tokens=args.new_tokens,
            do_sample=False, eos_token_id=eos, pad_token_id=eos,
        )
        cached = generate(checkpoint.model, tokens, args.new_tokens, eos_token_id=eos, reuse_rope=args.share_rope)
        uncached = generate(checkpoint.model, tokens, args.new_tokens, use_cache=False, eos_token_id=eos, reuse_rope=args.share_rope)
        torch.testing.assert_close(cached.cpu(), expected_tokens.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(uncached.cpu(), expected_tokens.cpu(), rtol=0, atol=0)

        # Compare each decode step's full vocabulary, not just its winning token.
        cache.reset()
        checkpoint.model(tokens, cache, reuse_rope=args.share_rope)
        ref_cache = device_reference(tokens, use_cache=True).past_key_values
        errors = []
        for position in range(tokens.shape[1], expected_tokens.shape[1]):
            expected_full = device_reference(expected_tokens[:, :position + 1], use_cache=False).logits[:, -1:]
            ref_step = device_reference(expected_tokens[:, position:position + 1], past_key_values=ref_cache, use_cache=True)
            ref_cache = ref_step.past_key_values
            actual_step = checkpoint.model(expected_tokens[:, position:position + 1].to(device), cache, reuse_rope=args.share_rope)
            errors.append({"cached_reference": compare(actual_step, ref_step.logits),
                           "full_reference": compare(actual_step, expected_full, atol=1e-3),
                           "hf_cached_vs_full": compare(ref_step.logits, expected_full, atol=1e-3)})
        case["decode_logits"] = errors
        case["greedy_token_match_cached_and_uncached"] = True
        case["generated_ids"] = cached[0, tokens.shape[1]:].cpu().tolist()
        case["completion"] = checkpoint.tokenizer.decode(case["generated_ids"], skip_special_tokens=True)
        report["cases"].append(case)
        print(f"PASS on {device}: {text!r}; max logit error={case['logits']['max_abs_error']:.3g}", flush=True)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"Saved verification report to {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
