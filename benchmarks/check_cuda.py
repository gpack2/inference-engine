"""Fail-fast CUDA environment and tiny-engine parity check; no downloads."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import torch

from inference_engine import ModelConfig, TinyDecoder, generate
from inference_engine.device import resolve_device, synchronize
from metadata import runtime_metadata, source_metadata


@torch.inference_mode()
def check():
    device = resolve_device("cuda")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    model = TinyDecoder(ModelConfig(vocab_size=31, hidden_size=32, intermediate_size=48,
                                    num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=24))
    prompt = torch.tensor([[1, 7, 12, 4]])
    expected_logits = model(prompt)
    expected_tokens = generate(model, prompt, 8)
    model.to(device)
    tokens = prompt.to(device)
    actual_logits = model(tokens).cpu()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=2e-4, atol=2e-5)
    for use_cache in (True, False):
        actual = generate(model, tokens, 8, use_cache=use_cache)
        torch.testing.assert_close(actual.cpu(), expected_tokens, rtol=0, atol=0)
    cache = model.new_cache(capacity=12)
    chunked = torch.cat([model(tokens[:, :2], cache), model(tokens[:, 2:], cache)], dim=1)
    torch.testing.assert_close(chunked.cpu(), actual_logits, rtol=2e-4, atol=2e-5)
    synchronize(device)
    return {"passed": True, "runtime": runtime_metadata(device),
            "max_cpu_cuda_logit_error": (actual_logits - expected_logits).abs().max().item(),
            "cached_uncached_cpu_token_agreement": True, "chunked_prefill_agreement": True,
            "generated_ids": expected_tokens[0, prompt.shape[1]:].tolist(),
            "rtol": 2e-4, "atol": 2e-5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"kind": "cuda_preflight", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "torch": torch.__version__, "python": platform.python_version(),
              "platform": platform.platform(), **source_metadata(),
              "scope": "Tiny random-weight engine numerical checks on one CUDA device versus CPU. Not a pretrained-model or performance validation."}
    try:
        report.update(check())
    except (ValueError, RuntimeError, AssertionError) as error:
        report.update(passed=False, error=str(error))
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
