"""Device correctness checks against a CPU oracle; not performance benchmarks."""

import argparse
import sys

import torch
import torch.nn.functional as F

from attention import causal_attention
from devices import resolve_device


def reference(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)


def close(actual, expected):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-6)


def cpu_reference(q, k, v):
    return reference(q.cpu(), k.cpu(), v.cpu())


@torch.inference_mode()
def run_checks(operation, device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(17)

    def inputs(shape):
        # Seed on CPU then transfer so all backends receive identical values.
        return [torch.randn(shape, generator=generator).to(device) for _ in range(3)]

    # Awkward sizes avoid assuming powers of two or square feature dimensions.
    for shape in [(1, 1, 1, 3), (2, 3, 7, 5), (1, 2, 16, 8)]:
        q, k, v = inputs(shape)
        originals = [x.clone() for x in (q, k, v)]
        actual = operation(q, k, v)
        assert actual.device == q.device, "Output must stay on the input device"
        close(actual, cpu_reference(q, k, v))
        for after, before in zip((q, k, v), originals):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
    print("PASS: reference agreement across shapes; inputs unchanged")

    q, k, v = [x[..., ::2] for x in inputs((2, 3, 7, 10))]
    assert not q.is_contiguous()
    close(operation(q, k, v), cpu_reference(q, k, v))
    print("PASS: non-contiguous inputs")

    q, k, v = inputs((2, 3, 7, 5))
    expected = operation(q, k, v)
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[..., 4:, :] += 100
    changed_v[..., 4:, :] -= 100
    close(operation(q, changed_k, changed_v)[..., :4, :], expected[..., :4, :])
    print("PASS: changing future tokens cannot change earlier outputs")

    close(operation(q, k, v)[..., :1, :], v[..., :1, :])
    print("PASS: first token attends only to itself")

    # Zero queries produce uniform weights over each allowed prefix.
    zeros = torch.zeros_like(q)
    lengths = torch.arange(1, 8, dtype=v.dtype).reshape(1, 1, 7, 1)
    close(operation(zeros, k, v), v.cpu().cumsum(dim=-2) / lengths)
    print("PASS: uniform scores produce prefix means")

    # Stable softmax should handle large finite scores without NaNs/infinities.
    actual = operation(q * 100, k * 100, v)
    assert torch.isfinite(actual).all()
    close(actual, cpu_reference(q * 100, k * 100, v))
    print("PASS: large-score numerical stability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true", help="Check the PyTorch oracle only")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    print("Device:", device, "| Mode:", "PyTorch built-in" if args.reference else "explicit attention")
    try:
        run_checks(reference if args.reference else causal_attention, device)
    except AssertionError as error:
        print(f"CHECK FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
