"""Predict each shape/stride before reading the output on CPU or GPU."""

import argparse

import torch

from devices import resolve_device


def describe(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name:16} shape={tuple(tensor.shape)}, stride={tensor.stride()}, "
        f"contiguous={tensor.is_contiguous()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    print("Device:", device)
    # Think of a tensor as storage plus a shape, strides, and a storage offset.
    x = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)
    describe("x", x)
    transposed = x.transpose(1, 2)
    describe("transpose", transposed)
    packed = transposed.contiguous()
    describe("contiguous copy", packed)
    print("Same logical values:", torch.equal(transposed, packed))

    # Mutating a view demonstrates shared storage without pointer arithmetic.
    transposed[0, 0, 0] = -1
    print("After view edit, x[0,0,0]:", x[0, 0, 0].item())
    print("Independent copy [0,0,0]:", packed[0, 0, 0].item())

    # Broadcast a channel vector over batch and sequence dimensions.
    channel_bias = torch.arange(4, dtype=torch.float32, device=device)
    describe("x + bias", x + channel_bias)

    # Leading dimensions are batches of independent matrix multiplications.
    generator = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(2, 3, 5, 4, generator=generator).to(device)
    k = torch.randn(2, 3, 5, 4, generator=generator).to(device)
    describe("Q [B,H,T,D]", q)
    describe("K transposed", k.transpose(-2, -1))
    describe("Q @ K^T", q @ k.transpose(-2, -1))
    print("Explain: why does attention create a T-by-T matrix for each head?")


if __name__ == "__main__":
    main()
