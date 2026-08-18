"""Runtime device selection and explicit benchmark synchronization."""

import torch


def resolve_device(name: str = "auto") -> torch.device:
    if name not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError(f"Unknown device: {name}")
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this Mac/PyTorch build. Use --device cpu.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. It requires an NVIDIA GPU and CUDA-enabled PyTorch.")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    """Wait at benchmark boundaries, not after every production operation."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
