"""Runtime device selection and explicit benchmark synchronization."""

import torch


def available_devices() -> list[str]:
    """Backends available to this process, including visible CUDA devices."""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


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
        if torch.version.cuda is None:
            raise ValueError("This PyTorch build has no CUDA support. On the documented NVIDIA/Linux "
                             "setup, run uv sync --extra cuda124 and keep --extra cuda124 on uv run commands.")
        raise ValueError("CUDA-enabled PyTorch is installed but no usable NVIDIA GPU is visible. "
                         "Check nvidia-smi, driver compatibility, and CUDA_VISIBLE_DEVICES.")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    """Wait at benchmark boundaries, not after every production operation."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
