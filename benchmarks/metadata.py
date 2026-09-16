"""Identify the exact local source even when experiments precede a commit."""

import hashlib
from pathlib import Path
import subprocess
import platform

import torch


def runtime_metadata(device):
    """Capture the selected accelerator, not the host CPU's model name."""
    report = {"device": str(device), "cuda_runtime": torch.version.cuda,
              "float32_matmul_precision": torch.get_float32_matmul_precision()}
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        report["address_space_limit_bytes"] = {"soft": soft, "hard": hard}
    except (ImportError, AttributeError):
        report["address_space_limit_bytes"] = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report.update(hardware=props.name, device_index=torch.cuda.current_device() if device.index is None else device.index,
                      compute_capability=[props.major, props.minor], total_memory_bytes=props.total_memory,
                      compiled_architectures=torch.cuda.get_arch_list(),
                      matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10)
            report["driver_version"] = driver.stdout.strip().splitlines()[0] if driver.returncode == 0 and driver.stdout.strip() else None
        except (OSError, subprocess.TimeoutExpired):
            report["driver_version"] = None
    elif platform.system() == "Darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        report["hardware"] = chip.stdout.strip() if chip.returncode == 0 else None
    else:
        report["hardware"] = platform.processor()
    return report


def source_metadata():
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    paths = sorted(list((root / "src").rglob("*.py")) + list((root / "benchmarks").glob("*.py"))
                   + [root / "pyproject.toml", root / "uv.lock"])
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
