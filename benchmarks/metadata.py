"""Identify the exact local source even when experiments precede a commit."""

import hashlib
from pathlib import Path
import subprocess


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
