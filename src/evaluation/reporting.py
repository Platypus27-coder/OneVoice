"""Reproducible benchmark run metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_path(path: Path) -> tuple[str | None, int]:
    if path.is_file():
        return _hash_file(path), 1
    if not path.is_dir():
        return None, 0
    digest = hashlib.sha256()
    count = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_hash_file(child)))
        count += 1
    return digest.hexdigest(), count


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def create_run_manifest(
    output_path: str | Path,
    command: str,
    inputs: list[str | Path] = (),
    metadata: dict | None = None,
) -> dict:
    destination = Path(output_path)
    repo_root = Path(__file__).resolve().parents[2]
    dependencies = {}
    for name in ("numpy", "torch", "transformers", "sherpa-onnx", "funasr-onnx"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    payload = {
        "created_at_unix": time.time(),
        "command": command,
        "git_revision": _git_revision(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "dependencies": dependencies,
        "inputs": [],
        "metadata": metadata or {},
        "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    for value in inputs:
        path = Path(value)
        digest, file_count = _hash_path(path)
        payload["inputs"].append(
            {
                "path": str(path.resolve()),
                "sha256": digest,
                "file_count": file_count,
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
