"""Validate a FunASR resume checkpoint and quarantine it if it is corrupted.

FunASR resumes exclusively from ``<output-dir>/model.pt``.  A Colab runtime
can stop while Google Drive is writing that large file; moving an unreadable
file aside lets the next run start cleanly while retaining it for inspection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


def check_checkpoint(output_dir: Path, loader: Callable[[Path], object]) -> dict:
    checkpoint = output_dir / "model.pt"
    if not checkpoint.is_file():
        return {"status": "absent", "checkpoint": str(checkpoint)}
    try:
        loader(checkpoint)
    except Exception as exc:
        return {"status": "corrupt", "checkpoint": str(checkpoint), "error": str(exc)}
    return {"status": "valid", "checkpoint": str(checkpoint)}


def quarantine_checkpoint(output_dir: Path, result: dict) -> dict:
    if result["status"] != "corrupt":
        return result
    checkpoint = Path(result["checkpoint"])
    destination_dir = output_dir / "corrupt_checkpoints"
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = destination_dir / f"model.pt.{timestamp}.corrupt"
    shutil.move(str(checkpoint), str(destination))
    return {**result, "quarantined_to": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--quarantine-corrupt", action="store_true")
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to validate a training checkpoint") from exc

    result = check_checkpoint(
        args.output_dir,
        lambda checkpoint: torch.load(checkpoint, map_location="cpu", weights_only=False),
    )
    if result["status"] == "corrupt" and args.quarantine_corrupt:
        result = quarantine_checkpoint(args.output_dir, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "corrupt" and not args.quarantine_corrupt:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
