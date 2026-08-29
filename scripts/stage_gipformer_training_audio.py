"""Stage prepared GIPFormer train/dev WAVs from Drive onto local Colab SSD.

Training directly through the Google Drive FUSE mount makes each epoch
I/O-bound. This script copies only the already-approved train/dev files to an
ephemeral local directory, rewrites JSONL manifests to those local paths, and
never reads or emits a test record. It is safe to rerun: valid staged files are
reused and only missing files are copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        audio = Path(str(row.get("audio_path", "")))
        if not audio.is_file() or not str(row.get("text", "")).strip():
            raise ValueError(f"Invalid prepared row at {path}:{line_number}")
        if str(row.get("split", "")) == "test":
            raise ValueError(f"Test record appeared in prepared input: {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def cache_target(source: Path, cache_root: Path) -> Path:
    key = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:20]
    return cache_root / "audio" / f"{key}_{source.name}"


def stage_one(source: Path, target: Path) -> tuple[bool, int]:
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        return False, target.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    return True, target.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=3.0)
    args = parser.parse_args()
    if args.workers < 1 or args.min_free_gb < 0:
        parser.error("workers must be positive and min-free-gb cannot be negative")

    train_rows = read_rows(args.train)
    dev_rows = read_rows(args.dev)
    source_paths = {Path(str(row["audio_path"])).resolve() for row in train_rows + dev_rows}
    total_bytes = sum(path.stat().st_size for path in source_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output_dir).free
    required_bytes = total_bytes + int(args.min_free_gb * 1024**3)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Local disk is too small: need {(required_bytes / 1024**3):.2f} GB free, "
            f"have {(free_bytes / 1024**3):.2f} GB"
        )

    targets = {source: cache_target(source, args.output_dir) for source in source_paths}
    copied = 0
    reused = 0
    print(
        f"Staging {len(source_paths)} unique train/dev WAVs "
        f"({total_bytes / 1024**3:.2f} GB) to {args.output_dir}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(stage_one, source, targets[source]): source for source in source_paths}
        for index, future in enumerate(as_completed(futures), 1):
            changed, _ = future.result()
            copied += int(changed)
            reused += int(not changed)
            if index % 250 == 0 or index == len(futures):
                print(f"[stage] {index}/{len(futures)}; copied={copied}; reused={reused}", flush=True)

    def rewrite(rows: list[dict], name: str) -> None:
        rewritten = []
        for row in rows:
            source = Path(str(row["audio_path"])).resolve()
            local = targets[source]
            if not local.is_file():
                raise RuntimeError(f"Staged WAV missing after copy: {local}")
            rewritten.append({**row, "audio_path": str(local), "staged_from": str(source)})
        (args.output_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rewritten),
            encoding="utf-8",
        )

    rewrite(train_rows, "train.jsonl")
    rewrite(dev_rows, "dev.jsonl")
    report = {
        "source": {"train": str(args.train.resolve()), "dev": str(args.dev.resolve())},
        "source_sha256": {"train": sha256(args.train), "dev": sha256(args.dev)},
        "local_output": str(args.output_dir.resolve()),
        "unique_audio": len(source_paths),
        "copied": copied,
        "reused": reused,
        "total_bytes": total_bytes,
        "test_split_included": False,
    }
    (args.output_dir / "stage_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
