"""Prepare reproducible VI GIPFormer RNN-T train/dev JSONL inputs.

The V1 manifest represents each clean utterance once per noisy augmentation.
For adaptation we retain every unique noisy WAV and one copy of each physical
clean WAV.  Test rows are deliberately never emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_vietnamese(row: dict) -> bool:
    language = str(row.get("language", "")).strip()
    return not language or language == "vi"


def resolve_audio(root: Path, kind: str, name: str) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = root / kind / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing {kind} WAV: {path}")
    return path.resolve()


def prepare_rows(manifest: Path, split: str) -> list[dict]:
    root = manifest.parent
    records: list[dict] = []
    seen_clean: set[str] = set()
    seen_any: set[str] = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {manifest}:{number}") from exc
        if row.get("split") != split or not is_vietnamese(row):
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError(f"Missing transcript in {manifest}:{number}")
        noisy_name = str(row.get("audio", "")).strip()
        clean_name = str(row.get("clean_audio", "")).strip()
        if not noisy_name or not clean_name:
            raise ValueError(f"Missing clean/noisy pairing in {manifest}:{number}")
        for kind, name in (("noisy", noisy_name), ("clean", clean_name)):
            if kind == "clean" and name in seen_clean:
                continue
            path = resolve_audio(root, kind, name)
            if str(path) in seen_any:
                raise ValueError(f"Duplicate physical WAV in {split}: {path}")
            seen_any.add(str(path))
            if kind == "clean":
                seen_clean.add(name)
            item = {
                    "id": f"{split}:{kind}:{path.name}",
                    "audio_path": str(path),
                    "text": text,
                    "split": split,
                    "variant": kind,
                    "source_audio": name,
                    "source_manifest_line": number,
                }
            # Preserve an existing manifest duration when available.  The
            # trainer uses it for duration bucketing without rescanning every
            # WAV on Google Drive.
            duration = row.get("duration_s", row.get("duration_seconds", row.get("duration")))
            if duration is not None:
                item["duration_s"] = float(duration)
            records.append(item)
    if not records:
        raise ValueError(f"No VI rows selected for {split}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train = prepare_rows(args.manifest, "train")
    dev = prepare_rows(args.manifest, "dev")
    train_paths = {item["audio_path"] for item in train}
    dev_paths = {item["audio_path"] for item in dev}
    overlap = train_paths & dev_paths
    if overlap:
        raise ValueError(f"Train/dev audio leakage: {len(overlap)} WAV files")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("dev.jsonl", dev)):
        (args.output_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    train_texts = Counter(row["text"] for row in train)
    dev_texts = Counter(row["text"] for row in dev)
    report = {
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "test_split_included": False,
        "train": {"records": len(train), "variants": dict(Counter(row["variant"] for row in train))},
        "dev": {"records": len(dev), "variants": dict(Counter(row["variant"] for row in dev))},
        "train_dev_audio_overlap": 0,
        "exact_text_overlap": len(set(train_texts) & set(dev_texts)),
        "format": "absolute WAV path + transcript for GIPFormer RNN-T",
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
