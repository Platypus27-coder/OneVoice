"""Write a reproducible, optionally audio-deduplicated ASR manifest subset.

V1 clean audio occurs once for every noisy augmentation in the logical
manifest.  An ASR clean benchmark must evaluate each physical WAV once; this
utility produces that canonical subset without modifying the source manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--audio", choices=("clean", "noisy"), required=True)
    parser.add_argument("--dedupe-audio", action="store_true")
    args = parser.parse_args()

    selected: list[dict] = []
    seen_audio: set[str] = set()
    for line_number, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {args.manifest}:{line_number}") from exc
        if row.get("split") != args.split or row.get("language") != args.language:
            continue
        audio_name = str(row.get("clean_audio") if args.audio == "clean" else row.get("audio", "")).strip()
        if not audio_name:
            raise ValueError(f"Missing {args.audio} audio filename in {args.manifest}:{line_number}")
        if args.dedupe_audio and audio_name in seen_audio:
            continue
        seen_audio.add(audio_name)
        selected.append(row)

    if not selected:
        raise ValueError("No rows match the requested subset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    summary = {
        "source_manifest": str(args.manifest),
        "source_sha256": _sha256(args.manifest),
        "split": args.split,
        "language": args.language,
        "audio": args.audio,
        "dedupe_audio": args.dedupe_audio,
        "rows": len(selected),
        "unique_audio": len(seen_audio),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
