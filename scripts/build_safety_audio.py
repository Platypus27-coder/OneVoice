"""Pre-generate reviewed local safety audio with checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from tts.tts_engine import TTSEngine


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--safety-csv", default="data/onevoice_construction_v2/safety_fast_path.csv"
    )
    parser.add_argument("--output", default="artifacts/safety_audio")
    parser.add_argument("--profile", choices=["development", "premium"], default="development")
    parser.add_argument("--approval-id", required=True, help="Safety/voice approval record ID")
    parser.add_argument("--required-review-status", default="approved")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    engines = {}
    entries = []
    with open(args.safety_csv, "r", encoding="utf-8-sig", newline="") as handle:
        candidates = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("fixed_translation_candidate", "")).casefold() == "true"
        ]
    unreviewed = [
        row["safety_id"]
        for row in candidates
        if row.get("review_status") != args.required_review_status
    ]
    if unreviewed:
        raise RuntimeError(
            f"Refusing to generate {len(unreviewed)} unapproved safety phrases; "
            f"required review_status={args.required_review_status}"
        )
    rows = candidates
    for direction in ("vi2en", "en2vi"):
        tts = TTSEngine(config, profile=args.profile, offline=False)
        tts.load(direction)
        engines[direction] = tts
        for row in rows:
            text = row["en"] if direction == "vi2en" else row["vi"]
            audio, sample_rate = tts.synthesize(text, direction)
            if tts.is_silence(audio):
                raise RuntimeError(f"TTS returned silence for {row['safety_id']} ({direction})")
            filename = f"{row['safety_id']}_{direction}.wav"
            path = output / filename
            sf.write(path, audio, sample_rate)
            entries.append(
                {
                    "safety_id": row["safety_id"],
                    "direction": direction,
                    "path": filename,
                    "sample_rate": sample_rate,
                    "sha256": hash_file(path),
                    "engine_profile": args.profile,
                    "engine": tts.engine_name(direction),
                    "approval_id": args.approval_id,
                }
            )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source_csv": str(Path(args.safety_csv).resolve()),
                "source_sha256": hash_file(Path(args.safety_csv)),
                "approval_id": args.approval_id,
                "entries": entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(entries)} verified safety audio files in {output}")


if __name__ == "__main__":
    main()
