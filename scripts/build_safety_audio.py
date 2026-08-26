"""Pre-generate reviewed local safety audio with checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from tts.tts_engine import TTSEngine


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(path: Path, payload: dict) -> None:
    """Atomically write either a resumable partial or deployable final manifest."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".json") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary_manifest = Path(handle.name)
    temporary_manifest.replace(path)


def load_resume_entries(manifest: Path, source_sha256: str, approval_id: str) -> dict[tuple[str, str], dict]:
    """Return checksum-verified completed entries from an equivalent prior run."""
    if not manifest.is_file():
        return {}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("source_sha256") != source_sha256 or payload.get("approval_id") != approval_id:
        raise RuntimeError(
            "Existing safety-audio manifest belongs to another source CSV or approval ID; "
            "choose a new output directory rather than mixing reviewed assets."
        )
    entries = {}
    for entry in payload.get("entries", []):
        key = (str(entry.get("safety_id", "")), str(entry.get("direction", "")))
        path = manifest.parent / str(entry.get("path", ""))
        if key[0] and key[1] and path.is_file() and hash_file(path) == entry.get("sha256"):
            entries[key] = entry
    return entries


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
    parser.add_argument("--resume", action="store_true", help="Reuse checksum-verified WAVs from an interrupted equivalent run")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_csv = Path(args.safety_csv)
    source_sha256 = hash_file(source_csv)
    manifest_path = output / "manifest.json"
    partial_path = output / "manifest.partial.json"
    resume_path = manifest_path if manifest_path.is_file() else partial_path
    resumed = load_resume_entries(resume_path, source_sha256, args.approval_id) if args.resume else {}
    if (manifest_path.is_file() or partial_path.is_file()) and not args.resume:
        raise FileExistsError(
            f"{manifest_path} or {partial_path} already exists. Pass --resume to reuse verified WAVs, "
            "or choose a new output directory."
        )
    engines = {}
    entries = []
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
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
    def manifest_payload() -> dict:
        return {
            "schema_version": 2,
            "source_csv": str(source_csv.resolve()),
            "source_sha256": source_sha256,
            "approval_id": args.approval_id,
            "entries": entries,
        }

    for direction in ("vi2en", "en2vi"):
        tts = TTSEngine(config, profile=args.profile, offline=False)
        tts.load(direction)
        engines[direction] = tts
        for row in rows:
            key = (row["safety_id"], direction)
            if key in resumed:
                entries.append(resumed[key])
                print(f"[Safety audio] resumed {key[0]}/{key[1]}")
                continue
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
            # Do not publish partial audio to the runtime. The sibling partial
            # manifest is only for safe resume after Colab interruption.
            write_manifest(partial_path, manifest_payload())
    write_manifest(manifest_path, manifest_payload())
    partial_path.unlink(missing_ok=True)
    print(f"Generated or resumed {len(entries)} verified safety audio files in {output}")


if __name__ == "__main__":
    main()
