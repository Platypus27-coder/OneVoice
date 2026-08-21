"""Recover the usable portion of a missing OneVoice V1 audio manifest.

V1 filenames retain the utterance ID and noisy variant, so transcript, split,
domain and clean/noisy pairing can be recovered from utterances_all.csv. The
old generator selected speaker, noise, SNR and reverb randomly without encoding
them in filenames; those fields are deliberately marked unrecoverable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


NOISY_PATTERN = re.compile(r"^(?P<utterance_id>.+)_n(?P<variant>\d+)$")
UNRECOVERABLE = "unrecoverable_v1"


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {str(row.get("utterance_id", "")).strip(): row for row in rows}
    indexed.pop("", None)
    if not indexed:
        raise ValueError(f"No utterance_id rows found in {path}")
    return indexed


def recover(dataset_root: Path, metadata_csv: Path) -> tuple[list[dict], dict]:
    clean_dir = dataset_root / "clean"
    noisy_dir = dataset_root / "noisy"
    if not clean_dir.is_dir() or not noisy_dir.is_dir():
        raise FileNotFoundError("Dataset root must contain clean/ and noisy/")
    metadata = load_metadata(metadata_csv)
    entries: list[dict] = []
    errors: list[str] = []
    used_clean: set[str] = set()
    clean_names = {path.name for path in clean_dir.glob("*.wav")}
    noisy_paths = sorted(noisy_dir.glob("*.wav"))
    print(
        f"[Manifest recovery] indexed {len(clean_names)} clean and "
        f"{len(noisy_paths)} noisy WAV filenames",
        flush=True,
    )

    for index, noisy_path in enumerate(noisy_paths, start=1):
        match = NOISY_PATTERN.fullmatch(noisy_path.stem)
        if match is None:
            errors.append(f"unrecognized noisy filename: {noisy_path.name}")
            continue
        uid = match.group("utterance_id")
        row = metadata.get(uid)
        if row is None:
            errors.append(f"utterance not found in metadata: {uid}")
            continue
        clean_name = f"{uid}_clean.wav"
        if clean_name not in clean_names:
            errors.append(f"missing paired clean WAV: {clean_name}")
            continue
        used_clean.add(clean_name)
        entries.append(
            {
                "utterance_id": uid,
                "pair_id": row.get("pair_id", ""),
                "frame_pattern_id": row.get("frame_pattern_id", uid),
                "audio": noisy_path.name,
                "clean_audio": clean_name,
                "language": "vi",
                "text": row.get("vi", ""),
                "translation": row.get("en", ""),
                "domain": row.get("domain", "unknown"),
                "intent": row.get("intent", "unknown"),
                "risk_level": row.get("risk_level", "unknown"),
                "split": row.get("split", "train"),
                "speaker_id": UNRECOVERABLE,
                "noise_type": UNRECOVERABLE,
                "snr_db": None,
                "reverb": None,
                "rir_id": UNRECOVERABLE,
                "synthetic_speech": True,
                "synthetic_noise_mix": True,
                "sample_rate": 16000,
                "manifest_provenance": "recovered_from_v1_filenames",
                "noisy_variant": int(match.group("variant")),
            }
        )
        if index % 1000 == 0 or index == len(noisy_paths):
            print(
                f"[Manifest recovery] paired {index}/{len(noisy_paths)} noisy WAVs",
                flush=True,
            )

    unused_clean = sorted(clean_names - used_clean)
    if unused_clean:
        errors.append(f"clean WAVs without noisy pairs: {len(unused_clean)}")
    report = {
        "status": "PARTIAL" if entries else "FAILED",
        "entries": len(entries),
        "clean_files": len(clean_names),
        "noisy_files": len(noisy_paths),
        "unrecoverable_fields": ["speaker_id", "noise_type", "snr_db", "reverb", "rir_id"],
        "errors": errors,
    }
    if not entries:
        raise ValueError(json.dumps(report, ensure_ascii=False))
    return entries, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover a missing V1 manifest from WAV filenames")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.dataset_root / "manifest.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")
    entries, report = recover(args.dataset_root, args.metadata_csv)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in entries),
        encoding="utf-8",
    )
    temporary.replace(output)
    report_path = output.with_name("manifest_recovery_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "manifest": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
