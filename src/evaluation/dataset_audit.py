"""Logical and physical audit for clean/noisy OneVoice audio manifests."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest line {line_number} is not an object")
            rows.append(row)
    return rows


def audit_audio_manifest(
    manifest_path: str | Path,
    physical: bool = True,
    expected_noisy: int | None = None,
    expected_clean: int | None = None,
    language: str | None = None,
    min_speakers: int | None = None,
    require_realized_snr: bool = False,
) -> dict:
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    rows = _read_jsonl(manifest)
    if language is not None:
        rows = [row for row in rows if row.get("language", "vi") == language]
        if not rows:
            raise ValueError(f"No rows found for language={language}")
    root = manifest.parent
    noisy_names = [str(row.get("audio", "")) for row in rows]
    clean_names = [str(row.get("clean_audio", "")) for row in rows]
    errors: list[str] = []

    if any(not name for name in noisy_names):
        errors.append("missing noisy audio filename")
    if any(not name for name in clean_names):
        errors.append("missing clean audio filename")
    duplicate_noisy = [name for name, count in Counter(noisy_names).items() if count > 1]
    if duplicate_noisy:
        errors.append(f"duplicate noisy filenames: {len(duplicate_noisy)}")
    if any(not str(row.get("text", "")).strip() for row in rows):
        errors.append("missing transcript")
    if any(not str(row.get("translation", "")).strip() for row in rows):
        errors.append("missing translation")
    if any(not str(row.get("split", "")).strip() for row in rows):
        errors.append("missing split")
    if any(
        not str(row.get("speaker_id", "")).strip()
        or str(row.get("speaker_id", "")).casefold() == "unknown"
        for row in rows
    ):
        errors.append("missing or unknown speaker identity")
    if require_realized_snr and any(row.get("realized_snr_db") is None for row in rows):
        errors.append("missing realized_snr_db")

    split_by_clean: dict[str, set[str]] = defaultdict(set)
    split_by_text: dict[str, set[str]] = defaultdict(set)
    split_by_pattern: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row.get("split", ""))
        split_by_clean[str(row.get("clean_audio", ""))].add(split)
        split_by_text[str(row.get("text", "")).strip()].add(split)
        pattern = str(row.get("frame_pattern_id", "")).strip()
        if pattern:
            split_by_pattern[pattern].add(split)
    clean_split_leaks = sum(len(splits) > 1 for splits in split_by_clean.values())
    text_split_leaks = sum(len(splits) > 1 for splits in split_by_text.values())
    if clean_split_leaks:
        errors.append(f"clean pairs crossing splits: {clean_split_leaks}")
    if text_split_leaks:
        errors.append(f"exact text crossing splits: {text_split_leaks}")
    pattern_split_leaks = sum(len(splits) > 1 for splits in split_by_pattern.values())
    if pattern_split_leaks:
        errors.append(f"frame patterns crossing splits: {pattern_split_leaks}")

    missing_noisy: list[str] = []
    missing_clean: list[str] = []
    invalid_audio: list[str] = []
    invalid_pairs: list[str] = []
    if physical:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("soundfile is required for physical audio audit") from exc
        noisy_info = {}
        clean_info = {}
        for name in sorted(set(noisy_names)):
            path = root / "noisy" / name
            if not path.is_file():
                missing_noisy.append(name)
                continue
            try:
                info = sf.info(path)
                if info.samplerate != 16000 or info.frames <= 0 or info.channels != 1:
                    invalid_audio.append(name)
                else:
                    noisy_info[name] = info
            except RuntimeError:
                invalid_audio.append(name)
        for name in sorted(set(clean_names)):
            path = root / "clean" / name
            if not path.is_file():
                missing_clean.append(name)
                continue
            try:
                info = sf.info(path)
                if info.samplerate != 16000 or info.frames <= 0 or info.channels != 1:
                    invalid_audio.append(name)
                else:
                    clean_info[name] = info
            except RuntimeError:
                invalid_audio.append(name)
        if missing_noisy:
            errors.append(f"missing noisy WAV files: {len(missing_noisy)}")
        if missing_clean:
            errors.append(f"missing clean WAV files: {len(missing_clean)}")
        if invalid_audio:
            errors.append(f"invalid WAV files: {len(invalid_audio)}")
        for row in rows:
            noisy = noisy_info.get(str(row.get("audio", "")))
            clean = clean_info.get(str(row.get("clean_audio", "")))
            if noisy and clean and abs(noisy.frames - clean.frames) > 1:
                invalid_pairs.append(str(row.get("audio", "")))
        if invalid_pairs:
            errors.append(f"clean/noisy duration mismatch: {len(invalid_pairs)}")

    unique_noisy = len(set(noisy_names))
    unique_clean = len(set(clean_names))
    if expected_noisy is not None and unique_noisy != expected_noisy:
        errors.append(f"expected {expected_noisy} noisy files, found {unique_noisy}")
    if expected_clean is not None and unique_clean != expected_clean:
        errors.append(f"expected {expected_clean} clean files, found {unique_clean}")
    speaker_count = len(
        {str(row.get("speaker_id", "")) for row in rows if str(row.get("speaker_id", "")).strip()}
    )
    if min_speakers is not None and speaker_count < min_speakers:
        errors.append(f"expected at least {min_speakers} speakers, found {speaker_count}")

    return {
        "manifest": str(manifest.resolve()),
        "rows": len(rows),
        "unique_noisy": unique_noisy,
        "unique_clean": unique_clean,
        "splits": dict(Counter(str(row.get("split", "unknown")) for row in rows)),
        "speakers": dict(Counter(str(row.get("speaker_id", "unknown")) for row in rows)),
        "noise_types": dict(Counter(str(row.get("noise_type", "unknown")) for row in rows)),
        "snr_db": dict(Counter(str(row.get("snr_db", "unknown")) for row in rows)),
        "missing_noisy": missing_noisy,
        "missing_clean": missing_clean,
        "invalid_audio": invalid_audio,
        "invalid_pairs": invalid_pairs,
        "frame_pattern_split_leaks": pattern_split_leaks,
        "errors": errors,
        "passed": not errors,
    }
