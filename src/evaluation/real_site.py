"""Validation for the consent-aware, group-isolated real-site pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED = {
    "utterance_id", "audio", "language", "transcript", "translation",
    "site_id", "session_id", "speaker_id", "split", "domain", "intent",
    "risk_level", "consent_recorded",
}


def _holdout_digest(rows: list[dict]) -> str:
    fixed = [row for row in rows if row.get("split") == "test"]
    fixed.sort(key=lambda row: str(row.get("utterance_id", "")))
    encoded = json.dumps(fixed, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_real_site_manifest(
    manifest_path: str | Path,
    physical: bool = True,
    holdout_lock_path: str | Path | None = None,
    final_gate: bool = False,
) -> dict:
    manifest = Path(manifest_path)
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    if final_gate and not 500 <= len(rows) <= 2000:
        errors.append(f"final pilot requires 500-2000 utterances, found {len(rows)}")
    ids: list[str] = []
    groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    valid_languages = {"vi", "en"}
    valid_splits = {"train", "dev", "test"}
    valid_risks = {"low", "medium", "high", "critical"}

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"line {index}: row is not an object")
            continue
        missing = sorted(REQUIRED - row.keys())
        if missing:
            errors.append(f"line {index}: missing {', '.join(missing)}")
            continue
        ids.append(str(row["utterance_id"]))
        if row["language"] not in valid_languages:
            errors.append(f"line {index}: invalid language")
        if row["split"] not in valid_splits:
            errors.append(f"line {index}: invalid split")
        if row["risk_level"] not in valid_risks:
            errors.append(f"line {index}: invalid risk_level")
        if row["consent_recorded"] is not True:
            errors.append(f"line {index}: consent_recorded must be true")
        for field in ("utterance_id", "audio", "transcript", "translation", "site_id", "session_id", "speaker_id"):
            if not str(row[field]).strip():
                errors.append(f"line {index}: empty {field}")
        group = (str(row["site_id"]), str(row["session_id"]), str(row["speaker_id"]))
        groups[group].add(str(row["split"]))

    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate utterance IDs: {len(duplicates)}")
    leaking = [group for group, splits in groups.items() if len(splits) > 1]
    if leaking:
        errors.append(f"speaker/session/site groups crossing splits: {len(leaking)}")

    invalid_audio: list[str] = []
    missing_audio: list[str] = []
    if physical:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("soundfile is required for physical real-site audit") from exc
        for row in rows:
            audio_path = manifest.parent / str(row.get("audio", ""))
            if not audio_path.is_file():
                missing_audio.append(str(row.get("audio", "")))
                continue
            try:
                info = sf.info(audio_path)
                if info.frames <= 0 or info.channels != 1 or info.samplerate != 16000:
                    invalid_audio.append(str(row["audio"]))
            except RuntimeError:
                invalid_audio.append(str(row["audio"]))
        if missing_audio:
            errors.append(f"missing WAV files: {len(missing_audio)}")
        if invalid_audio:
            errors.append(f"invalid WAV files: {len(invalid_audio)}")

    digest = _holdout_digest(rows)
    if final_gate and not any(row.get("split") == "test" for row in rows):
        errors.append("final pilot requires a non-empty fixed test holdout")
    lock_matches = None
    if holdout_lock_path:
        lock_file = Path(holdout_lock_path)
        if not lock_file.is_file():
            errors.append(f"holdout lock not found: {lock_file}")
            lock_matches = False
        else:
            locked = json.loads(lock_file.read_text(encoding="utf-8"))
            lock_matches = locked.get("test_sha256") == digest
            if not lock_matches:
                errors.append("test holdout differs from its lock")

    return {
        "manifest": str(manifest.resolve()),
        "rows": len(rows),
        "splits": dict(Counter(str(row.get("split", "unknown")) for row in rows)),
        "sites": len({str(row.get("site_id", "")) for row in rows}),
        "sessions": len({str(row.get("session_id", "")) for row in rows}),
        "speakers": len({str(row.get("speaker_id", "")) for row in rows}),
        "test_sha256": digest,
        "holdout_lock_matches": lock_matches,
        "missing_audio": missing_audio,
        "invalid_audio": invalid_audio,
        "errors": errors,
        "passed": not errors,
    }


def write_holdout_lock(manifest_path: str | Path, output_path: str | Path) -> None:
    report = audit_real_site_manifest(manifest_path, physical=False)
    if not report["passed"]:
        raise ValueError("Cannot lock an invalid manifest: " + "; ".join(report["errors"]))
    Path(output_path).write_text(
        json.dumps(
            {
                "manifest": str(Path(manifest_path).resolve()),
                "test_sha256": report["test_sha256"],
                "test_rows": report["splits"].get("test", 0),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
