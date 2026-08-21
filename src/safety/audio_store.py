"""Verified pre-generated safety audio lookup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class SafetyAudioError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SafetyAudioStore:
    def __init__(
        self,
        manifest_path: str | Path,
        source_csv: str | Path | None = None,
    ):
        manifest = Path(manifest_path)
        if not manifest.is_file():
            raise SafetyAudioError(f"Safety audio manifest not found: {manifest}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) < 2:
            raise SafetyAudioError("Safety audio manifest schema_version 2 is required")
        if not str(payload.get("approval_id", "")).strip():
            raise SafetyAudioError("Safety audio manifest is missing approval_id")
        if source_csv is not None:
            source = Path(source_csv)
            expected_source = str(payload.get("source_sha256", "")).casefold()
            if not source.is_file() or not expected_source or _sha256(source) != expected_source:
                raise SafetyAudioError(
                    "Safety audio source CSV checksum does not match runtime safety data"
                )
        try:
            import soundfile as sf
        except ImportError as exc:
            raise SafetyAudioError(
                "soundfile is required to validate local safety audio"
            ) from exc
        self._entries: dict[tuple[str, str], tuple[Path, int]] = {}
        for row in payload.get("entries", []):
            path = Path(str(row.get("path", "")))
            if not path.is_absolute():
                path = manifest.parent / path
            if not path.is_file():
                raise SafetyAudioError(f"Safety audio file not found: {path}")
            expected = str(row.get("sha256", "")).casefold()
            if not expected or _sha256(path) != expected:
                raise SafetyAudioError(f"Safety audio checksum mismatch: {path}")
            key = (str(row.get("safety_id", "")), str(row.get("direction", "")))
            if not all(key) or key in self._entries:
                raise SafetyAudioError(f"Invalid or duplicate safety audio key: {key}")
            if int(row.get("sample_rate", 0)) <= 0:
                raise SafetyAudioError(f"Invalid sample rate for safety audio: {path}")
            try:
                info = sf.info(path)
            except RuntimeError as exc:
                raise SafetyAudioError(f"Undecodable safety audio: {path}") from exc
            if (
                info.frames <= 0
                or info.channels != 1
                or info.samplerate != int(row["sample_rate"])
            ):
                raise SafetyAudioError(f"Invalid safety audio format: {path}")
            self._entries[key] = (path, int(row.get("sample_rate", 0)))

    def get(self, safety_id: str, direction: str) -> tuple[np.ndarray, int] | None:
        entry = self._entries.get((safety_id, direction))
        if not entry:
            return None
        try:
            import soundfile as sf
        except ImportError as exc:
            raise SafetyAudioError("soundfile is required to load safety audio") from exc
        path, expected_rate = entry
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if expected_rate and sample_rate != expected_rate:
            raise SafetyAudioError(
                f"Safety audio sample-rate mismatch for {path}: {sample_rate} != {expected_rate}"
            )
        return audio.astype(np.float32), sample_rate
