"""Reconcile reviewed safety phrases with the checksum-locked audio bundle.

This is a release gate, not an audio generator.  It compares the complete
196-row safety suite with the reviewed canonical rows (normally 126) and then
validates that every canonical ID has one verified WAV for each direction.
The report is written even when the gate fails so a Colab run leaves useful
evidence on Drive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary = Path(handle.name)
    temporary.replace(path)


def reconcile(
    safety_csv: Path,
    audio_manifest: Path,
    benchmark_csv: Path | None = None,
    expected_benchmark_rows: int = 196,
    expected_canonical_rows: int = 126,
    required_review_status: str = "approved",
    expected_approval_id: str | None = None,
) -> dict:
    """Return a machine-readable safety reconciliation report."""
    errors: list[str] = []
    source_rows = load_rows(safety_csv)
    candidates = [
        row
        for row in source_rows
        if str(row.get("fixed_translation_candidate", "")).casefold() == "true"
    ]
    source_ids = [str(row.get("safety_id", "")).strip() for row in candidates]
    duplicate_ids = sorted(
        key for key, count in Counter(source_ids).items() if not key or count > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate or empty canonical safety IDs: {duplicate_ids[:8]}")
    if len(source_rows) != expected_benchmark_rows:
        errors.append(
            f"safety suite row count {len(source_rows)} != expected {expected_benchmark_rows}"
        )
    if len(candidates) != expected_canonical_rows:
        errors.append(
            f"canonical safety row count {len(candidates)} != expected {expected_canonical_rows}"
        )
    for row in candidates:
        safety_id = row.get("safety_id", "<missing>")
        if row.get("review_status", "").strip() != required_review_status:
            errors.append(f"{safety_id}: review_status is not {required_review_status}")
        if not row.get("reviewer", "").strip() or not row.get("reviewed_at", "").strip():
            errors.append(f"{safety_id}: missing reviewer/reviewed_at")
        if not row.get("vi", "").strip() or not row.get("en", "").strip():
            errors.append(f"{safety_id}: missing VI or EN phrase")

    benchmark = load_rows(benchmark_csv or safety_csv)
    benchmark_ids = [str(row.get("safety_id", "")).strip() for row in benchmark]
    benchmark_id_set = {key for key in benchmark_ids if key}
    candidate_id_set = {key for key in source_ids if key}
    missing_in_benchmark = sorted(candidate_id_set - benchmark_id_set)
    if missing_in_benchmark:
        errors.append(f"canonical IDs missing from benchmark suite: {missing_in_benchmark[:8]}")

    try:
        payload = json.loads(audio_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        payload = {}
        errors.append(f"cannot read audio manifest: {exc}")
    expected_source_hash = sha256(safety_csv)
    if payload.get("source_sha256", "").casefold() != expected_source_hash:
        errors.append("audio manifest source_sha256 does not match reviewed safety CSV")
    approval_id = str(payload.get("approval_id", "")).strip()
    if not approval_id:
        errors.append("audio manifest is missing approval_id")
    if expected_approval_id and approval_id != expected_approval_id:
        errors.append(f"audio manifest approval_id {approval_id!r} != {expected_approval_id!r}")

    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    entry_map: dict[tuple[str, str], dict] = {}
    duplicate_entries: list[str] = []
    for entry in entries:
        key = (str(entry.get("safety_id", "")).strip(), str(entry.get("direction", "")).strip())
        if key in entry_map:
            duplicate_entries.append("/".join(key))
        entry_map[key] = entry
    if duplicate_entries:
        errors.append(f"duplicate audio manifest keys: {duplicate_entries[:8]}")

    try:
        import soundfile as sf
    except ImportError:
        sf = None
        errors.append("soundfile is required for physical safety WAV validation")

    manifest_root = audio_manifest.parent
    missing_keys: list[str] = []
    invalid_audio: list[str] = []
    verified_entries = 0
    for safety_id in sorted(candidate_id_set):
        for direction in ("vi2en", "en2vi"):
            key = (safety_id, direction)
            entry = entry_map.get(key)
            label = f"{safety_id}/{direction}"
            if not entry:
                missing_keys.append(label)
                continue
            path = Path(str(entry.get("path", "")))
            if not path.is_absolute():
                path = manifest_root / path
            expected_hash = str(entry.get("sha256", "")).casefold()
            try:
                valid = path.is_file() and len(expected_hash) == 64 and sha256(path) == expected_hash
                if sf is not None and valid:
                    info = sf.info(path)
                    valid = (
                        info.frames > 0
                        and info.channels == 1
                        and info.samplerate == int(entry.get("sample_rate", 0))
                    )
            except (OSError, RuntimeError, ValueError):
                valid = False
            if not valid:
                invalid_audio.append(label)
            else:
                verified_entries += 1
    if missing_keys:
        errors.append(f"missing VI/EN audio entries: {missing_keys[:8]}")
    if invalid_audio:
        errors.append(f"invalid or checksum-mismatched WAV entries: {invalid_audio[:8]}")
    expected_keys = {(sid, direction) for sid in candidate_id_set for direction in ("vi2en", "en2vi")}
    orphan_keys = sorted("/".join(key) for key in entry_map if key not in expected_keys)
    if orphan_keys:
        errors.append(f"orphan audio manifest entries: {orphan_keys[:8]}")

    report = {
        "schema_version": 1,
        "safety_csv": str(safety_csv.resolve()),
        "safety_csv_sha256": expected_source_hash,
        "audio_manifest": str(audio_manifest.resolve()),
        "approval_id": approval_id,
        "benchmark_rows": len(benchmark),
        "canonical_rows": len(candidates),
        "noncanonical_variant_rows": max(0, len(benchmark) - len(candidates)),
        "canonical_ids": len(candidate_id_set),
        "expected_audio_entries": len(expected_keys),
        "manifest_entries": len(entries),
        "verified_audio_entries": verified_entries,
        "missing_audio_entries": missing_keys,
        "invalid_audio_entries": invalid_audio,
        "orphan_audio_entries": orphan_keys,
        "review_status_counts": dict(Counter(str(row.get("review_status", "")).strip() for row in candidates)),
        "errors": errors,
        "passed": not errors,
        "scope": "internal-demo only; synthetic safety audio is not real-site evidence",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety-csv", required=True, type=Path)
    parser.add_argument("--audio-manifest", required=True, type=Path)
    parser.add_argument("--benchmark-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-benchmark-rows", type=int, default=196)
    parser.add_argument("--expected-canonical-rows", type=int, default=126)
    parser.add_argument("--required-review-status", default="approved")
    parser.add_argument("--approval-id")
    args = parser.parse_args()
    report = reconcile(
        args.safety_csv,
        args.audio_manifest,
        args.benchmark_csv,
        args.expected_benchmark_rows,
        args.expected_canonical_rows,
        args.required_review_status,
        args.approval_id,
    )
    atomic_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
