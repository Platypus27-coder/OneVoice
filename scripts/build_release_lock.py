"""Materialize the OneVoice V2 release lock from a hashed artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_model(values: list[str]) -> dict:
    key, source, revision, license_name, direction, artifact_prefix = values
    if direction not in {"vi2en", "en2vi"}:
        raise ValueError(f"Invalid model direction: {direction}")
    if not revision.strip() or revision.strip().casefold() in {"main", "master", "unknown"}:
        raise ValueError(f"Model {key} requires an immutable revision, not {revision!r}")
    if not license_name.strip():
        raise ValueError(f"Model {key} requires license metadata")
    return {
        "key": key,
        "source": source,
        "revision": revision,
        "license": license_name,
        "direction": direction,
        "profiles": ["development"],
        "artifact_prefix": artifact_prefix.rstrip("/") + "/",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        action="append",
        nargs=6,
        required=True,
        metavar=("KEY", "SOURCE", "REVISION", "LICENSE", "DIRECTION", "ARTIFACT_PREFIX"),
    )
    parser.add_argument("--safety-source", required=True, type=Path)
    parser.add_argument("--safety-manifest", required=True, type=Path)
    parser.add_argument("--safety-review-revision", required=True)
    parser.add_argument("--release-id", default="onevoice-v2-rc1")
    args = parser.parse_args()

    manifest_path = args.artifact_manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError("Release lock requires a schema_version 2 artifact manifest")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Artifact manifest is empty")
    for entry in artifacts:
        if len(str(entry.get("sha256", ""))) != 64:
            raise ValueError(f"Artifact lacks concrete SHA-256: {entry.get('name')}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for entry in artifacts:
        artifact_path = Path(str(entry.get("path", "")))
        if not artifact_path.is_absolute():
            artifact_path = (manifest_path.parent / artifact_path).resolve()
        entry["path"] = Path(os.path.relpath(artifact_path, output.parent)).as_posix()

    models = []
    for raw in args.model:
        model = parse_model(raw)
        names = [
            str(entry.get("name", ""))
            for entry in artifacts
            if str(entry.get("name", "")).startswith(model["artifact_prefix"])
        ]
        if not names:
            raise ValueError(
                f"No hashed artifacts match model prefix {model['artifact_prefix']!r}"
            )
        model["artifacts"] = names
        del model["artifact_prefix"]
        models.append(model)

    safety_source = args.safety_source.resolve()
    safety_manifest = args.safety_manifest.resolve()
    if not safety_source.is_file() or not safety_manifest.is_file():
        raise FileNotFoundError("Safety source CSV and safety manifest must both exist")
    safety_payload = json.loads(safety_manifest.read_text(encoding="utf-8"))
    source_digest = sha256(safety_source)
    if str(safety_payload.get("source_sha256", "")).casefold() != source_digest:
        raise ValueError("Safety source CSV does not match safety audio manifest source_sha256")
    manifest_approval = str(safety_payload.get("approval_id", "")).strip()
    if manifest_approval and manifest_approval != args.safety_review_revision:
        raise ValueError(
            "Safety review revision does not match safety audio manifest approval_id: "
            f"{args.safety_review_revision!r} != {manifest_approval!r}"
        )

    release_lock = {
        **payload,
        "manifest_kind": "onevoice.release_lock",
        "release_id": args.release_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "models": models,
        "safety_provenance": {
            "source_csv": str(safety_source),
            "source_sha256": source_digest,
            "audio_manifest": str(safety_manifest),
            "audio_manifest_sha256": sha256(safety_manifest),
            "review_revision": args.safety_review_revision,
        },
    }
    output.write_text(
        json.dumps(release_lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote release lock with {len(models)} models: {output}")


if __name__ == "__main__":
    main()
