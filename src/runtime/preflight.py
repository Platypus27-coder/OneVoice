"""Strict local-artifact preflight for production/offline profiles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


class ArtifactPreflightError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifacts(
    manifest_path: str | Path,
    direction: str,
    profile: str,
    sample_rate: int | None = None,
) -> dict[str, Any]:
    """Validate required local artifacts without attempting any download."""
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise ArtifactPreflightError(f"Artifact manifest not found: {manifest_file}")
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactPreflightError(f"Invalid artifact manifest: {exc}") from exc

    schema_version = int(payload.get("schema_version", 1))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ArtifactPreflightError("Artifact manifest must contain an 'artifacts' list")

    root = manifest_file.parent
    checked: list[str] = []
    errors: list[str] = []
    if schema_version >= 2:
        if payload.get("manifest_kind") == "onevoice.release_lock":
            models = payload.get("models")
            if not isinstance(models, list) or not models:
                errors.append("release lock requires a non-empty 'models' list")
            else:
                applicable = [model for model in models if model.get("direction") == direction]
                if not applicable:
                    errors.append(f"release lock has no model for direction={direction}")
                for model in applicable:
                    revision = str(model.get("revision", "")).strip().casefold()
                    if not revision or revision in {"main", "master", "unknown"}:
                        errors.append(
                            f"{model.get('key', 'model')}: immutable revision is required"
                        )
                    if not str(model.get("license", "")).strip():
                        errors.append(f"{model.get('key', 'model')}: license is required")
            safety = payload.get("safety_provenance")
            if not isinstance(safety, dict):
                errors.append("release lock requires safety_provenance")
            else:
                for key in (
                    "source_csv",
                    "source_sha256",
                    "audio_manifest",
                    "audio_manifest_sha256",
                    "review_revision",
                ):
                    if not str(safety.get(key, "")).strip():
                        errors.append(f"safety_provenance.{key} is required")
                for path_key, hash_key in (
                    ("source_csv", "source_sha256"),
                    ("audio_manifest", "audio_manifest_sha256"),
                ):
                    raw_safety_path = str(safety.get(path_key, "")).strip()
                    expected_safety_hash = str(safety.get(hash_key, "")).strip().casefold()
                    if not raw_safety_path or len(expected_safety_hash) != 64:
                        continue
                    safety_path = Path(raw_safety_path)
                    if not safety_path.is_absolute():
                        safety_path = root / safety_path
                    if not safety_path.is_file():
                        errors.append(f"safety_provenance.{path_key}: file not found")
                    elif _sha256(safety_path) != expected_safety_hash:
                        errors.append(f"safety_provenance.{path_key}: SHA-256 mismatch")
        supported_rates = payload.get("sample_rates")
        if not isinstance(supported_rates, list) or not supported_rates:
            errors.append("schema v2 requires a non-empty 'sample_rates' list")
        elif sample_rate is not None and sample_rate not in supported_rates:
            errors.append(
                f"runtime sample rate {sample_rate} is not declared by the manifest"
            )

        for backend in payload.get("required_backends", []):
            if not isinstance(backend, dict):
                errors.append("required backend entry is not an object")
                continue
            directions = set(backend.get("directions", ["vi2en", "en2vi"]))
            profiles = set(backend.get("profiles", ["development", "edge", "premium"]))
            if direction not in directions or profile not in profiles:
                continue
            name = str(backend.get("name", "unnamed backend"))
            module = str(backend.get("python_module", "")).strip()
            if not module:
                errors.append(f"{name}: missing python_module")
            elif importlib.util.find_spec(module) is None:
                errors.append(f"{name}: Python backend '{module}' is not installed")

    for entry in artifacts:
        if not isinstance(entry, dict):
            errors.append("artifact entry is not an object")
            continue
        directions = set(entry.get("directions", ["vi2en", "en2vi"]))
        profiles = set(entry.get("profiles", ["development", "edge", "premium"]))
        if direction not in directions or profile not in profiles:
            continue
        name = str(entry.get("name", entry.get("path", "unnamed")))
        raw_path = str(entry.get("path", "")).strip()
        if not raw_path:
            errors.append(f"{name}: missing path")
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append(f"{name}: file not found ({path})")
            continue
        expected = str(entry.get("sha256", "")).strip().casefold()
        if schema_version >= 2 and not str(entry.get("license", "")).strip():
            errors.append(f"{name}: license metadata is required by schema v2")
            continue
        if schema_version >= 2 and len(expected) != 64:
            errors.append(f"{name}: a concrete SHA-256 is required by schema v2")
            continue
        if expected and _sha256(path) != expected:
            errors.append(f"{name}: SHA-256 mismatch")
            continue
        checked.append(name)

    if schema_version >= 2 and not checked:
        errors.append(
            f"no artifacts apply to direction={direction}, profile={profile}"
        )
    if errors:
        raise ArtifactPreflightError("Offline preflight failed:\n- " + "\n- ".join(errors))
    return {"manifest": str(manifest_file.resolve()), "checked": checked}
