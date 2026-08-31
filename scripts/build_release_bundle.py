"""Build a direction-scoped OneVoice release bundle or inventory.

``inventory`` is the safe default for Google Drive: it creates a portable
manifest/receipt without duplicating large model files. ``copy`` materializes
the same layout under the output directory and resumes files whose SHA-256
already matches. Both modes are fail-closed and never download artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ASSET_DESTINATIONS = {
    "gipformer": "asr/gipformer",
    "sensevoice_fp32": "asr/sensevoice_fp32",
    "mt_vi2en": "mt/envit5",
    "mt_en2vi": "mt/envit5",
    "safety_audio": "safety_audio",
    "reviewed_safety_csv": "site_packs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        temporary = Path(handle.name)
    temporary.replace(path)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_artifact(manifest: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def asset_name(name: str) -> tuple[str, str]:
    head, separator, tail = name.partition("/")
    return head, tail if separator else Path(name).name


def destination_for(name: str, direction: str) -> Path:
    head, tail = asset_name(name)
    if head not in ASSET_DESTINATIONS:
        raise ValueError(f"Unknown artifact asset prefix: {name}")
    # Keep the two MT directions distinguishable inside their direction bundle.
    if head == "mt_vi2en" and direction != "vi2en":
        raise ValueError(f"VI→EN MT artifact leaked into {direction}: {name}")
    if head == "mt_en2vi" and direction != "en2vi":
        raise ValueError(f"EN→VI MT artifact leaked into {direction}: {name}")
    relative = Path(tail) if tail else Path(Path(name).name)
    if head == "reviewed_safety_csv":
        relative = Path("reviewed_safety.csv")
    return Path(ASSET_DESTINATIONS[head]) / relative


def build_bundle(
    artifact_manifest: Path,
    output_dir: Path,
    direction: str,
    mode: str = "inventory",
    release_lock: Path | None = None,
    config: Path | None = None,
) -> dict:
    payload = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) < 2:
        raise ValueError("P4 requires a schema v2 artifact manifest")
    entries = payload.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Artifact manifest has no artifacts")
    selected = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Artifact entry is not an object")
        directions = set(entry.get("directions", ["vi2en", "en2vi"]))
        if direction not in directions:
            continue
        name = str(entry.get("name", "")).strip()
        raw_path = str(entry.get("path", "")).strip()
        expected = str(entry.get("sha256", "")).casefold()
        license_name = str(entry.get("license", "")).strip()
        if not name or not raw_path or len(expected) != 64 or not license_name:
            raise ValueError(f"Incomplete artifact entry for {direction}: {entry}")
        source = resolve_artifact(artifact_manifest, raw_path)
        if not source.is_file():
            raise FileNotFoundError(f"Missing {name}: {source}")
        actual = sha256(source)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
        selected.append((entry, source, actual, destination_for(name, direction)))
    if not selected:
        raise ValueError(f"No artifacts apply to direction={direction}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    reused = copied = 0
    used_destinations: set[str] = set()
    for entry, source, digest, relative_destination in sorted(selected, key=lambda item: item[3].as_posix()):
        key = relative_destination.as_posix()
        if key in used_destinations:
            raise ValueError(f"Destination collision in {direction}: {key}")
        used_destinations.add(key)
        destination = output_dir / relative_destination
        if mode == "copy":
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                if sha256(destination) != digest:
                    raise FileExistsError(
                        f"Refusing to overwrite different bundle file: {destination}"
                    )
                reused += 1
            else:
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                shutil.copy2(source, temporary)
                if sha256(temporary) != digest:
                    temporary.unlink(missing_ok=True)
                    raise IOError(f"Copied file failed SHA-256 verification: {source}")
                temporary.replace(destination)
                copied += 1
            stored_path = relative_destination.as_posix()
        else:
            stored_path = str(source.resolve())
        manifest_entries.append(
            {
                "name": str(entry["name"]),
                "path": stored_path,
                "sha256": digest,
                "license": str(entry["license"]),
                "directions": [direction],
                "profiles": entry.get("profiles", ["development"]),
                "source_path": str(source.resolve()),
            }
        )

    receipt = {
        "schema_version": 1,
        "manifest_kind": "onevoice.direction_release_bundle",
        "direction": direction,
        "mode": mode,
        "portable": mode == "copy",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "source_artifact_manifest": str(artifact_manifest.resolve()),
        "source_artifact_manifest_sha256": sha256(artifact_manifest),
        "release_lock": str(release_lock.resolve()) if release_lock else None,
        "release_lock_sha256": sha256(release_lock) if release_lock else None,
        "config": str(config.resolve()) if config else None,
        "config_sha256": sha256(config) if config else None,
        "artifact_count": len(manifest_entries),
        "copied": copied,
        "reused": reused,
        "artifacts": manifest_entries,
        "network": "not used",
        "scope": "release-candidate demo; synthetic safety audio is internal-demo evidence only",
    }
    bundle_manifest = {
        "schema_version": 2,
        "manifest_kind": "onevoice.direction_release_bundle",
        "direction": direction,
        "portable": mode == "copy",
        "artifacts": manifest_entries,
    }
    bundle_manifest_path = output_dir / "manifest.json"
    atomic_json(bundle_manifest_path, bundle_manifest)
    receipt["bundle_manifest_sha256"] = sha256(bundle_manifest_path)
    atomic_json(output_dir / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--direction", choices=("vi2en", "en2vi"), required=True)
    parser.add_argument("--mode", choices=("inventory", "copy"), default="inventory")
    parser.add_argument("--release-lock", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if not args.artifact_manifest.is_file():
        raise FileNotFoundError(args.artifact_manifest)
    if args.release_lock and not args.release_lock.is_file():
        raise FileNotFoundError(args.release_lock)
    if args.config and not args.config.is_file():
        raise FileNotFoundError(args.config)
    receipt = build_bundle(
        args.artifact_manifest,
        args.output_dir,
        args.direction,
        args.mode,
        args.release_lock,
        args.config,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
