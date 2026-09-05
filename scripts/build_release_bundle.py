"""Build a direction-scoped OneVoice release bundle or inventory.

``inventory`` is the safe default for Google Drive: it creates a hash-locked
receipt without duplicating large model files. ``copy`` materializes the same
layout under the output directory and resumes files whose SHA-256 already
matches. With ``--runtime-config``, copy also writes a local edge config.
Both modes are fail-closed and never download artifacts.
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

import yaml


ASSET_DESTINATIONS = {
    # The copied layout deliberately matches the runtime config paths.  A
    # bundle can therefore be executed from its own root without Drive paths.
    "gipformer": "models/gipformer",
    "sensevoice_fp32": "models/sensevoice_en_construction_v1_onnx_fp32",
    "mt_vi2en": "models/envit5_finetuned_vi2en_v1",
    "mt_en2vi": "models/envit5_finetuned_en2vi_v1",
    "mt_vi2en_ort": "models/mt/vi2en_ort",
    "mt_en2vi_ort": "models/mt/en2vi_ort",
    "safety_audio": "artifacts/safety_audio",
    "reviewed_safety_csv": "data/onevoice_construction_v2",
    "construction_data": "data/onevoice_construction_v2",
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
    if head == "mt_vi2en_ort" and direction != "vi2en":
        raise ValueError(f"VI→EN edge MT artifact leaked into {direction}: {name}")
    if head == "mt_en2vi_ort" and direction != "en2vi":
        raise ValueError(f"EN→VI edge MT artifact leaked into {direction}: {name}")
    relative = Path(tail) if tail else Path(Path(name).name)
    if head == "reviewed_safety_csv":
        relative = Path("safety_fast_path_review.csv")
    return Path(ASSET_DESTINATIONS[head]) / relative


def _filter_backends(backends: object, direction: str) -> list[dict]:
    if not isinstance(backends, list):
        return []
    result = []
    for backend in backends:
        if not isinstance(backend, dict):
            continue
        directions = set(backend.get("directions", ["vi2en", "en2vi"]))
        if direction in directions:
            result.append(dict(backend))
    return result


def write_portable_runtime_config(
    source_config: Path,
    destination: Path,
    direction: str,
    asset_names: set[str],
) -> None:
    """Write an edge config whose relative paths are rooted at a copied bundle.

    This is intentionally stricter than the inventory mode: the context and
    safety sources are required alongside the model bundles.  It produces a
    local runtime contract, not an Android app; Android still needs native
    adapters for the listed Python backends.
    """
    required = {
        "vi2en": {"gipformer", "mt_vi2en_ort", "safety_audio", "reviewed_safety_csv", "construction_data"},
        "en2vi": {"sensevoice_fp32", "mt_en2vi_ort", "safety_audio", "reviewed_safety_csv", "construction_data"},
    }[direction]
    missing = sorted(required - asset_names)
    if missing:
        raise ValueError(
            "Cannot materialize portable runtime config; missing artifact roots: "
            + ", ".join(missing)
        )
    payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid runtime config: {source_config}")
    pipeline = payload.setdefault("pipeline", {})
    pipeline.update(
        {
            "offline": True,
            "profile": "edge",
            "artifact_manifest": "manifest.json",
            "construction_data_dir": "data/onevoice_construction_v2",
            "safety_source_csv": "data/onevoice_construction_v2/safety_fast_path_review.csv",
            "safety_audio_manifest": "artifacts/safety_audio/manifest.json",
        }
    )
    if direction == "vi2en":
        payload.setdefault("asr", {})["gipformer_model_dir"] = "models/gipformer"
        payload.setdefault("translation", {}).setdefault("directions", {}).setdefault("vi2en", {})[
            "edge_model_dir"
        ] = "models/mt/vi2en_ort"
    else:
        payload.setdefault("sensevoice", {})["model_path"] = "models/sensevoice_en_construction_v1_onnx_fp32"
        payload.setdefault("translation", {}).setdefault("directions", {}).setdefault("en2vi", {})[
            "edge_model_dir"
        ] = "models/mt/en2vi_ort"
    profiles = payload.setdefault("profiles", {})
    profiles.setdefault("edge", {})["allow_downloads"] = False
    destination.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def build_bundle(
    artifact_manifest: Path,
    output_dir: Path,
    direction: str,
    mode: str = "inventory",
    release_lock: Path | None = None,
    config: Path | None = None,
    runtime_config: Path | None = None,
) -> dict:
    if runtime_config is not None and mode != "copy":
        raise ValueError("--runtime-config requires --mode copy")
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
        "sample_rates": payload.get("sample_rates", [16000]),
        "required_backends": _filter_backends(payload.get("required_backends"), direction),
        "artifacts": manifest_entries,
    }
    bundle_manifest_path = output_dir / "manifest.json"
    atomic_json(bundle_manifest_path, bundle_manifest)
    receipt["bundle_manifest_sha256"] = sha256(bundle_manifest_path)
    if runtime_config is not None:
        runtime_config_path = output_dir / "runtime_config.yaml"
        write_portable_runtime_config(
            runtime_config, runtime_config_path, direction,
            {asset_name(str(entry["name"]))[0] for entry, *_rest in selected},
        )
        receipt["runtime_config"] = runtime_config_path.name
        receipt["runtime_config_sha256"] = sha256(runtime_config_path)
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
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help="Source config to rewrite for a copied, self-contained edge bundle.",
    )
    args = parser.parse_args()
    if not args.artifact_manifest.is_file():
        raise FileNotFoundError(args.artifact_manifest)
    if args.release_lock and not args.release_lock.is_file():
        raise FileNotFoundError(args.release_lock)
    if args.config and not args.config.is_file():
        raise FileNotFoundError(args.config)
    if args.runtime_config and not args.runtime_config.is_file():
        raise FileNotFoundError(args.runtime_config)
    receipt = build_bundle(
        args.artifact_manifest,
        args.output_dir,
        args.direction,
        args.mode,
        args.release_lock,
        args.config,
        args.runtime_config,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
