"""Build a concrete offline manifest from local files and a checked-in spec."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OneVoice offline artifact manifest")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", default="artifacts/manifest.json", type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if int(spec.get("schema_version", 0)) != 2:
        raise ValueError("Artifact spec must use schema_version 2")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for entry in spec.get("artifacts", []):
        source_value = str(entry.get("source_path", "")).strip()
        if not source_value:
            raise ValueError("Every artifact requires source_path")
        source = Path(source_value)
        if not source.is_absolute():
            source = (args.spec.parent / source).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Artifact not found: {source}")
        license_name = str(entry.get("license", "")).strip()
        if not license_name:
            raise ValueError(f"Missing license for {source}")
        manifest_entry = {key: value for key, value in entry.items() if key != "source_path"}
        manifest_entry["path"] = Path(os.path.relpath(source, output.parent)).as_posix()
        manifest_entry["sha256"] = sha256(source)
        artifacts.append(manifest_entry)

    manifest = {
        "schema_version": 2,
        "sample_rates": spec.get("sample_rates", [16000]),
        "required_backends": spec.get("required_backends", []),
        "artifacts": artifacts,
    }
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output} with {len(artifacts)} hashed artifacts")


if __name__ == "__main__":
    main()
