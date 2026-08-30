"""Build an artifact spec by recursively inventorying approved runtime bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_asset(values: list[str]) -> tuple[str, Path, list[str], str]:
    name, raw_path, raw_directions, license_name = values
    path = Path(raw_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Asset root not found: {path}")
    directions = [value.strip() for value in raw_directions.split(",") if value.strip()]
    if not directions or any(value not in {"vi2en", "en2vi"} for value in directions):
        raise ValueError(f"Invalid asset directions: {raw_directions}")
    if not license_name.strip():
        raise ValueError(f"Asset {name} is missing license provenance")
    return name, path, directions, license_name.strip()


def inventory(
    name: str,
    root: Path,
    directions: list[str],
    license_name: str,
    excluded_names: set[str],
) -> list[dict]:
    paths = [root] if root.is_file() else sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded_names
    )
    if not paths:
        raise FileNotFoundError(f"Asset root contains no files: {root}")
    result = []
    for path in paths:
        suffix = path.name if root.is_file() else path.relative_to(root).as_posix()
        result.append(
            {
                "name": f"{name}/{suffix}",
                "source_path": str(path),
                "license": license_name,
                "directions": directions,
                "profiles": ["development"],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--asset",
        action="append",
        nargs=4,
        metavar=("NAME", "PATH", "DIRECTIONS", "LICENSE"),
        required=True,
        help="Repeat for each file or bundle. DIRECTIONS is vi2en,en2vi or one direction.",
    )
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        metavar="FILENAME",
        help="Exclude an unapproved candidate file by its basename (repeatable).",
    )
    args = parser.parse_args()

    artifacts = []
    excluded_names = {name.strip() for name in args.exclude_name if name.strip()}
    for values in args.asset:
        artifacts.extend(inventory(*parse_asset(values), excluded_names))
    spec = {
        "schema_version": 2,
        "sample_rates": [16000],
        "required_backends": [
            {"name": "GIPFormer runtime", "python_module": "sherpa_onnx", "profiles": ["development"], "directions": ["vi2en"]},
            {"name": "SenseVoice runtime", "python_module": "funasr_onnx", "profiles": ["development"], "directions": ["en2vi"]},
            {"name": "EnViT5 runtime", "python_module": "transformers", "profiles": ["development"], "directions": ["vi2en", "en2vi"]},
            {"name": "Offline demo TTS", "python_module": "pyttsx3", "profiles": ["development"], "directions": ["vi2en", "en2vi"]},
        ],
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote artifact spec with {len(artifacts)} files: {args.output}")


if __name__ == "__main__":
    main()
