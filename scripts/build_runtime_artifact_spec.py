"""Build an artifact spec by recursively inventorying approved runtime bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# The same hashed local bundle is used by the development smoke profile and
# the offline edge profile.  Keeping both profiles in the generated manifest
# prevents edge startup from being rejected before model loading.
RUNTIME_PROFILES = ["development", "edge"]


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


def parse_asset_profiles(values: list[str]) -> tuple[str, list[str]]:
    name, raw_profiles = values
    profiles = [value.strip() for value in raw_profiles.split(",") if value.strip()]
    if not profiles or any(value not in RUNTIME_PROFILES for value in profiles):
        raise ValueError(
            f"Invalid profiles for asset {name}: {raw_profiles}; "
            f"allowed={','.join(RUNTIME_PROFILES)}"
        )
    return name, profiles


def inventory(
    name: str,
    root: Path,
    directions: list[str],
    license_name: str,
    excluded_names: set[str],
    profiles: list[str],
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
                "profiles": profiles,
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
        "--asset-profiles",
        action="append",
        nargs=2,
        metavar=("NAME", "PROFILES"),
        default=[],
        help=(
            "Restrict an asset to comma-separated runtime profiles "
            "(development,edge). Repeat per asset name."
        ),
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
    asset_profiles = dict(parse_asset_profiles(values) for values in args.asset_profiles)
    for values in args.asset:
        name, root, directions, license_name = parse_asset(values)
        artifacts.extend(
            inventory(
                name,
                root,
                directions,
                license_name,
                excluded_names,
                asset_profiles.get(name, RUNTIME_PROFILES.copy()),
            )
        )
    spec = {
        "schema_version": 2,
        "sample_rates": [16000],
        "required_backends": [
            {"name": "GIPFormer runtime", "python_module": "sherpa_onnx", "profiles": RUNTIME_PROFILES.copy(), "directions": ["vi2en"]},
            {"name": "SenseVoice runtime", "python_module": "funasr_onnx", "profiles": RUNTIME_PROFILES.copy(), "directions": ["en2vi"]},
            {"name": "EnViT5 development runtime", "python_module": "transformers", "profiles": ["development"], "directions": ["vi2en", "en2vi"]},
            {"name": "EnViT5 edge runtime", "python_module": "optimum.onnxruntime", "profiles": ["edge"], "directions": ["vi2en", "en2vi"]},
            {"name": "Offline demo TTS", "python_module": "pyttsx3", "profiles": RUNTIME_PROFILES.copy(), "directions": ["vi2en", "en2vi"]},
        ],
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote artifact spec with {len(artifacts)} files: {args.output}")


if __name__ == "__main__":
    main()
