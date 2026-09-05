"""Audit whether a copied direction bundle is ready to become an Android input.

The audit deliberately distinguishes three states:
* artifact portability (all hashed files are local to the bundle),
* runtime-config completeness (the copied Python edge runtime can locate them),
* Android readiness (native Android bindings still need to be implemented).

It never downloads, converts, or promotes a model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from scripts.verify_release_bundle import verify_static
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from verify_release_bundle import verify_static


ANDROID_ADAPTERS = {
    "vi2en": {
        "asr": "sherpa-onnx Android/JNI binding for the verified GIPFormer ONNX bundle",
        "mt": "ONNX Runtime Mobile Seq2Seq generation adapter for EnViT5",
        "tts": "Android TextToSpeech or a separately validated offline Vietnamese/English voice backend",
    },
    "en2vi": {
        "asr": "SenseVoice ONNX Android adapter with tokenizer/prompt parity validation",
        "mt": "ONNX Runtime Mobile Seq2Seq generation adapter for EnViT5",
        "tts": "Android TextToSpeech or a separately validated offline Vietnamese/English voice backend",
    },
}


def audit(bundle_dir: Path, direction: str) -> dict:
    static = verify_static(bundle_dir, direction)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((bundle_dir / "receipt.json").read_text(encoding="utf-8"))
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    absolute_paths = []
    for entry in manifest["artifacts"]:
        name = str(entry["name"])
        group = name.partition("/")[0]
        raw_path = Path(str(entry["path"]))
        path = raw_path if raw_path.is_absolute() else bundle_dir / raw_path
        groups[group]["files"] += 1
        groups[group]["bytes"] += path.stat().st_size
        if raw_path.is_absolute():
            absolute_paths.append(str(raw_path))
    portable = bool(static["portable"]) and not absolute_paths
    runtime_config = bundle_dir / str(receipt.get("runtime_config", "runtime_config.yaml"))
    blockers = []
    if not portable:
        blockers.append("artifact bundle is inventory-only or contains absolute paths")
    if not runtime_config.is_file():
        blockers.append("runtime_config.yaml is absent; copied artifacts have no local runtime contract")
    return {
        "schema_version": 1,
        "direction": direction,
        "bundle_dir": str(bundle_dir.resolve()),
        "artifact_portable": portable,
        "runtime_config_present": runtime_config.is_file(),
        "verified_artifacts": len(static["checked"]),
        "total_bytes": sum(group["bytes"] for group in groups.values()),
        "groups": dict(sorted(groups.items())),
        "android_adapters_required": ANDROID_ADAPTERS[direction],
        "android_app_ready": False,
        "blockers": blockers,
        "next_gate": (
            "Implement and validate native Android adapters; this audit does not "
            "claim that Python backends run on Android."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--direction", choices=("vi2en", "en2vi"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.bundle_dir, args.direction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["blockers"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
