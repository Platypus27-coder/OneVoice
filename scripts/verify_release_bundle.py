"""Verify a direction-scoped bundle and run a socket-blocked offline smoke.

The network guard is installed before importing the pipeline.  A successful
run therefore proves that the selected local bundle can load and route one
verified safety case without HF/ModelScope/gTTS/network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tempfile
from pathlib import Path

import yaml


class NetworkBlocked(RuntimeError):
    pass


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


def verify_static(bundle_dir: Path, direction: str) -> dict:
    manifest_path = bundle_dir / "manifest.json"
    receipt_path = bundle_dir / "receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"Bundle requires manifest.json and receipt.json: {bundle_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_kind") != "onevoice.direction_release_bundle":
        raise ValueError("Unexpected release bundle manifest_kind")
    if manifest.get("direction") != direction or receipt.get("direction") != direction:
        raise ValueError("Bundle direction does not match requested direction")
    if receipt.get("bundle_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Bundle receipt does not match manifest SHA-256")
    portable = bool(manifest.get("portable"))
    entries = manifest.get("artifacts", [])
    if not entries:
        raise ValueError("Direction bundle has no artifacts")
    checked = []
    for entry in entries:
        raw_path = str(entry.get("path", "")).strip()
        expected = str(entry.get("sha256", "")).casefold()
        if not raw_path or len(expected) != 64:
            raise ValueError(f"Invalid bundle artifact entry: {entry}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = bundle_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"Bundle artifact missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Bundle artifact hash mismatch: {path}")
        checked.append(str(entry.get("name", path.name)))
    return {"manifest": str(manifest_path.resolve()), "receipt": str(receipt_path.resolve()), "portable": portable, "checked": checked}


def block_network() -> None:
    def denied(*_args, **_kwargs):
        raise NetworkBlocked("network access is disabled by the release smoke harness")

    socket.socket.connect = denied  # type: ignore[assignment]
    socket.create_connection = denied  # type: ignore[assignment]
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "MODELSCOPE_OFFLINE": "1",
            "ONEVOICE_OFFLINE": "1",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--direction", choices=("vi2en", "en2vi"), required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("development", "edge"), default="edge")
    parser.add_argument("--skip-smoke", action="store_true", help="Only verify hashes/config; do not load the pipeline")
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "direction": args.direction,
        "profile": args.profile,
        "offline": True,
        "network_blocked": True,
        "bundle_dir": str(args.bundle_dir.resolve()),
        "input_file": str(args.input_file.resolve()),
        "passed": False,
    }
    try:
        report["bundle"] = verify_static(args.bundle_dir, args.direction)
        if not args.config.is_file() or not args.input_file.is_file():
            raise FileNotFoundError("Config and input file are required")
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not bool(config.get("pipeline", {}).get("offline")):
            raise ValueError("Offline smoke requires pipeline.offline=true in config")
        block_network()
        if not args.skip_smoke:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from pipeline import OneVoicePipeline

            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            pipeline = OneVoicePipeline(
                config_path=str(args.config),
                direction=args.direction,
                profile=args.profile,
                offline=True,
                report_dir=str(args.report_dir),
            )
            result = pipeline.process_file(str(args.input_file), str(args.output_file))
            report["output_file"] = str(Path(result).resolve())
            report["output_sha256"] = sha256(Path(result))
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(args.report_dir / f"bundle_verify_{args.direction}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
