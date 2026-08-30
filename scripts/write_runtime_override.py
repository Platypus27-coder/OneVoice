"""Write a local/offline OneVoice runtime config without changing the base config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_dir(path: Path, label: str, marker: str) -> None:
    if not path.is_dir() or not (path / marker).is_file():
        raise FileNotFoundError(f"Missing {label} bundle ({marker}): {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="config/config.yaml", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gipformer-dir", required=True, type=Path)
    parser.add_argument("--sensevoice-dir", required=True, type=Path)
    parser.add_argument("--mt-vi2en-dir", required=True, type=Path)
    parser.add_argument("--mt-en2vi-dir", required=True, type=Path)
    parser.add_argument("--safety-csv", required=True, type=Path)
    parser.add_argument("--safety-manifest", required=True, type=Path)
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        help="Optional concrete SHA-256 manifest required before offline startup.",
    )
    parser.add_argument(
        "--release-lock",
        type=Path,
        help="Concrete release_lock_v2.json; preferred over --artifact-manifest.",
    )
    parser.add_argument("--profile", choices=["development", "premium"], default="development")
    args = parser.parse_args()

    require_file(args.base_config, "base config")
    require_dir(args.gipformer_dir, "GIPFormer", "tokens.txt")
    require_dir(args.sensevoice_dir, "SenseVoice FP32", "model.onnx")
    require_dir(args.mt_vi2en_dir, "VI→EN MT", "config.json")
    require_dir(args.mt_en2vi_dir, "EN→VI MT", "config.json")
    require_file(args.safety_csv, "reviewed safety CSV")
    require_file(args.safety_manifest, "safety audio manifest")
    if args.artifact_manifest is not None:
        require_file(args.artifact_manifest, "runtime artifact manifest")
    if args.release_lock is not None:
        require_file(args.release_lock, "runtime release lock")

    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config["asr"]["gipformer_model_dir"] = str(args.gipformer_dir.resolve())
    config["sensevoice"]["model_path"] = str(args.sensevoice_dir.resolve())
    config["sensevoice"]["quantize"] = False  # INT8 candidate failed its quality gate.
    config["sensevoice"]["allow_remote_fallback"] = False
    config["sensevoice"].pop("remote_model", None)
    # Normal offline E2E uses a real local system voice. This is explicitly a
    # development/internal-demo backend, never a network or silence fallback.
    config["tts"]["offline_engine"] = "pyttsx3"
    config["translation"]["directions"]["vi2en"]["local_model_dir"] = str(args.mt_vi2en_dir.resolve())
    config["translation"]["directions"]["en2vi"]["local_model_dir"] = str(args.mt_en2vi_dir.resolve())
    config["pipeline"]["profile"] = args.profile
    config["pipeline"]["offline"] = True
    config["pipeline"]["safety_source_csv"] = str(args.safety_csv.resolve())
    config["pipeline"]["safety_audio_manifest"] = str(args.safety_manifest.resolve())
    if args.artifact_manifest is not None:
        config["pipeline"]["artifact_manifest"] = str(args.artifact_manifest.resolve())
    if args.release_lock is not None:
        locked = str(args.release_lock.resolve())
        config["pipeline"]["artifact_manifest"] = locked
        config["pipeline"]["release_lock"] = locked

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"Wrote offline runtime override: {args.output}")


if __name__ == "__main__":
    main()
