"""Export a verified SenseVoice trainer checkpoint to a portable FP32 ONNX bundle.

The export intentionally creates a new runtime bundle.  It never replaces the
base ModelScope cache or the original PyTorch trainer checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


RUNTIME_REQUIRED_NAMES = {
    "model.onnx",
    "config.yaml",
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_runtime_bundle(stage: Path, output: Path) -> list[dict]:
    """Copy runtime assets only; do not duplicate the base PyTorch weights."""
    output.mkdir(parents=True, exist_ok=True)
    copied: list[dict] = []
    for source in sorted(path for path in stage.rglob("*") if path.is_file()):
        relative = source.relative_to(stage)
        # The ONNX runtime needs model.onnx and frontend/tokenizer assets, but
        # not the 1 GB base PyTorch model that was used solely for export.
        if source.suffix in {".pt", ".bin", ".safetensors", ".ckpt"}:
            continue
        # A previous baseline run can leave model_quant.onnx in the ModelScope
        # cache. This invocation exports FP32 only, so that stale INT8 file
        # would silently point the runtime at base weights rather than the
        # fine-tuned checkpoint.
        if source.name == "model_quant.onnx" or source.name.startswith("model_quant.onnx."):
            continue
        if source.suffix == ".onnx" and source.name != "model.onnx":
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    names = {item["path"] for item in copied}
    missing = sorted(name for name in RUNTIME_REQUIRED_NAMES if name not in names)
    if missing:
        raise FileNotFoundError(f"Export is missing required runtime assets: {missing}")
    return copied


def load_finetuned_model(base_dir: Path, checkpoint: Path, device: str):
    import torch
    from funasr import AutoModel

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint_data:
        raise ValueError(f"{checkpoint} is not a FunASR trainer checkpoint with state_dict")
    wrapper = AutoModel(
        model=str(base_dir),
        trust_remote_code=False,
        device=device,
        disable_update=True,
        disable_pbar=True,
    )
    trained = {
        key.removeprefix("module."): value
        for key, value in checkpoint_data["state_dict"].items()
    }
    missing, unexpected = wrapper.model.load_state_dict(trained, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch; missing="
            + repr(missing[:10])
            + "; unexpected="
            + repr(unexpected[:10])
        )
    wrapper.model.eval()
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="iic/SenseVoiceSmall")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu", help="Use cpu for portable deterministic export")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    complete_manifest = args.output_dir / "export_manifest.json"
    if complete_manifest.is_file() and (args.output_dir / "model.onnx").is_file():
        recorded = json.loads(complete_manifest.read_text(encoding="utf-8"))
        if recorded.get("checkpoint_sha256") == sha256(args.checkpoint):
            print(f"[SenseVoice export] complete bundle already exists: {args.output_dir}")
            return
        raise FileExistsError(
            f"{args.output_dir} already contains an export for another checkpoint; choose a new --output-dir."
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"{args.output_dir} is incomplete/non-empty; choose a new --output-dir rather than overwriting it."
        )

    try:
        from modelscope import snapshot_download
    except ImportError as error:
        raise ImportError("Install modelscope before exporting SenseVoice.") from error

    base_dir = Path(snapshot_download(args.base_model, cache_dir=str(args.cache_dir) if args.cache_dir else None))
    args.stage_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_dir, args.stage_dir, dirs_exist_ok=True)
    print(f"[SenseVoice export] staged base model in {args.stage_dir}", flush=True)

    wrapper = load_finetuned_model(args.stage_dir, args.checkpoint, args.device)
    exported_dir = Path(
        wrapper.export(type="onnx", quantize=False, device=args.device, opset_version=args.opset)
    )
    model_path = exported_dir / "model.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(f"FunASR export finished without model.onnx in {exported_dir}")

    try:
        import onnx

        onnx.checker.check_model(str(model_path))
    except ImportError as error:
        raise ImportError("Install onnx to validate the exported bundle.") from error
    except Exception as error:
        raise RuntimeError(f"ONNX validation failed for {model_path}: {error}") from error

    copied = copy_runtime_bundle(exported_dir, args.output_dir)
    manifest = {
        "format": "onevoice.sensevoice.onnx.fp32.v1",
        "base_model": args.base_model,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "precision": "fp32",
        "quantization": "not_performed",
        "runtime_loader_requirement": "Load model.onnx with quantize=False until a separately validated INT8 export exists.",
        "opset": args.opset,
        "export_device": args.device,
        "artifacts": copied,
    }
    complete_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
