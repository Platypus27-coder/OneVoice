"""Create a separately traceable INT8 SenseVoice ONNX runtime bundle.

The input bundle must already have passed the FP32 held-out benchmark.  This
tool never modifies it: it copies the runtime assets to a new directory and
creates ``model_quant.onnx`` there.  The FunASR ONNX runtime selects that file
only when configured with ``quantize=True``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_fp32_runtime(source: Path, output: Path) -> None:
    """Copy FP32 assets but deliberately omit any previous INT8 artifact."""
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        destination = output / relative
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        if item.name == "model_quant.onnx" or item.name.startswith("model_quant.onnx."):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    source_model = source / "model.onnx"
    if not source_model.is_file():
        raise FileNotFoundError(f"Missing FP32 model.onnx: {source_model}")
    if source == output:
        raise ValueError("--output-dir must differ from --input-dir; preserve the verified FP32 bundle.")

    complete = output / "quantization_manifest.json"
    quantized_model = output / "model_quant.onnx"
    source_hash = sha256(source_model)
    if complete.is_file() and quantized_model.is_file():
        recorded = json.loads(complete.read_text(encoding="utf-8"))
        if recorded.get("source_model_sha256") == source_hash:
            print(f"[SenseVoice INT8] complete bundle already exists: {output}")
            return
        raise FileExistsError(f"{output} belongs to another FP32 source; choose a new --output-dir.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is non-empty/incomplete; choose a new --output-dir.")

    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as error:
        raise ImportError("Install 'onnx' and 'onnxruntime' before INT8 quantization.") from error

    copy_fp32_runtime(source, output)
    temporary = output / "model_quant.partial.onnx"
    try:
        quantize_dynamic(
            model_input=str(output / "model.onnx"),
            model_output=str(temporary),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        onnx.checker.check_model(str(temporary))
        temporary.replace(quantized_model)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "format": "onevoice.sensevoice.onnx.int8.v1",
        "source_bundle": str(source),
        "source_model_sha256": source_hash,
        "source_export_manifest_sha256": sha256(source / "export_manifest.json")
        if (source / "export_manifest.json").is_file()
        else None,
        "method": "onnxruntime.quantization.quantize_dynamic",
        "weight_type": "QInt8",
        "per_channel": True,
        "runtime_loader_requirement": "Load with quantize=True; benchmark this bundle before deployment.",
        "artifacts": {
            "model.onnx": {"bytes": (output / "model.onnx").stat().st_size, "sha256": sha256(output / "model.onnx")},
            "model_quant.onnx": {"bytes": quantized_model.stat().st_size, "sha256": sha256(quantized_model)},
        },
    }
    complete.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
