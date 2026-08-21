"""Compile, profile and numerically check an already-validated ONNX model.

This script intentionally does not export a Hugging Face seq2seq model. Encoder,
decoder and generation/cache graphs must first be exported and verified by the
owning model adapter. Qualcomm AI Hub is then used only for deployment work.

Example (hosted-device result, not field-device validation):
  python notebooks/export_qai.py --model models/frozen.onnx \
      --device "<AI Hub Snapdragon 8 Gen 3 device name>" \
      --inputs calibration_sample.npz --report-dir reports/qai/frozen-fp16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _job_record(job) -> dict:
    status = job.get_status() if hasattr(job, "get_status") else "submitted"
    return {
        "job_id": str(getattr(job, "job_id", getattr(job, "id", "unknown"))),
        "status": str(status),
        "url": str(getattr(job, "url", "")),
    }


def _wait(job) -> None:
    if hasattr(job, "wait"):
        job.wait()


def _load_inputs(path: Path) -> dict[str, list[np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        if not payload.files:
            raise ValueError("Input NPZ contains no arrays")
        return {name: [np.asarray(payload[name])] for name in payload.files}


def _local_onnx(model_path: Path, inputs: dict[str, list[np.ndarray]]) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    feed = {name: values[0] for name, values in inputs.items()}
    return session.run(None, feed)


def _numerical_report(local: list[np.ndarray], hosted: dict) -> list[dict]:
    hosted_arrays = [np.asarray(values[0]) for values in hosted.values()]
    if len(local) != len(hosted_arrays):
        raise RuntimeError(
            f"Output count differs: local={len(local)}, hosted={len(hosted_arrays)}"
        )
    report = []
    for index, (expected, actual) in enumerate(zip(local, hosted_arrays)):
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        if expected.shape != actual.shape:
            raise RuntimeError(
                f"Output {index} shape differs: {expected.shape} != {actual.shape}"
            )
        absolute = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-8)
        report.append(
            {
                "output_index": index,
                "shape": list(expected.shape),
                "max_abs_error": float(absolute.max(initial=0.0)),
                "max_relative_error": float((absolute / denominator).max(initial=0.0)),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualcomm AI Hub compile/profile/numerical validation"
    )
    parser.add_argument("--model", required=True, type=Path, help="Frozen ONNX graph")
    parser.add_argument("--device", required=True, help="Exact AI Hub hosted device name")
    parser.add_argument("--inputs", required=True, type=Path, help="Named input arrays (.npz)")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--compile-options", default="--target_runtime qnn")
    parser.add_argument("--profile-options", default="")
    parser.add_argument(
        "--quality-gate-report",
        type=Path,
        help="Required passed gate JSON when compile options request quantization",
    )
    args = parser.parse_args()

    if not os.environ.get("QAI_HUB_API_TOKEN"):
        raise RuntimeError("Set QAI_HUB_API_TOKEN; placeholder credentials are forbidden")
    if not args.model.is_file() or args.model.suffix.casefold() != ".onnx":
        raise FileNotFoundError(f"Validated ONNX model not found: {args.model}")
    if not args.inputs.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {args.inputs}")
    quantized = "int8" in args.compile_options.casefold() or "quant" in args.compile_options.casefold()
    if quantized:
        if not args.quality_gate_report or not args.quality_gate_report.is_file():
            raise RuntimeError("Quantized compile requires --quality-gate-report")
        quality_gate = json.loads(args.quality_gate_report.read_text(encoding="utf-8"))
        if quality_gate.get("passed") is not True:
            raise RuntimeError("Quantized compile refused because quality gate did not pass")
    else:
        quality_gate = None

    import qai_hub as hub

    args.report_dir.mkdir(parents=True, exist_ok=True)
    device = hub.Device(args.device)
    inputs = _load_inputs(args.inputs)
    local_outputs = _local_onnx(args.model, inputs)
    started = time.time()

    compile_job = hub.submit_compile_job(
        model=args.model,
        device=device,
        name=f"onevoice-{args.model.stem}",
        options=args.compile_options,
    )
    _wait(compile_job)
    target_model = compile_job.get_target_model()

    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
        name=f"onevoice-{args.model.stem}-profile",
        options=args.profile_options,
    )
    inference_job = hub.submit_inference_job(
        model=target_model,
        device=device,
        name=f"onevoice-{args.model.stem}-correctness",
        inputs=inputs,
    )
    _wait(profile_job)
    _wait(inference_job)
    hosted_outputs = inference_job.download_output_data()

    profile_path = args.report_dir / "profile.json"
    profile_saved = False
    if hasattr(profile_job, "download_profile"):
        try:
            profile_job.download_profile(str(profile_path))
            profile_saved = profile_path.is_file()
        except (TypeError, OSError):
            profile_saved = False

    report = {
        "result_scope": "qualcomm_ai_hub_hosted_device",
        "field_device_validation": False,
        "model": str(args.model.resolve()),
        "device": args.device,
        "compile_options": args.compile_options,
        "profile_options": args.profile_options,
        "input_quality_gate": quality_gate,
        "elapsed_s": time.time() - started,
        "compile_job": _job_record(compile_job),
        "profile_job": _job_record(profile_job),
        "inference_job": _job_record(inference_job),
        "profile_downloaded": profile_saved,
        "numerical": _numerical_report(local_outputs, hosted_outputs),
    }
    (args.report_dir / "qai_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
