"""Measured ASR/denoiser benchmark for OneVoice V2 manifests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from asr.asr_manager import ASRManager
from audio.denoise import Denoiser
from context.engine import ConstructionContextEngine
from evaluation.metrics import corpus_error_rate
from evaluation.reporting import create_run_manifest


def read_manifest(path: Path, split: str, language: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split")) != split:
                continue
            row_language = row.get("language")
            if row_language and row_language != language:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No {language}/{split} samples found in {path}")
    return rows


def load_partial_predictions(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid resume checkpoint {path}:{line_number}") from error
        if not isinstance(row, dict) or not row.get("audio"):
            raise ValueError(f"Invalid prediction in resume checkpoint {path}:{line_number}")
        rows.append(row)
    return rows


def write_partial_predictions(path: Path, predictions: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--direction", choices=["vi2en", "en2vi"], default="vi2en")
    parser.add_argument("--split", default="test")
    parser.add_argument("--audio", choices=["clean", "noisy"], default="noisy")
    parser.add_argument(
        "--denoiser", choices=["passthrough", "rnnoise", "deepfilter"], default="passthrough"
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print measured progress every N samples (0 disables periodic progress).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume predictions.partial.jsonl in --report-dir")
    parser.add_argument("--save-every", type=int, default=25, help="Checkpoint every N predictions when --resume is set")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--sensevoice-model-dir",
        type=Path,
        help="Local SenseVoice ONNX bundle for EN→VI benchmarking; disables model download.",
    )
    parser.add_argument(
        "--sensevoice-quantize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Select model_quant.onnx when available. Use --no-sensevoice-quantize for an FP32 candidate.",
    )
    parser.add_argument("--report-dir", default="reports/asr_v2")
    args = parser.parse_args()

    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("ASR benchmark requires librosa") from exc

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.sensevoice_model_dir:
        if args.direction != "en2vi":
            parser.error("--sensevoice-model-dir is only valid for --direction en2vi")
        if not args.sensevoice_model_dir.is_dir():
            raise FileNotFoundError(f"SenseVoice ONNX bundle not found: {args.sensevoice_model_dir}")
        config.setdefault("sensevoice", {})["model_path"] = str(args.sensevoice_model_dir)
    if args.sensevoice_quantize is not None:
        if args.direction != "en2vi":
            parser.error("--sensevoice-quantize is only valid for --direction en2vi")
        config.setdefault("sensevoice", {})["quantize"] = args.sensevoice_quantize
    language = "vi" if args.direction == "vi2en" else "en"
    manifest = Path(args.manifest)
    rows = read_manifest(manifest, args.split, language)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if args.progress_every < 0:
        parser.error("--progress-every must be zero or a positive integer")
    if args.save_every < 0:
        parser.error("--save-every must be zero or a positive integer")

    output = Path(args.report_dir)
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "predictions.partial.jsonl"
    predictions: list[dict] = load_partial_predictions(partial_path) if args.resume else []
    resumed_count = len(predictions)
    expected_audio = {
        str(row["clean_audio"] if args.audio == "clean" else row["audio"])
        for row in rows
    }
    completed = {str(prediction["audio"]) for prediction in predictions}
    unknown = completed - expected_audio
    if unknown:
        raise ValueError(f"Resume checkpoint has {len(unknown)} audio names outside this benchmark run")
    if len(completed) != len(predictions):
        raise ValueError(f"Resume checkpoint contains duplicate audio names: {partial_path}")

    asr = ASRManager(config, offline=bool(args.sensevoice_model_dir))
    asr.load(args.direction)
    denoiser_config = dict(config.get("denoise", {}))
    denoiser_config["backend"] = args.denoiser
    denoiser = Denoiser(denoiser_config)
    denoiser.load()
    context = ConstructionContextEngine.from_data_dir(
        config["pipeline"]["construction_data_dir"]
    )
    root = manifest.parent
    benchmark_started = time.perf_counter()
    if predictions:
        print(f"[ASR benchmark] resuming {resumed_count}/{len(rows)} saved predictions from {partial_path}", flush=True)
    print(
        f"[ASR benchmark] {args.direction}/{args.audio}: {len(rows)} {args.split} samples "
        f"(progress every {args.progress_every or 'disabled'})",
        flush=True,
    )

    for row in rows:
        name = row["clean_audio"] if args.audio == "clean" else row["audio"]
        if name in completed:
            continue
        audio_path = root / args.audio / name
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        started = time.perf_counter()
        enhanced = denoiser.process(audio, 16000)
        result = asr.transcribe(enhanced, args.direction)
        elapsed_ms = (time.perf_counter() - started) * 1000
        reference = str(row["text"])
        prediction = str(result.get("text", ""))
        ref_context = context.analyze(reference, args.direction)
        hyp_context = context.analyze(prediction, args.direction)
        ref_terms = {mention.canonical_id for mention in ref_context.canonical_mentions}
        hyp_terms = {mention.canonical_id for mention in hyp_context.canonical_mentions}
        critical_ref = {
            mention.canonical_id
            for mention in ref_context.canonical_mentions
            if mention.risk_level in {"high", "critical"}
        }
        safety_ref = ref_context.safety_candidates[0] if ref_context.safety_candidates else None
        safety_hyp = hyp_context.safety_candidates[0] if hyp_context.safety_candidates else None
        predictions.append(
            {
                "audio": name,
                "reference": reference,
                "prediction": prediction,
                "latency_ms": round(elapsed_ms, 3),
                "domain": row.get("domain", "unknown"),
                "risk_level": row.get("risk_level", "unknown"),
                "noise_type": row.get("noise_type", "clean"),
                "snr_db": row.get("snr_db", row.get("target_snr_db", "clean")),
                "term_hits": len(ref_terms & hyp_terms),
                "term_total": len(ref_terms),
                "critical_hits": len(critical_ref & hyp_terms),
                "critical_total": len(critical_ref),
                "safety_hits": int(
                    bool(safety_ref and safety_hyp and safety_ref.safety_id == safety_hyp.safety_id)
                ),
                "safety_total": int(safety_ref is not None),
            }
        )
        completed.add(name)
        index = len(predictions)
        if args.resume and args.save_every and (index % args.save_every == 0 or index == len(rows)):
            write_partial_predictions(partial_path, predictions)
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            elapsed_s = time.perf_counter() - benchmark_started
            rate = (index - resumed_count) / elapsed_s if elapsed_s else 0.0
            remaining_s = (len(rows) - index) / rate if rate else 0.0
            print(
                f"[ASR benchmark] processed {index}/{len(rows)} "
                f"({rate:.2f} samples/s; ETA {remaining_s / 60:.1f} min)",
                flush=True,
            )

    def summarize(items: list[dict]) -> dict:
        references = [item["reference"] for item in items]
        hypotheses = [item["prediction"] for item in items]
        term_hits = sum(item["term_hits"] for item in items)
        term_total = sum(item["term_total"] for item in items)
        critical_hits = sum(item["critical_hits"] for item in items)
        critical_total = sum(item["critical_total"] for item in items)
        safety_hits = sum(item["safety_hits"] for item in items)
        safety_total = sum(item["safety_total"] for item in items)
        latency = sorted(item["latency_ms"] for item in items)
        percentile = lambda p: latency[min(round((len(latency) - 1) * p), len(latency) - 1)]
        term_recall = term_hits / term_total if term_total else None
        return {
            "samples": len(items),
            "wer": corpus_error_rate(references, hypotheses, "word"),
            "cer": corpus_error_rate(references, hypotheses, "char"),
            "construction_term_recall": term_recall,
            "construction_term_error_rate": 1 - term_recall if term_recall is not None else None,
            "critical_term_recall": critical_hits / critical_total if critical_total else None,
            "safety_phrase_recall": safety_hits / safety_total if safety_total else None,
            "latency_p50_ms": percentile(0.50),
            "latency_p95_ms": percentile(0.95),
            "empty_predictions": sum(not item["prediction"].strip() for item in items),
        }

    aggregate = {
        **summarize(predictions),
        "direction": args.direction,
        "audio": args.audio,
        "denoiser": args.denoiser,
        "asr_model_dir": str(args.sensevoice_model_dir) if args.sensevoice_model_dir else None,
        "asr_quantized": config.get("sensevoice", {}).get("quantize", True) if args.direction == "en2vi" else None,
        "breakdowns": {},
    }
    for field in ("domain", "risk_level", "noise_type", "snr_db"):
        groups = defaultdict(list)
        for item in predictions:
            groups[str(item[field])].append(item)
        aggregate["breakdowns"][field] = {
            key: summarize(items) for key, items in sorted(groups.items())
        }

    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "breakdown.json").write_text(
        json.dumps(aggregate["breakdowns"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    create_run_manifest(
        output / "run_manifest.json",
        command="benchmark_asr_v2",
        inputs=[
            manifest,
            args.config,
            config["asr"].get("gipformer_model_dir", "models/gipformer")
            if args.direction == "vi2en"
            else (args.sensevoice_model_dir or config["sensevoice"].get("model_path", "models/sensevoice")),
        ],
        metadata=vars(args),
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
