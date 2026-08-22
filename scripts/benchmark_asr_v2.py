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
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--report-dir", default="reports/asr_v2")
    args = parser.parse_args()

    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("ASR benchmark requires librosa") from exc

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    language = "vi" if args.direction == "vi2en" else "en"
    manifest = Path(args.manifest)
    rows = read_manifest(manifest, args.split, language)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if args.progress_every < 0:
        parser.error("--progress-every must be zero or a positive integer")

    asr = ASRManager(config, offline=False)
    asr.load(args.direction)
    denoiser_config = dict(config.get("denoise", {}))
    denoiser_config["backend"] = args.denoiser
    denoiser = Denoiser(denoiser_config)
    denoiser.load()
    context = ConstructionContextEngine.from_data_dir(
        config["pipeline"]["construction_data_dir"]
    )
    predictions: list[dict] = []
    root = manifest.parent
    benchmark_started = time.perf_counter()
    print(
        f"[ASR benchmark] {args.direction}/{args.audio}: {len(rows)} {args.split} samples "
        f"(progress every {args.progress_every or 'disabled'})",
        flush=True,
    )

    for index, row in enumerate(rows, start=1):
        name = row["clean_audio"] if args.audio == "clean" else row["audio"]
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
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            elapsed_s = time.perf_counter() - benchmark_started
            rate = index / elapsed_s if elapsed_s else 0.0
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
        "breakdowns": {},
    }
    for field in ("domain", "risk_level", "noise_type", "snr_db"):
        groups = defaultdict(list)
        for item in predictions:
            groups[str(item[field])].append(item)
        aggregate["breakdowns"][field] = {
            key: summarize(items) for key, items in sorted(groups.items())
        }

    output = Path(args.report_dir)
    output.mkdir(parents=True, exist_ok=True)
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
            else config["sensevoice"].get("model_path", "models/sensevoice"),
        ],
        metadata=vars(args),
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
