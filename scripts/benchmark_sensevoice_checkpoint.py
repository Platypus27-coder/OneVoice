"""Benchmark a trained SenseVoice PyTorch checkpoint without changing runtime ONNX."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from context.engine import ConstructionContextEngine
from evaluation.metrics import corpus_error_rate
from evaluation.reporting import create_run_manifest


def read_rows(path: Path, split: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("language") == "en" and json.loads(line).get("split") == split
    ]
    if not rows:
        raise ValueError(f"No English {split} rows in {path}")
    return rows


def clean_text(value: str) -> str:
    return re.sub(r"<\|.*?\|>", "", value).strip()


def load_model(checkpoint: Path, device: str):
    import torch
    from funasr import AutoModel

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in state:
        raise ValueError(f"{checkpoint} is not a trainer checkpoint with state_dict")
    wrapper = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True, device=device, disable_update=True)
    trained = {key.removeprefix("module."): value for key, value in state["state_dict"].items()}
    missing, unexpected = wrapper.model.load_state_dict(trained, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch; missing=" + repr(missing[:10]) + "; unexpected=" + repr(unexpected[:10])
        )
    wrapper.model.eval()
    return wrapper


def summarize(items: list[dict]) -> dict:
    references = [item["reference"] for item in items]
    hypotheses = [item["prediction"] for item in items]
    latency = sorted(item["latency_ms"] for item in items)
    percentile = lambda p: latency[min(round((len(latency) - 1) * p), len(latency) - 1)]
    term_hits, term_total = sum(x["term_hits"] for x in items), sum(x["term_total"] for x in items)
    critical_hits, critical_total = sum(x["critical_hits"] for x in items), sum(x["critical_total"] for x in items)
    safety_hits, safety_total = sum(x["safety_hits"] for x in items), sum(x["safety_total"] for x in items)
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
        "empty_predictions": sum(not item["prediction"] for item in items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", choices=("clean", "noisy"), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--construction-data-dir", type=Path, default=Path("data/onevoice_construction_v2"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    rows = read_rows(args.manifest, args.split)
    if args.max_samples:
        rows = rows[: args.max_samples]
    model = load_model(args.checkpoint, args.device)
    context = ConstructionContextEngine.from_data_dir(args.construction_data_dir)
    root = args.manifest.parent
    started_all = time.perf_counter()
    predictions: list[dict] = []
    print(f"[SenseVoice checkpoint] {len(rows)} {args.split}/{args.audio} samples", flush=True)

    for index, row in enumerate(rows, 1):
        name = str(row["clean_audio"] if args.audio == "clean" else row["noisy_audio"])
        audio_path = root / args.audio / name
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        started = time.perf_counter()
        result = model.generate(input=str(audio_path), cache={}, language="en", use_itn=True, batch_size=1)
        elapsed_ms = (time.perf_counter() - started) * 1000
        prediction = clean_text(str(result[0].get("text", ""))) if result else ""
        reference = str(row["text"])
        ref_context = context.analyze(reference, "en2vi")
        hyp_context = context.analyze(prediction, "en2vi")
        ref_terms = {mention.canonical_id for mention in ref_context.canonical_mentions}
        hyp_terms = {mention.canonical_id for mention in hyp_context.canonical_mentions}
        critical_ref = {mention.canonical_id for mention in ref_context.canonical_mentions if mention.risk_level in {"high", "critical"}}
        safety_ref = ref_context.safety_candidates[0] if ref_context.safety_candidates else None
        safety_hyp = hyp_context.safety_candidates[0] if hyp_context.safety_candidates else None
        predictions.append({
            "audio": name, "reference": reference, "prediction": prediction, "raw": str(result[0].get("text", "")) if result else "",
            "latency_ms": round(elapsed_ms, 3), "domain": row.get("domain", "unknown"), "risk_level": row.get("risk_level", "unknown"),
            "noise_type": row.get("noise_type", "clean"), "snr_db": row.get("realized_snr_db", "clean"),
            "term_hits": len(ref_terms & hyp_terms), "term_total": len(ref_terms),
            "critical_hits": len(critical_ref & hyp_terms), "critical_total": len(critical_ref),
            "safety_hits": int(bool(safety_ref and safety_hyp and safety_ref.safety_id == safety_hyp.safety_id)), "safety_total": int(safety_ref is not None),
        })
        if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
            elapsed = time.perf_counter() - started_all
            rate = index / elapsed if elapsed else 0
            print(f"[SenseVoice checkpoint] {index}/{len(rows)} ({rate:.2f} samples/s)", flush=True)

    aggregate = {**summarize(predictions), "audio": args.audio, "split": args.split, "checkpoint": str(args.checkpoint.resolve()), "route": "pytorch_checkpoint"}
    aggregate["breakdowns"] = {}
    for field in ("domain", "risk_level", "noise_type", "snr_db"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for prediction in predictions:
            groups[str(prediction[field])].append(prediction)
        aggregate["breakdowns"][field] = {key: summarize(group) for key, group in sorted(groups.items())}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with (args.report_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    (args.report_dir / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.report_dir / "breakdown.json").write_text(json.dumps(aggregate["breakdowns"], ensure_ascii=False, indent=2), encoding="utf-8")
    create_run_manifest(args.report_dir / "run_manifest.json", "benchmark_sensevoice_checkpoint", [args.manifest, args.checkpoint], vars(args))
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
