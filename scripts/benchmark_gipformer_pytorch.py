"""Benchmark the official GIPFormer PyTorch inference entry point reproducibly.

The upstream project exposes an Icefall-based PyTorch inference script, but not
an approved construction-domain training recipe.  This bridge deliberately
does *not* train anything: it proves that a pinned upstream checkpoint can be
loaded and compares its transcripts against the same held-out/dev slice used
by the existing ONNX runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context.engine import ConstructionContextEngine
from evaluation.metrics import corpus_error_rate
from evaluation.reporting import create_run_manifest


FILE_LINE = re.compile(r"^\s*File:\s*(.+?)\s*$")
TEXT_LINE = re.compile(r"^\s*Text:\s*(.*?)\s*$")
TIME_LINE = re.compile(r"^\s*Time:\s*([0-9.]+)s\b")


def read_rows(manifest: Path, split: str, audio: str, max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    selected_names: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {manifest}:{line_number}") from exc
        if row.get("split") != split:
            continue
        language = row.get("language")
        if language and language != "vi":
            continue
        name = str(row.get("clean_audio") if audio == "clean" else row.get("audio", "")).strip()
        if not name:
            raise ValueError(f"Missing {audio} audio filename in {manifest}:{line_number}")
        if not str(row.get("text", "")).strip():
            raise ValueError(f"Missing transcript in {manifest}:{line_number}")
        if name in selected_names:
            raise ValueError(
                f"Duplicate {audio} WAV selected: {name}. "
                "Create a canonical deduplicated manifest before benchmarking."
            )
        selected_names.add(name)
        rows.append({**row, "_audio_name": name})
    if max_samples:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"No Vietnamese {split}/{audio} rows selected")
    return rows


def load_partial(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid partial prediction {path}:{line_number}") from exc
        if not record.get("audio"):
            raise ValueError(f"Partial prediction misses audio at {path}:{line_number}")
        records.append(record)
    return records


def save_partial(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_upstream_output(stdout: str) -> dict[str, tuple[str, float | None]]:
    """Return ``resolved audio path -> (transcript, time_seconds)``.

    Parsing is intentionally strict: a changed upstream display format should
    fail the gate rather than silently associate an utterance with a wrong
    transcript.
    """
    result: dict[str, tuple[str, float | None]] = {}
    current_path: str | None = None
    current_text: str | None = None
    current_time: float | None = None

    def flush() -> None:
        nonlocal current_path, current_text, current_time
        if current_path is None:
            return
        if current_text is None:
            raise ValueError(f"Upstream output has File without Text: {current_path}")
        key = str(Path(current_path).resolve())
        if key in result:
            raise ValueError(f"Upstream output contains duplicate file: {current_path}")
        result[key] = (current_text, current_time)
        current_path = current_text = None
        current_time = None

    for line in stdout.splitlines():
        file_match = FILE_LINE.match(line)
        if file_match:
            flush()
            current_path = file_match.group(1)
            continue
        text_match = TEXT_LINE.match(line)
        if text_match and current_path is not None:
            current_text = text_match.group(1)
            continue
        time_match = TIME_LINE.match(line)
        if time_match and current_path is not None:
            current_time = float(time_match.group(1))
    flush()
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--infer-script", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable, help="Python executable from the upstream uv environment")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--icefall-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--audio", choices=("clean", "noisy"), default="noisy")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--baseline-aggregate", type=Path)
    parser.add_argument("--max-regression-pp", type=float, default=1.0)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_regression_pp < 0:
        parser.error("--max-regression-pp must be non-negative")
    for required, label in ((args.infer_script, "infer script"), (args.model_dir, "model directory")):
        if not required.exists():
            raise FileNotFoundError(f"Missing {label}: {required}")

    rows = read_rows(args.manifest, args.split, args.audio, args.max_samples)
    root = args.manifest.parent
    output = args.report_dir
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "predictions.partial.jsonl"
    predictions = load_partial(partial_path) if args.resume else []
    expected = {row["_audio_name"] for row in rows}
    completed = {record["audio"] for record in predictions}
    if not completed <= expected or len(completed) != len(predictions):
        raise ValueError("Partial predictions do not match this manifest selection")
    by_path = {
        str((root / args.audio / row["_audio_name"]).resolve()): row
        for row in rows
    }
    if any(not Path(path).is_file() for path in by_path):
        missing = [path for path in by_path if not Path(path).is_file()]
        raise FileNotFoundError(f"Missing {len(missing)} selected WAV(s), first: {missing[0]}")
    context = ConstructionContextEngine.from_data_dir(
        Path(__file__).resolve().parents[1] / "data" / "onevoice_construction_v2"
    )
    pending = [row for row in rows if row["_audio_name"] not in completed]
    raw_log = output / "upstream_stdout.log"
    started = time.perf_counter()

    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        paths = [str((root / args.audio / row["_audio_name"]).resolve()) for row in batch]
        command = [
            args.python, str(args.infer_script), "--audio", *paths,
            "--version", "1", "--model-dir", str(args.model_dir),
            "--icefall-dir", str(args.icefall_dir), "--device", args.device,
            "--decoding-method", "greedy_search",
        ]
        print("[GIPFormer PyTorch] > " + " ".join(command), flush=True)
        run = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with raw_log.open("a", encoding="utf-8") as handle:
            handle.write("\n\n> " + " ".join(command) + "\n" + run.stdout)
        if run.returncode:
            raise RuntimeError(f"Official GIPFormer inference failed (exit {run.returncode}); see {raw_log}")
        parsed = parse_upstream_output(run.stdout)
        if set(parsed) != set(paths):
            raise ValueError(
                "Upstream output paths differ from requested batch; expected "
                f"{len(paths)}, parsed {len(parsed)}. See {raw_log}"
            )
        for path in paths:
            row = by_path[path]
            prediction, elapsed_s = parsed[path]
            reference = str(row["text"])
            ref_context = context.analyze(reference, "vi2en")
            hyp_context = context.analyze(prediction, "vi2en")
            ref_terms = {item.canonical_id for item in ref_context.canonical_mentions}
            hyp_terms = {item.canonical_id for item in hyp_context.canonical_mentions}
            critical_ref = {
                item.canonical_id for item in ref_context.canonical_mentions
                if item.risk_level in {"high", "critical"}
            }
            reference_safety = ref_context.safety_candidates[0] if ref_context.safety_candidates else None
            hypothesis_safety = hyp_context.safety_candidates[0] if hyp_context.safety_candidates else None
            predictions.append({
                "audio": row["_audio_name"], "reference": reference, "prediction": prediction,
                "latency_ms": round((elapsed_s or 0.0) * 1000, 3),
                "domain": row.get("domain", "unknown"), "risk_level": row.get("risk_level", "unknown"),
                "noise_type": row.get("noise_type", "clean"),
                "snr_db": row.get("snr_db", row.get("target_snr_db", "clean")),
                "term_hits": len(ref_terms & hyp_terms), "term_total": len(ref_terms),
                "critical_hits": len(critical_ref & hyp_terms), "critical_total": len(critical_ref),
                "safety_hits": int(bool(reference_safety and hypothesis_safety and reference_safety.safety_id == hypothesis_safety.safety_id)),
                "safety_total": int(reference_safety is not None),
            })
        save_partial(partial_path, predictions)
        print(f"[GIPFormer PyTorch] saved {len(predictions)}/{len(rows)} predictions", flush=True)

    aggregate = summarize(predictions)
    aggregate.update({"split": args.split, "audio": args.audio, "model": "official_gipformer_v1_pytorch"})
    gate = {"passed": False, "reason": "baseline aggregate was not supplied"}
    if args.baseline_aggregate:
        baseline = json.loads(args.baseline_aggregate.read_text(encoding="utf-8"))
        deltas = {metric: (aggregate[metric] - float(baseline[metric])) * 100 for metric in ("wer", "cer")}
        gate = {
            "passed": aggregate["samples"] == int(baseline["samples"]) and aggregate["empty_predictions"] == 0
            and all(delta <= args.max_regression_pp for delta in deltas.values()),
            "samples_match": aggregate["samples"] == int(baseline["samples"]),
            "empty_predictions": aggregate["empty_predictions"],
            "delta_pp": deltas,
            "max_regression_pp": args.max_regression_pp,
        }
    aggregate["compatibility_gate"] = gate
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader(); writer.writerows(predictions)
    (output / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    create_run_manifest(output / "run_manifest.json", "benchmark_gipformer_pytorch", [args.manifest, args.infer_script, args.model_dir], vars(args))
    print(json.dumps({**aggregate, "elapsed_seconds": round(time.perf_counter() - started, 2)}, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise SystemExit("GIPFormer compatibility gate did not pass; do not fine-tune this checkpoint.")


if __name__ == "__main__":
    main()
