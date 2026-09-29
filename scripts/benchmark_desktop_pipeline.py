"""Benchmark every local test WAV through the current offline streaming pipeline.

Run from a direction-scoped bundle. No sample limit is applied by default.
The two optional limits are for smoke runs only. Replay does not capture a
microphone or play thousands of sentences to speakers. Buffer-generation
latency is reported separately from device playback, which is not measured.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context.engine import ConstructionContextEngine
from evaluation.metrics import corpus_error_rate, normalize_metric_text
from evaluation.reporting import create_run_manifest
from profile_release_runtime import NoNetwork, RSSSampler, percentile
from run_release_e2e import atomic_json, read_jsonl, select_safety_audio_cases, sha256
from run_streaming_e2e import read_latency, validate_stream_result


def select_full_cases(manifest: Path, direction: str, audio_kind: str,
                      context: ConstructionContextEngine) -> list[dict]:
    language = "vi" if direction == "vi2en" else "en"
    cases, seen = [], set()
    for row in read_jsonl(manifest):
        if row.get("split") != "test" or row.get("language", language) != language:
            continue
        name = str(row.get("clean_audio") if audio_kind == "clean" else row.get("audio") or "")
        path = Path(name)
        if not path.is_absolute():
            nested = manifest.parent / audio_kind / path
            path = nested if nested.is_file() else manifest.parent / path
        if not name or not path.is_file():
            raise FileNotFoundError(f"Missing test audio (not skipped): {path}")
        path = path.resolve()
        if path in seen:
            continue
        text = str(row.get("text") or row.get("transcript") or "").strip()
        translation = str(row.get("translation") or "").strip()
        if not text or not translation:
            raise ValueError(f"Test reference text/translation is missing: {path}")
        seen.add(path)
        route = "safety" if context.analyze(text, direction).safety_candidates else "normal"
        cases.append({
            "case_id": f"{direction}-test-{audio_kind}-{len(cases) + 1:05d}",
            "suite": f"test_{audio_kind}", "direction": direction,
            "expected_route": route, "input_path": str(path),
            "reference_text": text, "reference_translation": translation,
            "source_id": str(row.get("utterance_id") or row.get("id") or path.stem),
            "domain": row.get("domain"), "risk_level": row.get("risk_level"),
            "noise_type": row.get("noise_type"),
            "snr_db": row.get("snr_db", row.get("target_snr_db")),
        })
    if not cases:
        raise ValueError(f"No unique {language}/test/{audio_kind} audio in {manifest}")
    return cases


def contract_for(config: Path, manifest: Path, cases: list[dict], direction: str,
                 backend: str, realtime: bool, full: bool) -> dict:
    code = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py")) + [
        Path(__file__), ROOT / "scripts/run_streaming_e2e.py",
        ROOT / "scripts/run_release_e2e.py", ROOT / "scripts/profile_release_runtime.py",
    ]
    for path in paths:
        code.update(path.relative_to(ROOT).as_posix().encode())
        code.update(bytes.fromhex(sha256(path)))
    versions = {}
    for name in ("numpy", "onnxruntime", "optimum-onnx", "transformers", "sherpa-onnx", "funasr-onnx", "pyttsx3"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema_version": 1, "direction": direction, "backend_requested": backend,
        "config_sha256": sha256(config), "manifest_sha256": sha256(manifest),
        "runtime_code_sha256": code.hexdigest(), "platform": platform.platform(),
        "python": sys.version, "dependencies": versions,
        "realtime": realtime, "full_selected_corpus": full, "cases": cases,
    }


def validate_resume_contract(path: Path, contract: dict) -> None:
    if json.loads(path.read_text(encoding="utf-8")) != contract:
        raise RuntimeError("Refusing resume: runtime/backend/data/environment changed; use a new output directory")


def measure_case(pipeline, case: dict, case_dir: Path, realtime: bool) -> dict:
    pipeline.report_dir = case_dir
    started = time.perf_counter()
    result = {**case, "status": "fail", "asr_text": "", "translation": "", "latency": []}
    try:
        report = pipeline.stream_file(case["input_path"], realtime=realtime)
        result.update(report)
        suppressed = [error for item in report.get("translations", [])
                      if item.get("suppressed") for error in item.get("validation_errors", [])]
        if suppressed:
            raise RuntimeError("Translation suppressed: " + ", ".join(suppressed))
        result["latency"] = read_latency(case_dir)
        validate_stream_result(case, report, result["latency"])
        # Validate the actual generated arrays as well as their report metadata.
        for chunk in pipeline._stream_chunks:
            if not np.isfinite(chunk.audio).all() or pipeline.tts.is_silence(chunk.audio):
                raise RuntimeError("Generated audio is silent or contains non-finite samples")
        result["status"] = "pass"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["latency"] = list(pipeline._latency_log)
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        raise
    finally:
        endpoints = [item["hypothesis"] for item in pipeline._stream_trace if item.get("endpoint")]
        result["asr_text"] = " ".join(endpoints) or (pipeline._stream_trace[-1]["hypothesis"] if pipeline._stream_trace else "")
        result["committed_source"] = " ".join(entry.original_text for entry in pipeline.srt._entries)
        result["translation"] = " ".join(entry.translated_text for entry in pipeline.srt._entries)
        result["tts_engines"] = sorted({chunk.engine for chunk in pipeline._stream_chunks})
        result["runner_elapsed_ms"] = (time.perf_counter() - started) * 1000
        # Check critical fields against the reference, including ASR-induced loss.
        context = pipeline.context.analyze(case["reference_text"], case["direction"])
        result["reference_validation_errors"] = pipeline.context.validate_translation(
            result["translation"], context, case["direction"])
        result["reference_critical_fields_valid"] = bool(result["translation"]) and not result["reference_validation_errors"]
        result["reference_translation_exact"] = normalize_metric_text(result["translation"]) == normalize_metric_text(case["reference_translation"])
        atomic_json(case_dir / "result.json", result)
    return result


def summarize_results(rows: list[dict]) -> dict:
    latency = [item for row in rows if row["status"] == "pass" for item in row.get("latency", [])]
    summary = {
        "samples": len(rows), "passed": sum(row["status"] == "pass" for row in rows),
        "failed": sum(row["status"] != "pass" for row in rows),
        "asr_wer": corpus_error_rate([r["reference_text"] for r in rows], [r["asr_text"] for r in rows]),
        "asr_cer": corpus_error_rate([r["reference_text"] for r in rows], [r["asr_text"] for r in rows], "char"),
        "translation_reference_wer": corpus_error_rate([r["reference_translation"] for r in rows], [r["translation"] for r in rows]),
        "reference_critical_field_preservation": sum(r["reference_critical_fields_valid"] for r in rows) / len(rows),
        "translation_exact_match": sum(r["reference_translation_exact"] for r in rows) / len(rows),
        "empty_outputs": sum(not r["translation"] for r in rows),
        "tts_engines": dict(Counter(engine for row in rows for engine in row.get("tts_engines", []))),
    }
    for metric in ("commit_to_first_audio_ms", "speech_to_commit_ms", "asr_ms", "mt_ms", "tts_ms"):
        values = [float(item[metric]) for item in latency if item.get(metric) is not None]
        summary[metric.replace("_ms", "_p50_ms")] = percentile(values, 0.50)
        summary[metric.replace("_ms", "_p95_ms")] = percentile(values, 0.95)
    values = [float(row["complete_turn_ms"]) for row in rows if row.get("complete_turn_ms") is not None and row["status"] == "pass"]
    summary["complete_turn_p50_ms"], summary["complete_turn_p95_ms"] = percentile(values, 0.50), percentile(values, 0.95)
    return summary


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--direction", required=True, choices=("vi2en", "en2vi"))
    parser.add_argument("--audio", default="noisy", choices=("clean", "noisy"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tts-backend", default="auto", choices=("auto", "sapi", "espeak"))
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit-test-cases", type=int)
    parser.add_argument("--limit-safety-cases", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.progress_every <= 0 or any(v is not None and v <= 0 for v in (args.limit_test_cases, args.limit_safety_cases)):
        parser.error("progress interval and smoke limits must be positive")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_dir = Path(cfg["pipeline"]["construction_data_dir"])
    safety_csv = Path(cfg["pipeline"]["safety_source_csv"])
    safety_manifest = Path(cfg["pipeline"]["safety_audio_manifest"])
    context = ConstructionContextEngine.from_data_dir(str(data_dir), safety_path=str(safety_csv))
    test_cases = select_full_cases(args.manifest, args.direction, args.audio, context)
    import csv
    with safety_csv.open(encoding="utf-8-sig", newline="") as handle:
        safety_rows = list(csv.DictReader(handle))
    canonical_ids = {entry["safety_id"] for entry in json.loads(safety_manifest.read_text(encoding="utf-8"))["entries"]}
    source_by_id = {row["safety_id"]: row for row in safety_rows}
    if any(source_by_id.get(key, {}).get("review_status") != "approved" for key in canonical_ids):
        raise ValueError("Every canonical safety audio entry must have an approved source row")
    # The CSV also contains 70 unapproved text variants without runtime audio.
    # They are not part of the 126 approved canonical audio entries.
    safety_cases = select_safety_audio_cases(safety_manifest, safety_csv, args.direction, len(canonical_ids))
    for case in safety_cases:
        case["suite"] = "approved_safety"
    available = {f"test_{args.audio}": len(test_cases), "approved_safety": len(safety_cases)}
    cases = test_cases[:args.limit_test_cases] + safety_cases[:args.limit_safety_cases]
    for case in cases:
        case["input_sha256"] = sha256(Path(case["input_path"]))
    full = args.limit_test_cases is None and args.limit_safety_cases is None
    contract = contract_for(args.config, args.manifest, cases, args.direction, args.tts_backend, args.realtime, full)
    # Bind model/data hashes advertised by the portable bundle into the run.
    artifact_manifest = Path(cfg["pipeline"]["artifact_manifest"])
    contract["artifact_manifest_sha256"] = sha256(artifact_manifest)
    contract["context_data_sha256"] = {p.name: sha256(p) for p in sorted(data_dir.glob("*.csv"))}
    contract["safety_manifest_sha256"] = sha256(safety_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "benchmark_contract.json"
    if contract_path.is_file():
        if not args.resume:
            raise RuntimeError("Output already contains a run; pass --resume or choose a new output directory")
        validate_resume_contract(contract_path, contract)
    else:
        atomic_json(contract_path, contract)
        create_run_manifest(args.output_dir / "run_manifest.json", "benchmark_desktop_pipeline",
                            [args.config, args.manifest, artifact_manifest, safety_csv, safety_manifest],
                            {"backend_requested": args.tts_backend, "full_selected_corpus": full, "available_cases": available})
    print(json.dumps({"direction": args.direction, "available": available, "selected": len(cases),
                      "full_selected_corpus": full, "realtime": args.realtime}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    session_manifest = create_run_manifest(
        args.output_dir / f"session_{time.time_ns()}.json", "benchmark_desktop_pipeline",
        [args.config, args.manifest, artifact_manifest, safety_csv, safety_manifest],
        {"backend_requested": args.tts_backend, "full_selected_corpus": full,
         "benchmark_contract_sha256": sha256(contract_path), "available_cases": available})
    rows = []
    completed = {}
    for case in cases:
        path = args.output_dir / "cases" / case["case_id"] / "result.json"
        if args.resume and path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("input_sha256") != case["input_sha256"] or row.get("case_id") != case["case_id"]:
                raise RuntimeError(f"Corrupt/mismatched resume case: {path}")
            if row.get("status") in {"pass", "fail"}:
                completed[case["case_id"]] = row
    print(f"[benchmark] saved cases={len(completed)}; pending={len(cases) - len(completed)}", flush=True)
    started = time.perf_counter()
    log = args.output_dir / "runtime.log"
    with NoNetwork(), RSSSampler() as rss, log.open("a", encoding="utf-8", buffering=1) as output:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            from pipeline import OneVoicePipeline
            pipeline = OneVoicePipeline(config_path=str(args.config), direction=args.direction,
                                        profile="edge", offline=True, tts_backend=args.tts_backend)
            load_started = time.perf_counter()
            pipeline.load_models()
            load_ms = (time.perf_counter() - load_started) * 1000
        print(f"[benchmark] models ready in {load_ms / 1000:.1f}s; TTS={pipeline.tts.engine_name(args.direction)}", flush=True)
        new_cases = 0
        for index, case in enumerate(cases, 1):
            if case["case_id"] in completed:
                rows.append(completed[case["case_id"]])
            else:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    rows.append(measure_case(pipeline, case, args.output_dir / "cases" / case["case_id"], args.realtime))
                new_cases += 1
            if index % args.progress_every == 0 or index == len(cases):
                elapsed = time.perf_counter() - started
                eta = (len(cases) - index) * elapsed / max(1, new_cases)
                state = {"requested": len(cases), "completed": index,
                         "passed": sum(r["status"] == "pass" for r in rows),
                         "failed": sum(r["status"] != "pass" for r in rows),
                         "elapsed_seconds": round(elapsed, 2), "eta_seconds": round(eta, 2), "finished": False}
                atomic_json(args.output_dir / "progress.json", state)
                print(f"[benchmark] {index}/{len(cases)} passed={state['passed']} failed={state['failed']} ETA={eta / 60:.1f} min", flush=True)
    summary = {"schema_version": 1, "direction": args.direction, "profile": "edge",
               "git_revision": session_manifest["git_revision"],
               "offline": True, "network_blocked_in_python": True,
               "full_selected_corpus": full, "available_cases": available,
               "requested_cases": len(cases), "completed_cases": len(rows),
               "passed_cases": sum(r["status"] == "pass" for r in rows),
               "failed_cases": sum(r["status"] != "pass" for r in rows),
               "measurement_completed": len(rows) == len(cases),
               "load_time_ms": load_ms, "peak_process_rss_mb": rss.peak_mb,
               "memory_scope": "Python process RSS; child PowerShell SAPI memory excluded",
               "elapsed_seconds": time.perf_counter() - started,
               "realtime": args.realtime, "normal_commit_policy": pipeline.committer.normal_commit_policy,
               "tts_backend_requested": args.tts_backend, "tts_engine": pipeline.tts.engine_name(args.direction),
               "sapi_voice": (pipeline.tts._sapi_voice_name if args.direction == "vi2en"
                              else pipeline.tts._sapi_vi_voice_name),
               "suites": {suite: summarize_results([r for r in rows if r["suite"] == suite]) for suite in available},
               "routes": {route: summarize_results(group) for route in ("normal", "safety")
                          if (group := [r for r in rows if r["expected_route"] == route])},
               "failures": [{"case_id": r["case_id"], "error": r.get("error")} for r in rows if r["status"] != "pass"],
               "scope": "Full selected synthetic WAV corpus replay; no live microphone, sound-device or Bluetooth latency measurement. Percentiles include successful outputs; quality rates include failures as empty outputs."}
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "progress.json", {"completed": len(rows), "requested": len(cases), "finished": True})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["failed_cases"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
