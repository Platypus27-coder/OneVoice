"""Replay known failures, every approved safety case and deterministic controls.

This is a regression check after runtime fixes, not a fresh full-test accuracy
claim. Reference text is used only for selection/validation, never for inference.
The original WAVs, configuration and pass criteria are retained.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

from benchmark_desktop_pipeline import contract_for, measure_case, summarize_results, validate_resume_contract
from evaluation.reporting import create_run_manifest
from profile_release_runtime import NoNetwork, RSSSampler
from run_release_e2e import atomic_json, sha256


def select_regression_cases(cases: list[dict], baseline: dict[str, dict], controls: int) -> list[dict]:
    selected = {row["case_id"] for row in cases
                if row["suite"] == "approved_safety" or baseline[row["case_id"]]["status"] == "fail"}
    candidates = sorted((row for row in cases if row["case_id"] not in selected),
                        key=lambda row: hashlib.sha256(row["case_id"].encode()).hexdigest())
    selected.update(row["case_id"] for row in candidates[:controls])
    return [row for row in cases if row["case_id"] in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controls", type=int, default=100, help="Previously passing noisy controls per direction")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.controls < 1:
        parser.error("controls must be positive")
    source = args.baseline_dir / "benchmark_contract.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    baseline_summary = json.loads((args.baseline_dir / "summary.json").read_text(encoding="utf-8"))
    if not original["full_selected_corpus"] or not baseline_summary["measurement_completed"]:
        raise ValueError("Baseline must be a completed full-corpus run")
    if sha256(args.config) != original["config_sha256"]:
        raise ValueError("Configuration differs from baseline; use the original config for paired replay")
    baseline = {row["case_id"]: json.loads((args.baseline_dir / "cases" / row["case_id"] / "result.json").read_text(encoding="utf-8"))
                for row in original["cases"]}
    cases = select_regression_cases(original["cases"], baseline, args.controls)
    # Measure against the same corpus, including all failures. Verify waveform
    # identity before loading models; references do not enter the runtime.
    for row in cases:
        if sha256(Path(row["input_path"])) != row["input_sha256"]:
            raise ValueError(f"Baseline WAV changed: {row['case_id']}")
    contract = contract_for(args.config, source, cases, original["direction"], original["backend_requested"],
                            original["realtime"], False)
    contract.update(baseline_contract_sha256=sha256(source), runner_sha256=sha256(Path(__file__)),
                    controls=args.controls, scope="known-failure regression and approved safety coverage")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "benchmark_contract.json"
    if contract_path.is_file():
        if not args.resume:
            raise ValueError("Output already exists; pass --resume or use a new directory")
        validate_resume_contract(contract_path, contract)
    else:
        atomic_json(contract_path, contract)
    manifest = create_run_manifest(args.output_dir / "run_manifest.json", "run_desktop_regression",
                                   [source, args.config], {"controls": args.controls, "full_selected_corpus": False})
    print(f"[regression] {original['direction']}: {len(cases)} cases; all approved safety and baseline failures included", flush=True)
    rows = []
    started = time.perf_counter()
    with NoNetwork(), RSSSampler() as rss, (args.output_dir / "runtime.log").open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            from pipeline import OneVoicePipeline
            pipeline = OneVoicePipeline(config_path=str(args.config), direction=original["direction"],
                                        profile="edge", offline=True, tts_backend=original["backend_requested"])
            pipeline.load_models()
        for index, case in enumerate(cases, 1):
            case_dir = args.output_dir / "cases" / case["case_id"]
            result_path = case_dir / "result.json"
            if args.resume and result_path.is_file():
                row = json.loads(result_path.read_text(encoding="utf-8"))
                if row["input_sha256"] != case["input_sha256"]:
                    raise ValueError(f"Mismatched saved waveform: {case['case_id']}")
            else:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    row = measure_case(pipeline, case, case_dir, original["realtime"])
            row["baseline_status"] = baseline[case["case_id"]]["status"]
            rows.append(row)
            if index % 10 == 0 or index == len(cases):
                progress = {"completed": index, "requested": len(cases), "finished": False,
                            "passed": sum(row["status"] == "pass" for row in rows),
                            "elapsed_seconds": round(time.perf_counter() - started, 2)}
                atomic_json(args.output_dir / "progress.json", progress)
                print(f"[regression] {index}/{len(cases)} passed={progress['passed']}", flush=True)
    recovered = [row["case_id"] for row in rows if row["baseline_status"] == "fail" and row["status"] == "pass"]
    regressions = [row["case_id"] for row in rows if row["baseline_status"] == "pass" and row["status"] != "pass"]
    summary = {"schema_version": 1, "git_revision": manifest["git_revision"],
               "direction": original["direction"], "full_selected_corpus": False,
               "measurement_completed": len(rows) == len(cases), "offline": True,
               "network_blocked_in_python": True, "selected_cases": len(cases),
               "baseline_failed": sum(row["baseline_status"] == "fail" for row in rows),
               "current_failed": sum(row["status"] != "pass" for row in rows),
               "recovered": recovered, "regressions": regressions,
               "suites": {suite: summarize_results([row for row in rows if row["suite"] == suite])
                          for suite in sorted({row["suite"] for row in rows})},
               "failures": [{"case_id": row["case_id"], "reference_text": row["reference_text"],
                             "asr_text": row["asr_text"], "translation": row["translation"],
                             "error": row.get("error")} for row in rows if row["status"] != "pass"],
               "elapsed_seconds": time.perf_counter() - started, "peak_process_rss_mb": rss.peak_mb,
               "scope": "Post-fix regression on known failures, all canonical safety and deterministic passing controls; not an unbiased full-test accuracy estimate."}
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "progress.json", {"completed": len(rows), "requested": len(cases), "finished": True})
    print(json.dumps({key: summary[key] for key in ("direction", "selected_cases", "baseline_failed", "current_failed", "regressions")}
                     | {"recovered": len(recovered)}, ensure_ascii=False), flush=True)
    if summary["current_failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
