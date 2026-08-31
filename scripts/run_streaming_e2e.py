"""Run resumable fixed normal/safety streaming gates through the public runtime.

Unlike ``run_release_e2e.py`` (file-mode), this script feeds every selected
WAV through the 32 ms worker graph exposed by ``OneVoicePipeline.stream_file``.
It records per-turn reports on Drive and rejects unsafe routing, duplicate or
reordered commits, dropped frames, silent/no-output normal turns, and worker
errors.  The test split is never used for training by this runner.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline import OneVoicePipeline
from run_release_e2e import (
    atomic_json,
    select_cases,
    select_safety_audio_cases,
)
from context.engine import ConstructionContextEngine


def valid_resume(path: Path, expected_route: str) -> bool:
    """Return true only for a completed, valid prior streaming case."""
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            payload.get("status") == "pass"
            and payload.get("expected_route") == expected_route
            and int(payload.get("commits", 0)) >= 1
            and not payload.get("fatal_error")
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def validate_stream_result(case: dict, result: dict, latency: list[dict]) -> None:
    """Validate invariants that must hold for every streamed turn."""
    if result.get("direction") != case["direction"]:
        raise RuntimeError("Report direction does not match the requested case")
    if result.get("fatal_error"):
        raise RuntimeError(f"Streaming worker failed: {result['fatal_error']}")
    if int(result.get("dropped_audio_frames", 0)):
        raise RuntimeError("Streaming dropped audio frames")

    commits = [int(value) for value in result.get("commit_ids", [])]
    chunks = result.get("chunks", [])
    if not commits or len(commits) != len(chunks):
        raise RuntimeError("Every stream turn must produce one ordered audio chunk per commit")
    if commits != sorted(commits) or len(commits) != len(set(commits)):
        raise RuntimeError("Streaming committed duplicate or reordered output")
    if any(int(chunk.get("samples", 0)) <= 0 for chunk in chunks):
        raise RuntimeError("Streaming emitted an empty audio chunk")
    if any(row.get("validation_errors") for row in latency):
        raise RuntimeError("Translation critical-field validator rejected this stream turn")

    decisions = [str(item.get("decision", "")) for item in result.get("hypothesis_trace", [])]
    routes = {str(row.get("route", "")) for row in latency}
    engines = {str(chunk.get("engine", "")) for chunk in chunks}
    if case["expected_route"] == "safety":
        if decisions.count("COMMIT_SAFETY") != 1:
            raise RuntimeError("Safety stream must create exactly one safety commit")
        if "safety_audio" not in routes or engines != {"safety_audio"}:
            raise RuntimeError("Safety stream did not use the verified local safety audio")
    else:
        if "COMMIT_NORMAL" not in decisions:
            raise RuntimeError("Normal stream did not produce a normal semantic commit")
        if any(route in {"safety_fast_path", "safety_audio"} for route in routes):
            raise RuntimeError("Normal stream incorrectly used a safety route")
        if "safety_audio" in engines:
            raise RuntimeError("Normal stream incorrectly emitted safety audio")


def read_latency(case_dir: Path) -> list[dict]:
    path = case_dir / "latency.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing latency report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Streaming latency report is empty")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--vi-manifest", required=True, type=Path)
    parser.add_argument("--en-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--safety-manifest", type=Path)
    parser.add_argument("--safety-csv", type=Path)
    parser.add_argument("--cases-per-route", type=int, default=1)
    parser.add_argument("--profile", choices=["development", "edge", "premium"], default="development")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.cases_per_route <= 0:
        parser.error("--cases-per-route must be positive")
    for path in (args.config, args.vi_manifest, args.en_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_dir = config["pipeline"].get("construction_data_dir", "data/onevoice_construction_v2")
    safety_csv = Path(args.safety_csv or config["pipeline"].get("safety_source_csv") or Path(data_dir) / "safety_fast_path.csv")
    safety_manifest = Path(args.safety_manifest or config["pipeline"].get("safety_audio_manifest", "artifacts/safety_audio/manifest.json"))
    if not safety_csv.is_file() or not safety_manifest.is_file():
        raise FileNotFoundError("Verified safety source CSV and audio manifest are required")

    context = ConstructionContextEngine.from_data_dir(str(data_dir), safety_path=str(safety_csv))
    manifests = {"vi2en": args.vi_manifest, "en2vi": args.en_manifest}
    cases = []
    for direction, manifest in manifests.items():
        cases.extend(select_cases(manifest, direction, context, args.cases_per_route, routes=("normal",)))
        cases.extend(select_safety_audio_cases(safety_manifest, safety_csv, direction, args.cases_per_route))
    atomic_json(args.output_dir / "case_manifest.json", {"schema_version": 1, "cases": cases})

    results: list[dict] = []
    failures: list[dict] = []
    for direction in ("vi2en", "en2vi"):
        pipeline = OneVoicePipeline(
            config_path=str(args.config),
            direction=direction,
            profile=args.profile,
            offline=args.offline,
            report_dir=str(args.output_dir / direction),
        )
        for case in (item for item in cases if item["direction"] == direction):
            case_dir = args.output_dir / direction / case["case_id"]
            result_path = case_dir / "result.json"
            if args.resume and valid_resume(result_path, case["expected_route"]):
                print(f"[resume] {case['case_id']}", flush=True)
                results.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            pipeline.report_dir = case_dir
            started = time.perf_counter()
            try:
                report = pipeline.stream_file(case["input_path"])
                latency = read_latency(case_dir)
                validate_stream_result(case, report, latency)
                result = {
                    **case,
                    **report,
                    "latency": latency,
                    "status": "pass",
                    "runner_elapsed_ms": (time.perf_counter() - started) * 1000,
                }
                atomic_json(result_path, result)
                results.append(result)
                print(f"[PASS] {case['case_id']} commits={result['commits']}", flush=True)
            except Exception as exc:
                failure = {
                    **case,
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                    "runner_elapsed_ms": (time.perf_counter() - started) * 1000,
                }
                atomic_json(result_path, failure)
                failures.append(failure)
                print(f"[FAIL] {case['case_id']}: {failure['error']}", flush=True)
                break
        if failures:
            break

    summary = {
        "schema_version": 1,
        "config": str(args.config.resolve()),
        "offline": args.offline,
        "profile": args.profile,
        "requested_cases": len(cases),
        "passed": len(results),
        "failed": len(failures),
        "routes": {
            route: sum(item["expected_route"] == route for item in results)
            for route in ("normal", "safety")
        },
        "failures": failures,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
