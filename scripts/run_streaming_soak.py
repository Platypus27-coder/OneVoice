"""Run a resumable real-time streaming soak on fixed normal and safety cases.

The runner deliberately replays the same small, fixed held-out suite in a
round-robin loop.  It is a reliability test, not a new quality benchmark:
every turn must maintain the normal/safety route, commit order, worker health
and frame invariants already checked by ``run_streaming_e2e.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from context.engine import ConstructionContextEngine
from pipeline import OneVoicePipeline
from run_release_e2e import atomic_json, select_cases, select_safety_audio_cases
from run_streaming_e2e import read_latency, validate_stream_result


def process_rss_mb() -> float | None:
    """Best-effort peak resident memory; available on Colab/Linux only."""
    try:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except (ImportError, AttributeError):
        return None


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def initial_state(target_seconds: float) -> dict:
    return {
        "schema_version": 1,
        "target_seconds": target_seconds,
        "elapsed_seconds": 0.0,
        "completed_turns": 0,
        "passed_turns": 0,
        "failed_turns": 0,
        "last_error": None,
        "finished": False,
    }


def load_state(path: Path, target_seconds: float, resume: bool) -> dict:
    if not resume or not path.is_file():
        return initial_state(target_seconds)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if float(payload.get("target_seconds", target_seconds)) != target_seconds:
        raise RuntimeError("Refusing resume: requested soak duration changed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--vi-manifest", required=True, type=Path)
    parser.add_argument("--en-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--safety-manifest", type=Path)
    parser.add_argument("--safety-csv", type=Path)
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--profile", choices=["development", "edge", "premium"], default="development")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.duration_minutes <= 0:
        parser.error("--duration-minutes must be positive")
    for path in (args.config, args.vi_manifest, args.en_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_dir = config["pipeline"].get("construction_data_dir", "data/onevoice_construction_v2")
    safety_csv = Path(args.safety_csv or config["pipeline"].get("safety_source_csv") or Path(data_dir) / "safety_fast_path.csv")
    safety_manifest = Path(args.safety_manifest or config["pipeline"].get("safety_audio_manifest", "artifacts/safety_audio/manifest.json"))
    if not safety_csv.is_file() or not safety_manifest.is_file():
        raise FileNotFoundError("Verified safety source CSV and audio manifest are required")

    target_seconds = args.duration_minutes * 60.0
    state_path = args.output_dir / "soak_state.json"
    events_path = args.output_dir / "events.jsonl"
    state = load_state(state_path, target_seconds, args.resume)
    if state.get("finished"):
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    context = ConstructionContextEngine.from_data_dir(str(data_dir), safety_path=str(safety_csv))
    cases = []
    for direction, manifest in (("vi2en", args.vi_manifest), ("en2vi", args.en_manifest)):
        cases.extend(select_cases(manifest, direction, context, 1, routes=("normal",)))
        cases.extend(select_safety_audio_cases(safety_manifest, safety_csv, direction, 1))
    atomic_json(args.output_dir / "case_manifest.json", {"schema_version": 1, "cases": cases})
    pipelines = {
        direction: OneVoicePipeline(
            config_path=str(args.config),
            direction=direction,
            profile=args.profile,
            offline=args.offline,
            report_dir=str(args.output_dir / direction),
        )
        for direction in ("vi2en", "en2vi")
    }

    while float(state["elapsed_seconds"]) < target_seconds:
        turn_id = int(state["completed_turns"]) + 1
        case = cases[(turn_id - 1) % len(cases)]
        case_dir = args.output_dir / "turns" / f"{turn_id:06d}_{case['case_id']}"
        pipeline = pipelines[case["direction"]]
        pipeline.report_dir = case_dir
        started = time.perf_counter()
        event = {
            "turn_id": turn_id,
            "case_id": case["case_id"],
            "direction": case["direction"],
            "expected_route": case["expected_route"],
            "started_at": time.time(),
        }
        try:
            report = pipeline.stream_file(case["input_path"], realtime=args.realtime)
            latency = read_latency(case_dir)
            validate_stream_result(case, report, latency)
            event.update(
                status="pass",
                commits=report["commits"],
                dropped_audio_frames=report["dropped_audio_frames"],
                complete_turn_ms=report["complete_turn_ms"],
                routes=sorted({str(row.get("route", "")) for row in latency}),
            )
            state["passed_turns"] = int(state["passed_turns"]) + 1
        except Exception as exc:
            event.update(status="fail", error=f"{type(exc).__name__}: {exc}")
            state["failed_turns"] = int(state["failed_turns"]) + 1
            state["last_error"] = event["error"]
        elapsed = time.perf_counter() - started
        event["runner_elapsed_seconds"] = elapsed
        event["peak_rss_mb"] = process_rss_mb()
        append_jsonl(events_path, event)
        state["elapsed_seconds"] = float(state["elapsed_seconds"]) + elapsed
        state["completed_turns"] = turn_id
        atomic_json(state_path, state)
        print(json.dumps(event, ensure_ascii=False), flush=True)
        if event["status"] == "fail":
            break

    state["finished"] = (
        int(state["failed_turns"]) == 0 and float(state["elapsed_seconds"]) >= target_seconds
    )
    atomic_json(state_path, state)
    summary = {
        **state,
        "events": str(events_path.resolve()),
        "realtime": args.realtime,
        "profile": args.profile,
        "offline": args.offline,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if int(state["failed_turns"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
