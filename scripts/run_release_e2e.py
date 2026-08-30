"""Run resumable normal/safety file-mode E2E gates with the public runtime."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context.engine import ConstructionContextEngine
from pipeline import OneVoicePipeline


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid manifest row {path}:{line_number}")
            rows.append(row)
    return rows


def clean_audio_path(manifest: Path, row: dict) -> Path:
    raw = str(row.get("clean_audio") or row.get("audio") or "").strip()
    if not raw:
        raise ValueError("Manifest row is missing clean_audio/audio")
    path = Path(raw)
    if path.is_absolute():
        return path
    clean = manifest.parent / "clean" / path
    return clean if clean.is_file() else manifest.parent / path


def select_cases(
    manifest: Path,
    direction: str,
    context: ConstructionContextEngine,
    count: int,
    routes: tuple[str, ...] = ("normal", "safety"),
) -> list[dict]:
    selected = {"normal": [], "safety": []}
    seen_audio: set[str] = set()
    expected_language = "vi" if direction == "vi2en" else "en"
    for row in read_jsonl(manifest):
        if str(row.get("split")) != "test":
            continue
        if row.get("language") and row.get("language") != expected_language:
            continue
        text = str(row.get("text") or row.get("transcript") or "").strip()
        if not text:
            continue
        path = clean_audio_path(manifest, row)
        key = str(path.resolve())
        if key in seen_audio or not path.is_file():
            continue
        analysis = context.analyze(text, direction)
        route = "safety" if analysis.safety_candidates else "normal"
        if len(selected[route]) >= count:
            continue
        seen_audio.add(key)
        selected[route].append(
            {
                "case_id": f"{direction}-{route}-{len(selected[route]) + 1:03d}",
                "direction": direction,
                "expected_route": route,
                "input_path": str(path.resolve()),
                "reference_text": text,
                "reference_translation": str(row.get("translation", "")),
                "source_id": str(row.get("utterance_id") or row.get("id") or path.stem),
            }
        )
        if all(len(values) >= count for values in selected.values()):
            break
    shortages = {
        route: count - len(selected[route])
        for route in routes
        if len(selected[route]) < count
    }
    if shortages:
        raise ValueError(
            f"{manifest} does not contain enough unique test cases for {direction}: {shortages}"
        )
    return [case for route in routes for case in selected[route]]


def select_safety_audio_cases(
    safety_manifest: Path,
    safety_csv: Path,
    direction: str,
    count: int,
) -> list[dict]:
    """Select reviewed, checksum-verified safety WAVs as a fixed holdout."""
    payload = json.loads(safety_manifest.read_text(encoding="utf-8"))
    entries = {
        (str(row.get("safety_id")), str(row.get("direction"))): row
        for row in payload.get("entries", [])
    }
    source_rows = {}
    with safety_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source_rows[str(row.get("safety_id", ""))] = row
    input_direction = "en2vi" if direction == "vi2en" else "vi2en"
    cases = []
    for safety_id in sorted(source_rows):
        row = source_rows[safety_id]
        entry = entries.get((safety_id, input_direction))
        if not entry:
            continue
        path = Path(str(entry.get("path", "")))
        if not path.is_absolute():
            path = safety_manifest.parent / path
        if not path.is_file():
            continue
        text = str(row.get("vi" if direction == "vi2en" else "en", "")).strip()
        if not text:
            continue
        cases.append(
            {
                "case_id": f"{direction}-safety-{len(cases) + 1:03d}",
                "direction": direction,
                "expected_route": "safety",
                "safety_id": safety_id,
                "input_path": str(path.resolve()),
                "reference_text": text,
                "reference_translation": str(
                    row.get("en" if direction == "vi2en" else "vi", "")
                ),
                "source_id": safety_id,
            }
        )
        if len(cases) >= count:
            break
    if len(cases) < count:
        raise ValueError(
            f"Safety manifest has only {len(cases)} usable {direction} cases; "
            f"need {count}"
        )
    return cases


def valid_resume(result_path: Path) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        output = Path(result["output_path"])
        return (
            result.get("status") == "pass"
            and output.is_file()
            and sha256(output) == result.get("output_sha256")
        )
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_result(case: dict, result: dict, output: Path) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for release E2E WAV validation") from exc
    info = sf.info(output)
    if info.frames <= 0 or info.channels != 1 or info.samplerate <= 0:
        raise RuntimeError(f"Invalid output WAV: {output}")
    audio, _ = sf.read(output, dtype="float32", always_2d=False)
    if not len(audio) or float(abs(audio).max()) <= 1e-6:
        raise RuntimeError(f"Silent output WAV: {output}")
    route = str(result.get("route", ""))
    if case["expected_route"] == "safety":
        if route != "safety_audio" or not result.get("safety_id"):
            raise RuntimeError(f"Expected safety_audio route, got {route!r}")
    elif not route.startswith("normal_"):
        raise RuntimeError(f"Expected normal route, got {route!r}")
    if result.get("validation_errors"):
        raise RuntimeError(f"Critical-field validation failed: {result['validation_errors']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--vi-manifest", required=True, type=Path)
    parser.add_argument("--en-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--safety-manifest",
        type=Path,
        help="Reviewed safety audio manifest; defaults to pipeline.safety_audio_manifest.",
    )
    parser.add_argument(
        "--safety-csv",
        type=Path,
        help="Reviewed safety source CSV; defaults to pipeline.safety_source_csv.",
    )
    parser.add_argument("--cases-per-route", type=int, default=20)
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
    safety_csv = config["pipeline"].get("safety_source_csv") or str(
        Path(data_dir) / "safety_fast_path.csv"
    )
    context = ConstructionContextEngine.from_data_dir(data_dir, safety_path=safety_csv)
    safety_manifest = args.safety_manifest or Path(
        config["pipeline"].get("safety_audio_manifest", "artifacts/safety_audio/manifest.json")
    )
    safety_csv_path = args.safety_csv or Path(safety_csv)
    if not safety_manifest.is_file():
        raise FileNotFoundError(f"Safety audio manifest not found: {safety_manifest}")
    if not safety_csv_path.is_file():
        raise FileNotFoundError(f"Safety source CSV not found: {safety_csv_path}")
    manifests = {"vi2en": args.vi_manifest, "en2vi": args.en_manifest}
    cases = [
        case
        for direction, manifest in manifests.items()
        for case in (
            select_cases(manifest, direction, context, args.cases_per_route, routes=("normal",))
            + select_safety_audio_cases(
                safety_manifest, safety_csv_path, direction, args.cases_per_route
            )
        )
    ]
    atomic_json(args.output_dir / "case_manifest.json", {"schema_version": 1, "cases": cases})

    results = []
    failures = []
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
            if args.resume and valid_resume(result_path):
                print(f"[resume] {case['case_id']}", flush=True)
                results.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            output = case_dir / "output.wav"
            pipeline.report_dir = case_dir
            started = time.perf_counter()
            try:
                pipeline.process_file(case["input_path"], str(output))
                result = {**case, **(pipeline.last_file_result or {})}
                validate_result(case, result, output)
                if case["expected_route"] == "safety":
                    safety_source = (
                        pipeline.safety_audio.path_for(result["safety_id"], direction)
                        if pipeline.safety_audio
                        else None
                    )
                    if safety_source is None:
                        raise RuntimeError("Safety route has no verified source WAV")
                    expected_safety_sha256 = sha256(safety_source)
                    if sha256(output) != expected_safety_sha256:
                        raise RuntimeError("Safety output is not byte-exact with its manifest WAV")
                    result["safety_source_sha256"] = expected_safety_sha256
                result.update(
                    status="pass",
                    output_path=str(output.resolve()),
                    output_sha256=sha256(output),
                    runner_elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                atomic_json(result_path, result)
                results.append(result)
                print(f"[PASS] {case['case_id']} route={result['route']}", flush=True)
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
            route: sum(result.get("route") == route for result in results)
            for route in sorted({str(result.get("route")) for result in results})
        },
        "failures": failures,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures or len(results) != len(cases):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
