"""Measure cumulative RSS after each direction-specific runtime stage loads.

The deltas are allocator-aware observations, not model-file sizes.  They are
useful for selecting a physical device and must be collected again on that
device before a mobile claim is made.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

try:
    from scripts.profile_release_runtime import NoNetwork, RSSSampler
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from profile_release_runtime import NoNetwork, RSSSampler


def _sample(name: str, loader, sampler: RSSSampler, previous_mb: float) -> dict:
    started = time.perf_counter()
    loader()
    current_mb = sampler.peak_mb
    return {
        "stage": name,
        "load_time_ms": round((time.perf_counter() - started) * 1000, 3),
        "rss_after_mb": round(current_mb, 3),
        "rss_delta_mb": round(max(0.0, current_mb - previous_mb), 3),
    }


def profile(config_path: Path, direction: str, profile_name: str) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not bool(config.get("pipeline", {}).get("offline")):
        raise ValueError("Component profiling requires pipeline.offline=true")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pipeline import OneVoicePipeline
    from runtime.preflight import verify_artifacts

    manifest = config["pipeline"].get("artifact_manifest", "artifacts/manifest.json")
    preflight = verify_artifacts(manifest, direction, profile_name, sample_rate=int(config["audio"]["sample_rate"]))
    with NoNetwork(), RSSSampler() as sampler:
        pipeline = OneVoicePipeline(
            config_path=str(config_path), direction=direction,
            profile=profile_name, offline=True,
        )
        previous_mb = sampler.peak_mb
        stages = [{"stage": "pipeline_init_context_safety", "load_time_ms": 0.0, "rss_after_mb": round(previous_mb, 3), "rss_delta_mb": 0.0}]
        for name, loader in (
            ("denoiser", pipeline.denoiser.load),
            ("asr", lambda: pipeline.asr.load(direction=direction)),
            ("translation", pipeline.translator.load),
            ("tts", lambda: pipeline.tts.load(direction=direction)),
        ):
            row = _sample(name, loader, sampler, previous_mb)
            stages.append(row)
            previous_mb = row["rss_after_mb"]
    return {
        "schema_version": 1,
        "direction": direction,
        "profile": profile_name,
        "offline": True,
        "network_blocked": True,
        "preflight_checked": len(preflight["checked"]),
        "stages": stages,
        "peak_rss_mb": round(sampler.peak_mb, 3),
        "scope": "cumulative process RSS on this host; rerun on the selected Android device",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--direction", choices=("vi2en", "en2vi"), required=True)
    parser.add_argument("--profile", choices=("development", "edge"), default="edge")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "passed": False}
    try:
        report = profile(args.config, args.direction, args.profile)
        report["passed"] = True
    except Exception as exc:
        report.update({"direction": args.direction, "profile": args.profile, "error": f"{type(exc).__name__}: {exc}"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
