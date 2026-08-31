"""Profile the full offline streaming runtime for one direction.

This measures the selected release runtime (not a synthetic component stub),
with network sockets blocked before pipeline import.  It reports model load
time, per-route speech/commit/audio timings, complete-turn latency and peak
RSS.  A latency budget overrun is recorded as a blocker in the JSON report;
the command still exits successfully when measurement itself completed.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import yaml


class RSSSampler:
    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.peak = self._rss_bytes()
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _rss_bytes() -> int:
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except Exception:
            try:
                for line in Path("/proc/self/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
            except Exception:
                pass
        return 0

    def _run(self) -> None:
        while not self.stop.wait(self.interval_s):
            self.peak = max(self.peak, self._rss_bytes())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=1)
        self.peak = max(self.peak, self._rss_bytes())

    @property
    def peak_mb(self) -> float:
        return self.peak / (1024 * 1024)


class NoNetwork:
    def __enter__(self):
        self.original_socket = socket.socket
        self.original_create = socket.create_connection

        def blocked(*_args, **_kwargs):
            raise RuntimeError("Production runtime attempted network access")

        original_socket = self.original_socket

        class BlockedSocket(original_socket):
            def connect(self, *args, **kwargs):
                return blocked(*args, **kwargs)

        socket.socket = BlockedSocket
        socket.create_connection = blocked
        os.environ.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "MODELSCOPE_OFFLINE": "1",
                "ONEVOICE_OFFLINE": "1",
            }
        )
        return self

    def __exit__(self, *_args):
        socket.socket = self.original_socket
        socket.create_connection = self.original_create


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def summarize(rows: list[dict]) -> dict:
    return {
        "samples": len(rows),
        "commit_to_first_audio_p50_ms": percentile([row["commit_to_first_audio_ms"] for row in rows], 0.50),
        "commit_to_first_audio_p95_ms": percentile([row["commit_to_first_audio_ms"] for row in rows], 0.95),
        "speech_to_commit_p50_ms": percentile([row["speech_to_commit_ms"] for row in rows if row.get("speech_to_commit_ms") is not None], 0.50),
        "speech_to_commit_p95_ms": percentile([row["speech_to_commit_ms"] for row in rows if row.get("speech_to_commit_ms") is not None], 0.95),
        "complete_turn_p50_ms": percentile([row["complete_turn_ms"] for row in rows], 0.50),
        "complete_turn_p95_ms": percentile([row["complete_turn_ms"] for row in rows], 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--direction", choices=("vi2en", "en2vi"), required=True)
    parser.add_argument("--normal-input", required=True, type=Path)
    parser.add_argument("--safety-input", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("development", "edge"), default="edge")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-rss-mb", type=float, default=200.0)
    parser.add_argument("--normal-target-ms", type=float, default=1000.0)
    parser.add_argument("--safety-target-ms", type=float, default=300.0)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if not args.config.is_file() or not args.normal_input.is_file() or not args.safety_input.is_file():
        raise FileNotFoundError("config, normal-input and safety-input must exist")

    report = {
        "schema_version": 1,
        "direction": args.direction,
        "profile": args.profile,
        "offline": True,
        "network_blocked": True,
        "normal_input": str(args.normal_input.resolve()),
        "safety_input": str(args.safety_input.resolve()),
        "repeats": args.repeats,
        "passed": False,
        "measurement_completed": False,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not config.get("pipeline", {}).get("offline"):
            raise ValueError("P5 requires pipeline.offline=true")
        with NoNetwork(), RSSSampler() as rss:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from pipeline import OneVoicePipeline

            pipeline = OneVoicePipeline(
                config_path=str(args.config), direction=args.direction,
                profile=args.profile, offline=True,
                report_dir=str(args.report_dir / "load"),
            )
            load_started = time.perf_counter()
            pipeline.load_models()
            load_ms = (time.perf_counter() - load_started) * 1000
            route_rows = {"normal": [], "safety": []}
            for route, input_path in (("normal", args.normal_input), ("safety", args.safety_input)):
                for repeat in range(1, args.repeats + 1):
                    case_dir = args.report_dir / route / str(repeat)
                    pipeline.report_dir = case_dir
                    started = time.perf_counter()
                    result = pipeline.stream_file(str(input_path), realtime=False)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    if result.get("fatal_error") or result.get("dropped_audio_frames"):
                        raise RuntimeError(f"stream {route}/{repeat} failed: {result}")
                    latency_path = case_dir / "latency_summary.json"
                    latency = json.loads(latency_path.read_text(encoding="utf-8")) if latency_path.is_file() else {}
                    route_summary = latency.get(route, {})
                    if not route_summary.get("samples"):
                        raise RuntimeError(f"expected {route} commit was not observed in {case_dir}")
                    route_rows[route].append({
                        "repeat": repeat,
                        "commit_to_first_audio_ms": float(route_summary["commit_to_first_audio_p50_ms"]),
                        "speech_to_commit_ms": route_summary.get("speech_to_commit_p50_ms"),
                        "complete_turn_ms": float(result["complete_turn_ms"]),
                        "wall_elapsed_ms": elapsed_ms,
                        "commits": result.get("commits", 0),
                    })
        report.update({
            "load_time_ms": round(load_ms, 3),
            "peak_rss_mb": round(rss.peak_mb, 3),
            "routes": {route: summarize(rows) for route, rows in route_rows.items()},
            "latency_targets_ms": {"normal": args.normal_target_ms, "safety": args.safety_target_ms},
            "measurement_completed": True,
        })
        normal_p95 = report["routes"]["normal"]["commit_to_first_audio_p95_ms"]
        safety_p95 = report["routes"]["safety"]["commit_to_first_audio_p95_ms"]
        blockers = []
        if report["peak_rss_mb"] > args.max_rss_mb:
            blockers.append(f"peak RSS {report['peak_rss_mb']:.1f} MB > {args.max_rss_mb:.1f} MB")
        if normal_p95 is not None and normal_p95 >= args.normal_target_ms:
            blockers.append(f"normal commit→audio p95 {normal_p95:.1f} ms >= {args.normal_target_ms:.1f} ms")
        if safety_p95 is not None and safety_p95 >= args.safety_target_ms:
            blockers.append(f"safety commit→audio p95 {safety_p95:.1f} ms >= {args.safety_target_ms:.1f} ms")
        report["blockers"] = blockers
        report["passed"] = not blockers
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    temporary = args.report_dir / "profile.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.report_dir / "profile.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if "error" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
