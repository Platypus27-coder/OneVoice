"""Measure full edge model-load RSS while proving startup makes no network calls."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import OneVoicePipeline


class RSSSampler:
    def __init__(self, interval_s: float = 0.01):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("Install psutil for native RSS measurement") from exc
        self.process = psutil.Process()
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.peak = self.process.memory_info().rss
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(self.interval_s):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join(timeout=1)
        self.peak = max(self.peak, self.process.memory_info().rss)


class NoNetwork:
    def __enter__(self):
        self.original_socket = socket.socket
        self.original_create = socket.create_connection

        def blocked(*args, **kwargs):
            raise RuntimeError("Production runtime attempted network access")

        original_socket = self.original_socket

        class BlockedSocket(original_socket):
            def connect(self, *args, **kwargs):
                return blocked(*args, **kwargs)

        socket.socket = BlockedSocket
        socket.create_connection = blocked
        return self

    def __exit__(self, exc_type, exc, traceback):
        socket.socket = self.original_socket
        socket.create_connection = self.original_create


def main() -> None:
    parser = argparse.ArgumentParser(description="OneVoice edge RSS/offline gate")
    parser.add_argument("--direction", choices=["vi2en", "en2vi"], required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--site-pack")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-rss-mb", type=float, default=200.0)
    args = parser.parse_args()

    started = time.perf_counter()
    error = None
    rss = RSSSampler()
    try:
        with NoNetwork(), rss:
            pipeline = OneVoicePipeline(
                config_path=args.config,
                direction=args.direction,
                profile="edge",
                site_pack_path=args.site_pack,
                offline=True,
            )
            pipeline.load_models()
    except Exception as exc:
        error = repr(exc)
    peak_mb = rss.peak / (1024 * 1024)
    network_access = bool(error and "attempted network access" in error)
    report = {
        "profile": "edge",
        "direction": args.direction,
        "network_access_observed": network_access,
        "load_time_ms": (time.perf_counter() - started) * 1000,
        "peak_rss_mb": peak_mb,
        "max_rss_mb": args.max_rss_mb,
        "startup_error": error,
        "passed": error is None and peak_mb < args.max_rss_mb,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
