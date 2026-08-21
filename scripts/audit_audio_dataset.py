"""Audit a OneVoice clean/noisy manifest, including physical WAV validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from evaluation.dataset_audit import audit_audio_manifest
from evaluation.reporting import create_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--logical-only", action="store_true")
    parser.add_argument("--expected-noisy", type=int, default=16128)
    parser.add_argument("--expected-clean", type=int, default=8064)
    parser.add_argument("--language", choices=["vi", "en"])
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--require-realized-snr", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--report-dir", default="reports/data_audit")
    args = parser.parse_args()

    report = audit_audio_manifest(
        args.manifest,
        physical=not args.logical_only,
        expected_noisy=args.expected_noisy,
        expected_clean=args.expected_clean,
        language=args.language,
        min_speakers=args.min_speakers,
        require_realized_snr=args.require_realized_snr,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    output = Path(args.report_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    create_run_manifest(
        output / "run_manifest.json",
        command="audit_audio_dataset",
        inputs=[args.manifest],
        metadata={"physical": not args.logical_only},
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
