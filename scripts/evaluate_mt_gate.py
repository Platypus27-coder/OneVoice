"""Compare an adapted MT checkpoint against the context-corrected baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-error-suite", required=True, type=Path)
    parser.add_argument("--candidate-error-suite", required=True, type=Path)
    parser.add_argument("--baseline-general", required=True, type=Path)
    parser.add_argument("--candidate-general", required=True, type=Path)
    parser.add_argument(
        "--metric", default="critical_field_preservation",
        choices=["critical_field_preservation", "exact_match"],
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    baseline_error = load(args.baseline_error_suite)
    candidate_error = load(args.candidate_error_suite)
    baseline_general = load(args.baseline_general)
    candidate_general = load(args.candidate_general)
    improvement = float(candidate_error[args.metric]) - float(baseline_error[args.metric])
    general_delta = float(candidate_general["exact_match"]) - float(
        baseline_general["exact_match"]
    )
    report = {
        "metric": args.metric,
        "error_suite_absolute_improvement": improvement,
        "general_exact_match_delta": general_delta,
        "passed": improvement >= 0.03 and general_delta >= -0.01,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
