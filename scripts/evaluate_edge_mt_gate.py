"""Gate an edge ONNX MT benchmark against the frozen development benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


LOWER_IS_BETTER = ("reference_wer",)
HIGHER_IS_BETTER = (
    "critical_field_preservation",
    "terminology_accuracy",
    "entity_preservation",
    "intent_preservation",
    "exact_match",
)
SUITES = ("test", "minimal", "safety")


def aggregate_path(root: Path, suite: str) -> Path:
    candidates = (root / suite / "raw" / "aggregate.json", root / suite / "aggregate.json")
    return next((path for path in candidates if path.is_file()), candidates[0])


def load_aggregate(root: Path, suite: str, direction: str) -> dict:
    path = aggregate_path(root, suite)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {suite} aggregate: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("direction") != direction or result.get("suite") != suite:
        raise ValueError(f"Unexpected aggregate identity in {path}: {result.get('direction')}/{result.get('suite')}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--direction", choices=["vi2en", "en2vi"], required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-regression-pp", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_regression_pp < 0:
        parser.error("--max-regression-pp must be non-negative")

    comparisons: dict[str, dict] = {}
    failures: list[str] = []
    for suite in SUITES:
        baseline = load_aggregate(args.baseline_root, suite, args.direction)
        candidate = load_aggregate(args.candidate_root, suite, args.direction)
        if baseline.get("samples") != candidate.get("samples"):
            failures.append(f"{suite}: sample count differs ({baseline.get('samples')} != {candidate.get('samples')})")
        metrics: dict[str, dict] = {}
        for metric in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER):
            before, after = baseline.get(metric), candidate.get(metric)
            if before is None or after is None:
                metrics[metric] = {"baseline": before, "candidate": after, "skipped": True}
                continue
            # Positive is always a regression, expressed in percentage points.
            regression_pp = (after - before) * 100 if metric in LOWER_IS_BETTER else (before - after) * 100
            passed = regression_pp <= args.max_regression_pp
            metrics[metric] = {
                "baseline": before,
                "candidate": after,
                "regression_pp": regression_pp,
                "passed": passed,
            }
            if not passed:
                failures.append(f"{suite}/{metric}: regression {regression_pp:.3f}pp")
        comparisons[suite] = {
            "baseline": str(aggregate_path(args.baseline_root, suite)),
            "candidate": str(aggregate_path(args.candidate_root, suite)),
            "metrics": metrics,
        }
    report = {
        "schema_version": 1,
        "direction": args.direction,
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline_root": str(args.baseline_root.resolve()),
        "candidate_root": str(args.candidate_root.resolve()),
        "max_regression_pp": args.max_regression_pp,
        "comparisons": comparisons,
        "failures": failures,
        "passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
