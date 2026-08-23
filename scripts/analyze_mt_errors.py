"""Summarize concrete MT validation failures from completed benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def analyze_report(path: str | Path, top: int = 20) -> dict:
    """Return deterministic error counts and review examples for one benchmark CSV."""
    csv_path = Path(path)
    errors: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    total = 0
    invalid = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            valid = _as_bool(row.get("critical_fields_valid", "False"))
            raw_errors = [item for item in row.get("validation_errors", "").split("|") if item]
            errors.update(raw_errors)
            if valid:
                continue
            invalid += 1
            if len(examples) < top:
                examples.append(
                    {
                        "source": row.get("source", ""),
                        "reference": row.get("reference", ""),
                        "prediction": row.get("prediction", ""),
                        "route": row.get("route", ""),
                        "validation_errors": "|".join(raw_errors),
                    }
                )
    if not total:
        raise ValueError(f"No prediction rows found in {csv_path}")
    return {
        "predictions": str(csv_path),
        "samples": total,
        "critical_invalid": invalid,
        "critical_field_preservation": (total - invalid) / total,
        "validation_error_counts": dict(errors.most_common()),
        "review_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")

    report = {"reports": [analyze_report(path, args.top) for path in args.inputs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
