"""Re-score existing MT predictions after deterministic validator changes.

This never loads an MT model and never overwrites the original report, so a
Colab user can update validation metrics without repeating an expensive run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from context.engine import ConstructionContextEngine
from evaluation.metrics import corpus_error_rate, normalize_metric_text
from evaluation.reporting import create_run_manifest


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[round((len(values) - 1) * fraction)]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: str) -> float:
    return float(value) if value.strip() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score cached MT prediction CSVs after validator-only changes."
    )
    parser.add_argument("--input-report-dir", required=True, type=Path)
    parser.add_argument("--output-report-dir", required=True, type=Path)
    parser.add_argument("--direction", required=True, choices=["vi2en", "en2vi"])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    source_dir = args.input_report_dir
    predictions_path = source_dir / "predictions.csv"
    aggregate_path = source_dir / "aggregate.json"
    if not predictions_path.is_file() or not aggregate_path.is_file():
        parser.error("--input-report-dir must contain predictions.csv and aggregate.json")
    if args.output_report_dir.exists() and any(args.output_report_dir.iterdir()):
        parser.error("--output-report-dir already contains files; choose a new directory")

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    engine = ConstructionContextEngine.from_data_dir(config["pipeline"]["construction_data_dir"])
    rows = _read_rows(predictions_path)
    if not rows:
        parser.error("predictions.csv has no rows")

    for row in rows:
        context = engine.analyze(row["source"], args.direction)
        errors = engine.validate_translation(row["prediction"], context, args.direction)
        translated_context = engine.analyze(
            row["prediction"], "en2vi" if args.direction == "vi2en" else "vi2en"
        )
        term_total = len(context.canonical_mentions)
        entity_total = sum(
            len(context.entities.get(key, []))
            for key in ("numbers", "units", "directions", "negations")
        )
        missing_terms = sum(error.startswith("missing_term:") for error in errors)
        entity_errors = sum(
            error.startswith(("missing_number:", "missing_unit:", "missing_direction:", "missing_negation"))
            for error in errors
        )
        row.update(
            {
                "critical_fields_valid": str(not errors),
                "validation_errors": "|".join(errors),
                "term_hits": str(max(0, term_total - missing_terms)),
                "term_total": str(term_total),
                "entity_hits": str(max(0, entity_total - entity_errors)),
                "entity_total": str(entity_total),
                "intent_preserved": str(
                    context.intent is None or translated_context.intent == context.intent
                ),
                "exact_match": str(
                    normalize_metric_text(row["prediction"])
                    == normalize_metric_text(row["reference"])
                ),
            }
        )

    original = json.loads(aggregate_path.read_text(encoding="utf-8"))
    total = len(rows)
    aggregate = {
        **original,
        "samples": total,
        "direction": args.direction,
        "critical_field_preservation": sum(
            row["critical_fields_valid"] == "True" for row in rows
        ) / total,
        "terminology_accuracy": (
            sum(int(row["term_hits"]) for row in rows)
            / sum(int(row["term_total"]) for row in rows)
            if sum(int(row["term_total"]) for row in rows)
            else None
        ),
        "entity_preservation": (
            sum(int(row["entity_hits"]) for row in rows)
            / sum(int(row["entity_total"]) for row in rows)
            if sum(int(row["entity_total"]) for row in rows)
            else None
        ),
        "intent_preservation": sum(row["intent_preserved"] == "True" for row in rows) / total,
        "exact_match": sum(row["exact_match"] == "True" for row in rows) / total,
        "reference_wer": corpus_error_rate(
            [row["reference"] for row in rows], [row["prediction"] for row in rows]
        ),
        "latency_p50_ms": _percentile([_as_float(row["latency_ms"]) for row in rows], 0.50),
        "latency_p95_ms": _percentile([_as_float(row["latency_ms"]) for row in rows], 0.95),
    }

    output = args.output_report_dir
    output.mkdir(parents=True)
    fieldnames = list(rows[0])
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    original_manifest = source_dir / "run_manifest.json"
    create_run_manifest(
        output / "run_manifest.json",
        command="revalidate_mt_predictions",
        inputs=[predictions_path, aggregate_path, Path(args.config)],
        metadata={
            "input_report_dir": str(source_dir),
            "original_manifest": str(original_manifest) if original_manifest.is_file() else None,
            "validator_only": True,
        },
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
