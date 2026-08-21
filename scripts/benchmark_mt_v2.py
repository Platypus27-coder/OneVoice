"""Measured EnViT5 benchmark on fixed V2 test and safety suites."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from context.engine import ConstructionContextEngine
from evaluation.metrics import corpus_error_rate, normalize_metric_text
from evaluation.reporting import create_run_manifest
from translation.mt_engine import Translator


def load_pairs(path: Path, direction: str) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if "a_vi" in row:
                variants = (
                    (row.get("a_vi", ""), row.get("a_en", ""), "a"),
                    (row.get("b_vi", ""), row.get("b_en", ""), "b"),
                )
                for vi, en, variant in variants:
                    vi, en = str(vi).strip(), str(en).strip()
                    if vi and en:
                        rows.append(
                            {
                                "source": vi if direction == "vi2en" else en,
                                "reference": en if direction == "vi2en" else vi,
                                "domain": row.get("domain", "unknown"),
                                "risk_level": "critical",
                                "minimal_pair_id": row.get("minimal_pair_id", ""),
                                "variant": variant,
                            }
                        )
                continue
            vi, en = row.get("vi", "").strip(), row.get("en", "").strip()
            if vi and en:
                rows.append(
                    {
                        "source": vi if direction == "vi2en" else en,
                        "reference": en if direction == "vi2en" else vi,
                        "domain": row.get("domain", "unknown"),
                        "risk_level": row.get("risk_level", row.get("severity", "unknown")),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=["vi2en", "en2vi"], default="vi2en")
    parser.add_argument("--suite", choices=["test", "minimal", "safety"], default="test")
    parser.add_argument("--with-context", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--report-dir", default="reports/mt_v2")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_root = Path(config["pipeline"]["construction_data_dir"])
    suite_files = {
        "test": "test.csv",
        "minimal": "minimal_pairs.csv",
        "safety": "safety_fast_path.csv",
    }
    data_file = data_root / suite_files[args.suite]
    pairs = load_pairs(data_file, args.direction)
    if args.max_samples:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise ValueError(f"No translation pairs found in {data_file}")

    translator = Translator(config)
    translator.load()
    context_engine = ConstructionContextEngine.from_data_dir(data_root)
    predictions = []
    for pair in pairs:
        context = context_engine.analyze(pair["source"], args.direction)
        started = time.perf_counter()
        if args.with_context and context.safety_candidates:
            prediction = context.safety_candidates[0].translated_text
            route = "safety_fast_path"
        elif args.with_context and context.translation_memory:
            prediction = context.translation_memory
            route = "translation_memory"
        else:
            source = (
                context_engine.canonicalize_source(pair["source"], context, args.direction)
                if args.with_context
                else pair["source"]
            )
            prediction = translator.translate(source, args.direction)
            route = "context_mt" if args.with_context else "raw_mt"
        latency_ms = (time.perf_counter() - started) * 1000
        validation_errors = context_engine.validate_translation(
            prediction, context, args.direction
        )
        translated_context = context_engine.analyze(
            prediction, "en2vi" if args.direction == "vi2en" else "vi2en"
        )
        term_total = len(context.canonical_mentions)
        missing_terms = sum(error.startswith("missing_term:") for error in validation_errors)
        entity_errors = sum(
            error.startswith(("missing_number:", "missing_unit:", "missing_direction:", "missing_negation"))
            for error in validation_errors
        )
        entity_total = sum(
            len(context.entities.get(key, []))
            for key in ("numbers", "units", "directions", "negations")
        )
        predictions.append(
            {
                **pair,
                "prediction": prediction,
                "route": route,
                "latency_ms": round(latency_ms, 3),
                "critical_fields_valid": not validation_errors,
                "validation_errors": "|".join(validation_errors),
                "term_hits": max(0, term_total - missing_terms),
                "term_total": term_total,
                "entity_hits": max(0, entity_total - entity_errors),
                "entity_total": entity_total,
                "intent_preserved": context.intent is None
                or translated_context.intent == context.intent,
                "exact_match": normalize_metric_text(prediction)
                == normalize_metric_text(pair["reference"]),
            }
        )
    latencies = sorted(row["latency_ms"] for row in predictions)
    aggregate = {
        "samples": len(predictions),
        "direction": args.direction,
        "suite": args.suite,
        "with_context": args.with_context,
        "reference_wer": corpus_error_rate(
            [row["reference"] for row in predictions],
            [row["prediction"] for row in predictions],
        ),
        "critical_field_preservation": sum(row["critical_fields_valid"] for row in predictions)
        / len(predictions),
        "terminology_accuracy": (
            sum(row["term_hits"] for row in predictions)
            / sum(row["term_total"] for row in predictions)
            if sum(row["term_total"] for row in predictions)
            else None
        ),
        "entity_preservation": (
            sum(row["entity_hits"] for row in predictions)
            / sum(row["entity_total"] for row in predictions)
            if sum(row["entity_total"] for row in predictions)
            else None
        ),
        "intent_preservation": sum(row["intent_preserved"] for row in predictions)
        / len(predictions),
        "exact_match": sum(row["exact_match"] for row in predictions) / len(predictions),
        "latency_p50_ms": latencies[round((len(latencies) - 1) * 0.50)],
        "latency_p95_ms": latencies[round((len(latencies) - 1) * 0.95)],
    }
    output = Path(args.report_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    create_run_manifest(
        output / "run_manifest.json",
        command="benchmark_mt_v2",
        inputs=[data_file, args.config, config["translation"].get("model_dir", "models/envit5")],
        metadata=vars(args),
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
