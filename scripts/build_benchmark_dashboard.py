"""Build a compact Markdown index from measured OneVoice benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON report: {path}") from exc


def _metric(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_dashboard(report_root: str | Path, output: str | Path) -> list[dict]:
    root = Path(report_root)
    rows: list[dict] = []
    for aggregate_path in sorted(root.rglob("aggregate.json")):
        aggregate = _read_json(aggregate_path)
        manifest_path = aggregate_path.with_name("run_manifest.json")
        manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
        model_reference = manifest.get("metadata", {}).get("model_reference", {})
        rows.append(
            {
                "report": aggregate_path.parent.relative_to(root).as_posix(),
                "command": manifest.get("command", "unknown"),
                "direction": aggregate.get("direction", "—"),
                "suite": aggregate.get("suite", aggregate.get("audio", "—")),
                "route": "context" if aggregate.get("with_context") else "raw",
                "model": model_reference.get("source", "—"),
                "error": aggregate.get("wer", aggregate.get("reference_wer")),
                "term": aggregate.get(
                    "construction_term_recall", aggregate.get("terminology_accuracy")
                ),
                "critical": aggregate.get("critical_term_recall", aggregate.get("critical_field_preservation")),
                "p95_ms": aggregate.get("latency_p95_ms"),
                "samples": aggregate.get("samples", "—"),
            }
        )
    lines = [
        "# OneVoice benchmark dashboard",
        "",
        "Generated from checked benchmark artifacts. A missing value is shown as `—`; it is not a pass.",
        "",
        "| Report | Command | Direction | Suite/audio | Route | Model | Error | Term | Critical | p95 ms | Samples |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {report} | {command} | {direction} | {suite} | {route} | {model} | {error} | {term} | {critical} | {p95_ms} | {samples} |".format(
                **{key: _metric(value) for key, value in row.items()}
            )
        )
    if not rows:
        lines.append("| *(no aggregate.json found)* | — | — | — | — | — | — | — | — | — | — |")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = build_dashboard(args.report_root, args.output)
    print(f"Wrote {args.output} with {len(rows)} benchmark rows")


if __name__ == "__main__":
    main()
