"""Build a compact Markdown index from measured OneVoice benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Works both as ``python -m scripts...`` and ``python scripts/...py``.
    from scripts.benchmark_selection import select_release_rows
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from benchmark_selection import select_release_rows


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


def build_dashboard(report_root: str | Path, output: str | Path, profile: str = "all") -> list[dict]:
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
    rows = select_release_rows(rows, profile)
    lines = [
        f"# OneVoice benchmark dashboard ({profile})",
        "",
        ("Current runtime scope only; historical and rejected fine-tune runs are excluded." if profile == "release" else "All discovered artifacts, including historical experiments."),
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
    parser.add_argument("--profile", choices=["all", "release"], default="all")
    args = parser.parse_args()
    rows = build_dashboard(args.report_root, args.output, args.profile)
    print(f"Wrote {args.output} with {len(rows)} benchmark rows")


if __name__ == "__main__":
    main()
