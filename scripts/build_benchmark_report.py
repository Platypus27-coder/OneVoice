"""Build a portable OneVoice benchmark report from aggregate artifacts.

The output is dependency-free HTML (inline SVG charts), Markdown, and JSON.
It intentionally reports synthetic and hosted/CPU measurements as evidence,
not as a real-site or field-device claim.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path


def load_aggregates(report_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(report_root.rglob("aggregate.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not payload.get("samples"):
            continue
        relative = path.parent.relative_to(report_root).as_posix()
        family = classify(relative, payload)
        rows.append({"report": relative, "family": family, "path": str(path), **payload})
    return rows


def classify(relative: str, payload: dict) -> str:
    lower = relative.casefold()
    if "/mt/" in f"/{lower}" or "benchmark_mt" in str(payload.get("command", "")):
        return "MT"
    if "gipformer" in lower or "asr" in lower or "wer" in payload:
        return "ASR"
    return "Other"


def pct(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value) * 100:.2f}%"


def number(value: object, digits: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.{digits}f}"


def gate(row: dict) -> str:
    critical = row.get("critical_term_recall")
    if critical is None:
        critical = row.get("critical_field_preservation")
    if isinstance(critical, (int, float)):
        return "PASS" if critical >= 0.95 else "BELOW 95%"
    return "EVIDENCE"


def esc(value: object) -> str:
    return html.escape(str(value))


def svg_chart(rows: list[dict], metric: str, title: str, lower_is_better: bool = False) -> str:
    values = [(row, row.get(metric)) for row in rows if isinstance(row.get(metric), (int, float))]
    if not values:
        return f"<p class='muted'>No {esc(metric)} artifacts found.</p>"
    values = values[:24]
    width, row_height, left = 920, 28, 300
    height = 48 + row_height * len(values)
    maximum = max(float(value) for _, value in values) or 1.0
    if metric.endswith("_ms"):
        scale_max = max(maximum, 1.0)
        labels = lambda value: f"{value:.0f} ms"
    else:
        scale_max = 1.0
        labels = lambda value: f"{value * 100:.1f}%"
    bars = []
    for index, (row, value) in enumerate(values):
        y = 34 + index * row_height
        bar_width = max(2, 570 * float(value) / scale_max)
        color = "#e56b6f" if lower_is_better else ("#2a9d8f" if float(value) >= 0.95 else "#e9c46a")
        label = row["report"]
        if len(label) > 43:
            label = "…" + label[-42:]
        bars.append(
            f"<text x='8' y='{y + 14}' class='chart-label'>{esc(label)}</text>"
            f"<rect x='{left}' y='{y}' width='{bar_width:.1f}' height='18' rx='4' fill='{color}'/>"
            f"<text x='{left + bar_width + 8:.1f}' y='{y + 14}' class='chart-value'>{labels(float(value))}</text>"
        )
    return f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{esc(title)}'>{''.join(bars)}</svg>"


def markdown_report(rows: list[dict], generated: str) -> str:
    lines = [
        "# OneVoice benchmark report",
        "",
        f"Generated: `{generated}`",
        "",
        "> Evidence is from checked benchmark artifacts. Synthetic audio and hosted/CPU measurements are not field-device or real-site validation.",
        "",
        "## Summary",
        "",
        f"- Aggregate artifacts: **{len(rows)}**",
        f"- MT artifacts: **{sum(row['family'] == 'MT' for row in rows)}**",
        f"- ASR artifacts: **{sum(row['family'] == 'ASR' for row in rows)}**",
        "",
        "## Artifacts",
        "",
        "| Family | Report | Direction/suite | Samples | WER/error | Critical | p95 ms | Gate |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        error = row.get("wer", row.get("reference_wer"))
        critical = row.get("critical_term_recall", row.get("critical_field_preservation"))
        direction = f"{row.get('direction', '—')} / {row.get('suite', row.get('audio', '—'))}"
        lines.append(
            f"| {row['family']} | `{row['report']}` | {direction} | {row.get('samples', '—')} | {pct(error)} | {pct(critical)} | {number(row.get('latency_p95_ms'))} | {gate(row)} |"
        )
    return "\n".join(lines) + "\n"


def html_report(rows: list[dict], generated: str) -> str:
    asr = [row for row in rows if row["family"] == "ASR"]
    mt = [row for row in rows if row["family"] == "MT"]
    cards = f"""
      <div class='cards'>
        <div class='card'><span>Artifacts</span><strong>{len(rows)}</strong></div>
        <div class='card'><span>MT runs</span><strong>{len(mt)}</strong></div>
        <div class='card'><span>ASR runs</span><strong>{len(asr)}</strong></div>
        <div class='card'><span>Real-site data</span><strong>Not yet</strong></div>
      </div>"""
    table_rows = []
    for row in rows:
        error = row.get("wer", row.get("reference_wer"))
        critical = row.get("critical_term_recall", row.get("critical_field_preservation"))
        direction = f"{row.get('direction', '—')} / {row.get('suite', row.get('audio', '—'))}"
        gate_status = gate(row)
        gate_class = "pass" if gate_status == "PASS" else ("warning" if gate_status == "BELOW 95%" else "evidence")
        table_rows.append(
            "<tr>"
            f"<td>{esc(row['family'])}</td><td><code>{esc(row['report'])}</code></td>"
            f"<td>{esc(direction)}</td><td>{esc(row.get('samples', '—'))}</td>"
            f"<td>{pct(error)}</td><td>{pct(critical)}</td><td>{number(row.get('latency_p95_ms'))}</td>"
            f"<td><span class='badge {gate_class}'>{esc(gate_status)}</span></td></tr>"
        )
    styles = """
      :root { color-scheme: light; --ink:#17212b; --muted:#607080; --line:#dbe4ea; --teal:#2a9d8f; --navy:#16324f; }
      body { margin:0; font:15px/1.5 Inter,system-ui,-apple-system,Segoe UI,sans-serif; color:var(--ink); background:#f5f8fa; }
      main { max-width:1240px; margin:0 auto; padding:42px 28px 80px; }
      h1 { color:var(--navy); margin:0 0 4px; font-size:34px; } h2 { color:var(--navy); margin-top:38px; }
      .subtitle,.muted { color:var(--muted); } .notice { padding:14px 18px; border-left:4px solid #e9c46a; background:#fff8e7; border-radius:6px; }
      .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:28px 0; }
      .card { background:white; border:1px solid var(--line); border-radius:12px; padding:17px 20px; box-shadow:0 3px 12px #16324f0d; }
      .card span { display:block; color:var(--muted); font-size:13px; } .card strong { display:block; color:var(--navy); font-size:24px; margin-top:4px; }
      .chart { background:white; border:1px solid var(--line); border-radius:12px; padding:14px; overflow:auto; }
      svg { width:100%; min-width:720px; height:auto; } .chart-label { fill:#425466; font-size:12px; } .chart-value { fill:#17212b; font-size:12px; font-weight:600; }
      table { border-collapse:collapse; width:100%; background:white; border:1px solid var(--line); font-size:13px; } th,td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; } th { background:#edf3f6; color:var(--navy); position:sticky; top:0; } code { white-space:normal; word-break:break-word; }
      .badge { border-radius:99px; padding:3px 8px; font-size:11px; font-weight:700; white-space:nowrap; } .pass { background:#d8f3dc; color:#176b3a; } .warning { background:#ffe4c7; color:#8a4b08; } .evidence { background:#e4edf3; color:#3c596c; }
      @media(max-width:700px){ main{padding:26px 14px} h1{font-size:27px} }
    """
    return f"""<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>OneVoice benchmark report</title><style>{styles}</style></head><body><main>
      <h1>OneVoice benchmark report</h1><div class='subtitle'>Generated {esc(generated)}</div>
      <p class='notice'>Các số liệu dưới đây lấy từ benchmark artifacts đã kiểm tra. Audio synthetic và hosted/CPU không phải validation công trường thực tế; chưa có real-site holdout.</p>
      {cards}
      <h2>ASR — error rate</h2><div class='chart'>{svg_chart(asr, 'wer', 'ASR WER', lower_is_better=True)}</div>
      <h2>ASR/MT — critical preservation</h2><div class='chart'>{svg_chart(asr + mt, 'critical_term_recall', 'Critical recall')}</div>
      <h2>MT — error rate</h2><div class='chart'>{svg_chart(mt, 'reference_wer', 'MT error rate', lower_is_better=True)}</div>
      <h2>All aggregate artifacts</h2><div class='chart'><table><thead><tr><th>Family</th><th>Report</th><th>Direction/suite</th><th>Samples</th><th>WER/error</th><th>Critical</th><th>p95 ms</th><th>Gate</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
      <p class='subtitle'>OneVoice V2 evidence report. Model promotion vẫn cần quality gate, offline preflight, edge profile và real-site holdout.</p>
    </main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_aggregates(args.report_root)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"generated_at": generated, "report_root": str(args.report_root.resolve()), "artifacts": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(markdown_report(rows, generated), encoding="utf-8")
    (args.output_dir / "report.html").write_text(html_report(rows, generated), encoding="utf-8")
    print(json.dumps({"artifacts": len(rows), "html": str(args.output_dir / "report.html"), "markdown": str(args.output_dir / "report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
