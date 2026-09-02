"""Build a GitHub-safe overview SVG from a checked release summary.

The chart deliberately keeps model-level quality and latency separate from the
full streaming pipeline profile.  A model benchmark measures one stage; P5
measures audio -> ASR -> MT -> TTS and therefore must not be compared as if it
were the same latency.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _label(row: dict) -> str:
    direction = str(row.get("direction", "?")).upper().replace("2", "→", 1)
    if row.get("family") == "ASR":
        suite = str(row.get("audio", "?"))
        return f"{direction} ASR / {suite}"
    suite = str(row.get("suite", "?"))
    return f"{direction} MT / {suite}"


def _rows(summary: dict) -> list[dict]:
    rows: list[dict] = []
    for source in summary.get("artifacts", []):
        if not isinstance(source, dict) or source.get("family") not in {"ASR", "MT"}:
            continue
        critical = source.get("critical_term_recall", source.get("critical_field_preservation"))
        error = source.get("wer", source.get("reference_wer"))
        latency = source.get("latency_p95_ms")
        if not all(_number(value) is not None for value in (critical, error, latency)):
            continue
        rows.append(
            {
                "label": _label(source),
                "critical": float(critical),
                "error": float(error),
                "latency": float(latency),
                "samples": source.get("samples", "—"),
            }
        )
    return rows


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _panel(
    rows: list[dict],
    *,
    x: int,
    y: int,
    width: int,
    title: str,
    key: str,
    maximum: float,
    suffix: str,
    lower_is_better: bool = False,
    threshold: float | None = None,
) -> str:
    chart_x = x + 196
    chart_width = width - 232
    row_height = 28
    lines = [
        f"<text x='{x}' y='{y}' class='panel-title'>{_esc(title)}</text>",
        f"<rect x='{x}' y='{y + 12}' width='{width}' height='{row_height * len(rows) + 32}' rx='12' class='panel-bg'/>",
    ]
    if threshold is not None:
        threshold_x = chart_x + chart_width * threshold / maximum
        lines.append(
            f"<path d='M{threshold_x:.1f} {y + 28} V{y + 22 + row_height * len(rows)}' class='threshold'/>",
        )
        lines.append(
            f"<text x='{threshold_x + 5:.1f}' y='{y + 28}' class='threshold-label'>95% gate</text>",
        )
    for index, row in enumerate(rows):
        row_y = y + 30 + index * row_height
        value = float(row[key])
        bar_width = max(2.0, chart_width * min(value, maximum) / maximum)
        if lower_is_better:
            color = "#f08a82"
        else:
            color = "#59c3a9" if value >= 0.95 else "#e9c46a"
        lines.extend(
            [
                f"<text x='{x + 12}' y='{row_y + 13}' class='label'>{_esc(row['label'])}</text>",
                f"<rect x='{chart_x}' y='{row_y}' width='{bar_width:.1f}' height='17' rx='5' fill='{color}' class='bar'/>",
                f"<text x='{chart_x + bar_width + 8:.1f}' y='{row_y + 13}' class='value'>{value * 100:.1f}{suffix}</text>"
                if suffix == "%"
                else f"<text x='{chart_x + bar_width + 8:.1f}' y='{row_y + 13}' class='value'>{value:.0f}{suffix}</text>",
            ]
        )
    return "\n".join(lines)


def build_svg(summary: dict) -> str:
    rows = _rows(summary)
    if not rows:
        raise ValueError("Release summary contains no numeric ASR/MT artifacts")
    generated = summary.get("generated_at", "unknown")
    width = 1440
    panel_width = 1380
    panel_height = 30 + 28 * len(rows) + 52
    panel_gap = 38
    height = 128 + panel_height * 3 + panel_gap * 2
    return f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' role='img' aria-labelledby='title desc'>
  <title id='title'>OneVoice current release benchmark overview</title>
  <desc id='desc'>Critical preservation, error rate, and model-level p95 latency for the current ASR and MT release artifacts.</desc>
  <defs>
    <filter id='rough' x='-2%' y='-2%' width='104%' height='104%'>
      <feTurbulence type='fractalNoise' baseFrequency='0.015' numOctaves='1' seed='23' result='noise'/>
      <feDisplacementMap in='SourceGraphic' in2='noise' scale='1.1'/>
    </filter>
  </defs>
  <rect width='100%' height='100%' fill='#0d1117'/>
  <g font-family="Georgia, 'Times New Roman', serif">
    <text x='{width // 2}' y='42' text-anchor='middle' class='title'>OneVoice · Current release benchmark</text>
    <text x='{width // 2}' y='69' text-anchor='middle' class='subtitle'>{len(rows)} checked ASR/MT artifacts · synthetic/hosted evidence</text>
    <text x='{width // 2}' y='94' text-anchor='middle' class='note'>Higher is better for preservation · lower is better for error · p95 is model-stage latency</text>
    <g filter='url(#rough)'>
      {_panel(rows, x=30, y=122, width=panel_width, title='Critical preservation', key='critical', maximum=1.0, suffix='%', threshold=0.95)}
      {_panel(rows, x=30, y=122 + panel_height + panel_gap, width=panel_width, title='Error rate (WER/reference error)', key='error', maximum=1.0, suffix='%', lower_is_better=True)}
      {_panel(rows, x=30, y=122 + (panel_height + panel_gap) * 2, width=panel_width, title='p95 model-stage latency', key='latency', maximum=max(row['latency'] for row in rows), suffix=' ms', lower_is_better=True)}
    </g>
    <text x='{width - 30}' y='{height - 18}' text-anchor='end' class='footer'>Source summary generated: {_esc(generated)} · P5 full-pipeline latency is reported separately</text>
  </g>
  <style>
    .title {{ fill:#f0f6fc; font-size:30px; font-weight:700; }}
    .subtitle {{ fill:#8b949e; font-size:15px; }}
    .note {{ fill:#8b949e; font-size:14px; }}
    .panel-title {{ fill:#f0f6fc; font-size:19px; font-weight:700; }}
    .panel-bg {{ fill:#111820; stroke:#303b46; stroke-width:1.3; }}
    .label {{ fill:#d8dee4; font-size:13px; }}
    .value {{ fill:#f0f6fc; font-size:13px; font-weight:700; }}
    .bar {{ filter:url(#rough); }}
    .threshold {{ stroke:#f0f6fc; stroke-width:1.4; stroke-dasharray:5 5; opacity:.55; }}
    .threshold-label {{ fill:#c9d1d9; font-size:11px; }}
    .footer {{ fill:#8b949e; font-size:12px; }}
  </style>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(summary), encoding="utf-8")
    print(f"Wrote {args.output} from {len(_rows(summary))} release artifacts")


if __name__ == "__main__":
    main()
