"""Apply the locked denoiser acceptance gate to four measured aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-clean", required=True, type=Path)
    parser.add_argument("--baseline-noisy", required=True, type=Path)
    parser.add_argument("--candidate-clean", required=True, type=Path)
    parser.add_argument("--candidate-noisy", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    base_clean, base_noisy = load(args.baseline_clean), load(args.baseline_noisy)
    cand_clean, cand_noisy = load(args.candidate_clean), load(args.candidate_noisy)

    def relative_gain(metric: str) -> float | None:
        baseline_value = base_noisy.get(metric)
        candidate_value = cand_noisy.get(metric)
        if baseline_value is None or candidate_value is None:
            return None
        baseline = float(baseline_value)
        return (baseline - float(candidate_value)) / baseline if baseline else 0.0

    wer_gain = relative_gain("wer")
    cter_gain = relative_gain("construction_term_error_rate")
    critical_base = base_noisy.get("critical_term_recall")
    critical_candidate = cand_noisy.get("critical_term_recall")
    critical_preserved = (
        critical_base is None
        or (
            critical_candidate is not None
            and float(critical_candidate) >= float(critical_base)
        )
    )
    clean_delta = float(cand_clean["wer"]) - float(base_clean["wer"])
    report = {
        "noisy_wer_relative_gain": wer_gain,
        "noisy_cter_relative_gain": cter_gain,
        "critical_term_recall_preserved": critical_preserved,
        "clean_wer_absolute_delta": clean_delta,
        "passed": (
            (wer_gain is not None and wer_gain >= 0.05)
            or (cter_gain is not None and cter_gain >= 0.05)
        )
        and critical_preserved
        and clean_delta <= 0.01,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
