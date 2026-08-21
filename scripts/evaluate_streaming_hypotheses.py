"""Replay logged rolling hypotheses through the deterministic commit controller."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context.engine import ConstructionContextEngine
from contracts import ASRHypothesis, CommitKind
from streaming.semantic_commit import SemanticCommitController, StablePrefixAligner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypotheses", required=True, type=Path)
    parser.add_argument("--data-dir", default="data/onevoice_construction_v2")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for line in args.hypotheses.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups[str(row["utterance_id"])].append(row)
    context_engine = ConstructionContextEngine.from_data_dir(args.data_dir)
    unsafe_commits = []
    duplicate_commits = []
    traces = []
    for utterance_id, rows in groups.items():
        rows.sort(key=lambda row: int(row.get("update_index", 0)))
        direction = rows[0].get("direction", "vi2en")
        final_text = str(rows[-1]["text"])
        final_safety = bool(context_engine.analyze(final_text, direction).safety_candidates)
        aligner = StablePrefixAligner()
        controller = SemanticCommitController(safety_confirmations=2)
        emitted_words = []
        for index, row in enumerate(rows):
            text = str(row["text"])
            endpoint = bool(row.get("endpoint", index == len(rows) - 1))
            stable, unstable = aligner.update(text)
            if endpoint:
                stable, unstable = text, ""
            hypothesis = ASRHypothesis(
                text=text,
                stable_prefix=stable,
                unstable_tail=unstable,
                direction=direction,
                started_at=0.0,
                updated_at=float(index),
                endpoint=endpoint,
            )
            decision = controller.decide(
                hypothesis, context_engine.analyze(text, direction)
            )
            trace = {
                "utterance_id": utterance_id,
                "update_index": index,
                "hypothesis": text,
                "stable_prefix": stable,
                "decision": decision.kind.value,
                "committed_text": decision.text,
            }
            traces.append(trace)
            if final_safety and decision.kind == CommitKind.NORMAL:
                unsafe_commits.append(trace)
            if decision.kind != CommitKind.WAIT:
                words = decision.text.casefold().split()
                if words and len(words) <= len(emitted_words) and emitted_words[-len(words):] == words:
                    duplicate_commits.append(trace)
                emitted_words.extend(words)
    report = {
        "utterances": len(groups),
        "updates": len(traces),
        "unsafe_commits": unsafe_commits,
        "duplicate_commits": duplicate_commits,
        "passed": not unsafe_commits and not duplicate_commits,
        "trace": traces,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "trace"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
