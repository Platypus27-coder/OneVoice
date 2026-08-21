from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.real_site import audit_real_site_manifest, write_holdout_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OneVoice real-site pilot")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--holdout-lock")
    parser.add_argument("--create-holdout-lock")
    parser.add_argument("--logical-only", action="store_true")
    parser.add_argument("--final-gate", action="store_true")
    args = parser.parse_args()
    if args.create_holdout_lock:
        write_holdout_lock(args.manifest, args.create_holdout_lock)
    report = audit_real_site_manifest(
        args.manifest,
        physical=not args.logical_only,
        holdout_lock_path=args.holdout_lock or args.create_holdout_lock,
        final_gate=args.final_gate,
    )
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
