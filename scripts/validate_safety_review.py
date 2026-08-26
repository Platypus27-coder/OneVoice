"""Validate reviewer decisions before safety audio generation."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row
        for row in rows
        if str(row.get("fixed_translation_candidate", "")).casefold() == "true"
    ]
    statuses = Counter(str(row.get("review_status", "")).strip() for row in candidates)
    errors = []
    for row in candidates:
        safety_id = row.get("safety_id", "<missing>")
        if row.get("review_status", "").strip() != "approved":
            errors.append(f"{safety_id}: review_status is not approved")
        elif not row.get("reviewer", "").strip() or not row.get("reviewed_at", "").strip():
            errors.append(f"{safety_id}: approved row is missing reviewer or reviewed_at")
    print({"candidates": len(candidates), "statuses": dict(statuses), "errors": len(errors)})
    if errors:
        preview = "\n- ".join(errors[:12])
        raise RuntimeError(f"Safety review is incomplete ({len(errors)} errors):\n- {preview}")
    print("Safety review is complete; audio generation may proceed.")


if __name__ == "__main__":
    main()
