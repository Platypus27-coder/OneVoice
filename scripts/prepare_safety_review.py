"""Create a human-reviewable copy of the safety fast-path CSV.

This intentionally never changes ``review_status``.  A reviewer must edit the
Drive copy and supply an explicit approval decision before safety audio can be
generated.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REVIEW_COLUMNS = ("reviewer", "reviewed_at", "review_notes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    required = {"safety_id", "vi", "en", "review_status", "fixed_translation_candidate"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"Safety CSV is missing required columns: {missing}")
    for field in REVIEW_COLUMNS:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in REVIEW_COLUMNS:
            row.setdefault(field, "")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} review rows to {args.output}")
    print("Set review_status to approved or rejected; approved rows require reviewer and reviewed_at.")


if __name__ == "__main__":
    main()
