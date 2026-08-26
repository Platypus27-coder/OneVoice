"""Create a traceable internal-demo approval copy of a safety review CSV.

This is deliberately opt-in and never replaces the original review file.  It
must not be used to represent safety-officer or production approval.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CONFIRMATION = "IMPACT_INTERNAL_DEMO_APPROVED"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at", required=True, help="YYYY-MM-DD")
    parser.add_argument("--confirm", required=True, help=f"Must equal {CONFIRMATION}")
    args = parser.parse_args()
    if args.confirm != CONFIRMATION:
        raise ValueError(f"Refusing approval: --confirm must equal {CONFIRMATION}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing approval copy: {args.output}")

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    for field in ("reviewer", "reviewed_at", "review_notes"):
        if field not in fields:
            raise ValueError(f"Input is not a safety review packet: missing {field}")

    approved = 0
    for row in rows:
        if str(row.get("fixed_translation_candidate", "")).casefold() != "true":
            continue
        status = str(row.get("review_status", "")).strip()
        if status == "rejected":
            raise ValueError(f"Refusing to overwrite rejected phrase: {row.get('safety_id')}")
        row["review_status"] = "approved"
        row["reviewer"] = args.reviewer
        row["reviewed_at"] = args.reviewed_at
        row["review_notes"] = (
            "Impact internal demo approval; not production safety-officer approval."
        )
        approved += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created internal-demo approval copy with {approved} approved fixed safety phrases: {args.output}")


if __name__ == "__main__":
    main()
