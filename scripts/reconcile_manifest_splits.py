"""Repair exact-text split leakage in a generated audio manifest without touching WAVs."""

from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path


SPLIT_PRIORITY = {"train": 0, "dev": 1, "test": 2}


def normalized_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).casefold().split())


def reconcile_rows(rows: list[dict], language: str) -> tuple[list[dict], dict]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("language") == language:
            groups[normalized_text(row.get("text", ""))].append(index)
    changed_rows = 0
    changed_groups = 0
    for indices in groups.values():
        splits = {str(rows[index].get("split", "")) for index in indices}
        if len(splits) < 2:
            continue
        target = max(splits, key=lambda split: SPLIT_PRIORITY.get(split, -1))
        changed_groups += 1
        for index in indices:
            row = rows[index]
            if row.get("split") != target:
                row.setdefault("source_split", row.get("split"))
                row["split"] = target
                row["split_resolution"] = "exact_text_group_test_dev_train_priority"
                changed_rows += 1
    return rows, {
        "language": language,
        "groups_checked": len(groups),
        "groups_reconciled": changed_groups,
        "rows_reassigned": changed_rows,
        "policy": "test > dev > train",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--language", choices=["vi", "en"], required=True)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows, report = reconcile_rows(rows, args.language)
    if args.in_place and report["rows_reassigned"]:
        backup = args.manifest.with_name(args.manifest.name + ".before_split_reconcile")
        if not backup.exists():
            shutil.copy2(args.manifest, backup)
        temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        temporary.replace(args.manifest)
        report["backup"] = str(backup)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
