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
    """Keep both exact text and frame-pattern groups in one split.

    A duplicate text can join two frame patterns, so resolving text groups alone
    may create a frame-pattern leak.  Connected components across both keys are
    the minimal safe repair unit.
    """
    text_groups: dict[str, list[int]] = defaultdict(list)
    pattern_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("language") == language:
            text_groups[normalized_text(row.get("text", ""))].append(index)
            pattern = str(row.get("frame_pattern_id", "")).strip()
            if pattern:
                pattern_groups[pattern].append(index)

    parent = {index: index for indices in text_groups.values() for index in indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for indices in (*text_groups.values(), *pattern_groups.values()):
        for index in indices[1:]:
            union(indices[0], index)

    components: dict[int, list[int]] = defaultdict(list)
    for index in parent:
        components[find(index)].append(index)
    changed_rows = 0
    changed_components = 0
    for indices in components.values():
        splits = {str(rows[index].get("split", "")) for index in indices}
        if len(splits) < 2:
            continue
        target = max(splits, key=lambda split: SPLIT_PRIORITY.get(split, -1))
        changed_components += 1
        for index in indices:
            row = rows[index]
            if row.get("split") != target:
                row.setdefault("source_split", row.get("split"))
                row["split"] = target
                row["split_resolution"] = "text_and_frame_pattern_component_test_dev_train_priority"
                changed_rows += 1
    return rows, {
        "language": language,
        "text_groups_checked": len(text_groups),
        "frame_pattern_groups_checked": len(pattern_groups),
        "components_reconciled": changed_components,
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
