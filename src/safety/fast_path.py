"""Deterministic lookup for reviewed safety phrases."""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class SafetyDataError(RuntimeError):
    pass


def normalize_match_text(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


@dataclass(slots=True, frozen=True)
class SafetyMatch:
    safety_id: str
    source_text: str
    translated_text: str
    intent: str
    severity: str


class SafetyFastPath:
    def __init__(
        self,
        csv_path: str | Path,
        overrides: Iterable[dict] = (),
        required_review_status: str | None = None,
    ):
        self._vi: dict[str, SafetyMatch] = {}
        self._en: dict[str, SafetyMatch] = {}
        self.required_review_status = required_review_status
        self._load_csv(Path(csv_path))
        self._load_overrides(overrides)

    def _load_csv(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Safety fast-path data not found: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if not _as_bool(row.get("fixed_translation_candidate", True)):
                    continue
                if (
                    self.required_review_status
                    and row.get("review_status") != self.required_review_status
                ):
                    raise SafetyDataError(
                        f"Safety phrase {row.get('safety_id', '<unknown>')} is not "
                        f"{self.required_review_status}"
                    )
                safety_id = row.get("safety_id", "").strip()
                vi = row.get("vi", "").strip()
                en = row.get("en", "").strip()
                if not safety_id or not vi or not en:
                    continue
                common = {
                    "safety_id": safety_id,
                    "intent": row.get("intent", "WARN_DANGER").strip(),
                    "severity": row.get("severity", "critical").strip(),
                }
                self._vi[normalize_match_text(vi)] = SafetyMatch(
                    source_text=vi, translated_text=en, **common
                )
                self._en[normalize_match_text(en)] = SafetyMatch(
                    source_text=en, translated_text=vi, **common
                )

    def _load_overrides(self, overrides: Iterable[dict]) -> None:
        for index, row in enumerate(overrides):
            if not isinstance(row, dict):
                continue
            if (
                self.required_review_status
                and row.get("review_status") != self.required_review_status
            ):
                raise SafetyDataError(
                    f"Site safety override {index} is not {self.required_review_status}"
                )
            vi = str(row.get("vi", "")).strip()
            en = str(row.get("en", "")).strip()
            if not vi or not en:
                continue
            common = {
                "safety_id": str(row.get("safety_id", f"SITE_{index + 1:04d}")),
                "intent": str(row.get("intent", "WARN_DANGER")),
                "severity": str(row.get("severity", "critical")),
            }
            self._vi[normalize_match_text(vi)] = SafetyMatch(
                source_text=vi, translated_text=en, **common
            )
            self._en[normalize_match_text(en)] = SafetyMatch(
                source_text=en, translated_text=vi, **common
            )

    def match(self, text: str, direction: str) -> SafetyMatch | None:
        normalized = normalize_match_text(text)
        table = self._vi if direction == "vi2en" else self._en
        return table.get(normalized)
