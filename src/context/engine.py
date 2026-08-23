"""Lightweight construction-domain context engine for OneVoice V2."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from contracts import CanonicalMention, ContextResult
from safety.fast_path import SafetyFastPath, normalize_match_text

from .site_pack import SitePack


@dataclass(slots=True, frozen=True)
class _Term:
    canonical_id: str
    domain: str
    surface: str
    vi_standard: str
    en_standard: str
    risk_level: str
    priority: int


_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("STOP_WORK", ("dừng lại", "dừng máy", "stop work", "stop immediately")),
    ("PROHIBIT_ACTION", ("không được", "đừng", "do not", "must not")),
    ("WARN_DANGER", ("cảnh báo", "nguy hiểm", "coi chừng", "warning", "danger")),
    ("REQUEST_INSPECTION", ("kiểm tra", "inspect", "check")),
    ("ASK_MEASUREMENT", ("bao nhiêu", "how much", "measurement")),
    ("ASK_LOCATION", ("ở đâu", "where")),
    ("REPORT_PROBLEM", ("bị lỗi", "hư", "rò rỉ", "failed", "broken", "leaking")),
    ("GIVE_INSTRUCTION", ("hãy", "ngay", "please", "immediately")),
)

_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?)(?!\w)")
_UNIT_RE = re.compile(
    r"\b(mm|cm|m|km|kg|t|ton|bar|psi|mpa|kw|kwh|v|a|hz|rpm|%|độ c|degrees? celsius)\b",
    re.IGNORECASE,
)
_DIRECTION_RE = re.compile(
    r"\b(trái|phải|lên|xuống|trước|sau|left|right|up|down|forward|backward)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(dừng|ngắt|kiểm tra|nâng|hạ|bật|tắt|stop|disconnect|inspect|check|raise|lower|turn on|turn off)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(không|đừng|chưa|không được|không thể|not|do not|don't|must not|never)\b",
    re.IGNORECASE,
)
_DIRECTION_TRANSLATIONS = {
    "trái": "left", "phải": "right", "lên": "up", "xuống": "down",
    "trước": "forward", "sau": "backward",
    "left": "trái", "right": "phải", "up": "lên", "down": "xuống",
    "forward": "trước", "backward": "sau",
}


def _normal(text: str) -> str:
    return normalize_match_text(unicodedata.normalize("NFC", text))


class ConstructionContextEngine:
    """Canonicalizes terms and derives deterministic construction context."""

    def __init__(
        self,
        terminology_path: str | Path,
        aliases_path: str | Path,
        safety_path: str | Path,
        site_pack: SitePack | None = None,
        required_safety_review_status: str | None = None,
    ):
        self.site_pack = site_pack
        self._terms: list[_Term] = []
        self._memory: dict[tuple[str, str], str] = {}
        self._load_terminology(Path(terminology_path), Path(aliases_path))
        self._load_site_terms(site_pack)
        self._tries = {
            "vi2en": self._build_trie("vi2en"),
            "en2vi": self._build_trie("en2vi"),
        }
        overrides = site_pack.safety_overrides if site_pack else ()
        self.safety = SafetyFastPath(
            safety_path,
            overrides=overrides,
            required_review_status=required_safety_review_status,
        )

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str | Path,
        site_pack: SitePack | None = None,
        required_safety_review_status: str | None = None,
    ) -> "ConstructionContextEngine":
        root = Path(data_dir)
        return cls(
            root / "terminology_master.csv",
            root / "term_aliases.csv",
            root / "safety_fast_path.csv",
            site_pack=site_pack,
            required_safety_review_status=required_safety_review_status,
        )

    def _load_terminology(self, terminology: Path, aliases: Path) -> None:
        masters: dict[str, dict[str, str]] = {}
        with terminology.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                canonical_id = row.get("canonical_id", "").strip()
                if canonical_id:
                    masters[canonical_id] = row

        surfaces: dict[tuple[str, str], _Term] = {}
        for canonical_id, row in masters.items():
            base = self._make_term(row.get("vi_standard", ""), row, priority=10)
            if base:
                surfaces[(base.surface, canonical_id)] = base
            for alias in str(row.get("vi_aliases", "")).split("|"):
                term = self._make_term(alias, row, priority=10)
                if term:
                    surfaces[(term.surface, canonical_id)] = term

        with aliases.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                master = masters.get(row.get("canonical_id", "").strip())
                if not master:
                    continue
                term = self._make_term(row.get("surface_form", ""), master, priority=10)
                if term:
                    surfaces[(term.surface, term.canonical_id)] = term
        self._terms = sorted(surfaces.values(), key=lambda item: (-len(item.surface), -item.priority))

    @staticmethod
    def _make_term(surface: str, row: dict[str, str], priority: int) -> _Term | None:
        normalized = _normal(surface)
        if not normalized:
            return None
        return _Term(
            canonical_id=row.get("canonical_id", "").strip(),
            domain=row.get("domain", "").strip(),
            surface=normalized,
            vi_standard=row.get("vi_standard", "").strip(),
            en_standard=row.get("en_standard", "").strip(),
            risk_level=row.get("risk_level", "normal").strip() or "normal",
            priority=priority,
        )

    def _load_site_terms(self, site_pack: SitePack | None) -> None:
        if not site_pack:
            return
        site_terms: list[_Term] = []
        lexical_rows = [
            *site_pack.local_terms,
            *site_pack.equipment_ids,
            *site_pack.zone_names,
            *site_pack.company_abbreviations,
            *site_pack.slang,
        ]
        for index, row in enumerate(lexical_rows):
            if not isinstance(row, dict):
                continue
            canonical_id = str(row.get("canonical_id", f"SITE_TERM_{index + 1:04d}"))
            master = {
                "canonical_id": canonical_id,
                "domain": str(row.get("domain", "site")),
                "vi_standard": str(row.get("vi_standard", row.get("vi", ""))),
                "en_standard": str(row.get("en_standard", row.get("en", ""))),
                "risk_level": str(row.get("risk_level", "normal")),
            }
            raw_surfaces: Iterable[str] = [master["vi_standard"], *row.get("aliases", [])]
            for surface in raw_surfaces:
                term = self._make_term(str(surface), master, priority=100)
                if term:
                    site_terms.append(term)
        self._terms = sorted(
            [*site_terms, *self._terms], key=lambda item: (-item.priority, -len(item.surface))
        )

        for row in site_pack.translation_memory:
            if not isinstance(row, dict):
                continue
            vi, en = str(row.get("vi", "")).strip(), str(row.get("en", "")).strip()
            if vi and en:
                self._memory[("vi2en", _normal(vi))] = en
                self._memory[("en2vi", _normal(en))] = vi

    def analyze(self, text: str, direction: str) -> ContextResult:
        if direction not in {"vi2en", "en2vi"}:
            raise ValueError("direction must be 'vi2en' or 'en2vi'")
        normalized = _normal(text)
        mentions = self._find_mentions(normalized, direction)
        domains = Counter(item.domain for item in mentions if item.domain)
        intent = self._detect_intent(normalized)
        safety = self.safety.match(text, direction)
        risk = self._max_risk(item.risk_level for item in mentions)
        if safety:
            risk = self._max_risk((risk, safety.severity))
        entities = self._extract_entities(text)
        return ContextResult(
            source_text=text,
            normalized_text=normalized,
            canonical_mentions=mentions,
            domain=domains.most_common(1)[0][0] if domains else None,
            intent=safety.intent if safety else intent,
            entities=entities,
            risk_level=risk,
            safety_candidates=[safety] if safety else [],
            translation_memory=self._memory.get((direction, normalized)),
        )

    def _find_mentions(self, normalized: str, direction: str) -> list[CanonicalMention]:
        found: list[CanonicalMention] = []
        words = normalized.split()
        trie = self._tries[direction]
        index = 0
        while index < len(words):
            node = trie
            candidates: list[tuple[int, int, _Term]] = []
            cursor = index
            while cursor < len(words) and words[cursor] in node:
                node = node[words[cursor]]
                cursor += 1
                for term in node.get("_terms", []):
                    candidates.append((term.priority, cursor - index, term))
            if not candidates:
                index += 1
                continue
            _, length, term = max(
                candidates,
                key=lambda item: (item[0], item[1], len(item[2].surface)),
            )
            found.append(
                CanonicalMention(
                    canonical_id=term.canonical_id,
                    domain=term.domain,
                    source_text=" ".join(words[index : index + length]),
                    vi_standard=term.vi_standard,
                    en_standard=term.en_standard,
                    risk_level=term.risk_level,
                )
            )
            index += length
        return found

    def _build_trie(self, direction: str) -> dict:
        root: dict = {}
        for term in self._terms:
            surface = term.surface if direction == "vi2en" else _normal(term.en_standard)
            tokens = surface.split()
            if not tokens:
                continue
            node = root
            for token in tokens:
                node = node.setdefault(token, {})
            node.setdefault("_terms", []).append(term)
        return root

    @staticmethod
    def _detect_intent(normalized: str) -> str | None:
        for intent, markers in _INTENT_RULES:
            if any(marker in normalized for marker in markers):
                return intent
        return None

    @staticmethod
    def _extract_entities(text: str) -> dict[str, object]:
        entities: dict[str, object] = {}
        numbers = _NUMBER_RE.findall(text)
        units = _UNIT_RE.findall(text)
        normalized = _normal(text)
        directions = []
        for match in _DIRECTION_RE.finditer(normalized):
            direction = match.group(0)
            following = normalized[match.end() :]
            # These Vietnamese words are direction tokens in isolation, but the
            # constructions below are temporal or action idioms. Requiring a
            # literal "forward/backward/up/down" in their translations creates
            # false safety failures (for example, "before continuing" and
            # "climb onto the scaffold").
            if direction in {"trước", "sau"} and re.match(r"\s+khi\b", following):
                continue
            if direction in {"lên", "xuống"} and re.match(
                r"\s+(?:giàn giáo|hố(?: đào)?)\b", following
            ):
                continue
            directions.append(direction)
        actions = _ACTION_RE.findall(text)
        negations = []
        for match in _NEGATION_RE.finditer(normalized):
            negation = match.group(0)
            # "đã ... chưa?" is a yes/no completion question, not a
            # prohibition/negative instruction that must render as "not".
            if negation == "chưa" and "?" in text[match.end() :]:
                continue
            negations.append(negation)
        if numbers:
            entities["numbers"] = numbers
        if units:
            entities["units"] = units
        if directions:
            entities["directions"] = directions
        if actions:
            entities["actions"] = actions
        if negations:
            entities["negations"] = negations
        return entities

    @staticmethod
    def _max_risk(levels: Iterable[str]) -> str:
        ranks = {"normal": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        values = [str(level).casefold() for level in levels]
        return max(values, key=lambda item: ranks.get(item, 0), default="normal")

    @staticmethod
    def canonicalize_source(text: str, context: ContextResult, direction: str) -> str:
        result = text
        for mention in sorted(
            context.canonical_mentions, key=lambda item: -len(item.source_text)
        ):
            replacement = mention.vi_standard if direction == "vi2en" else mention.en_standard
            if replacement:
                result = re.sub(
                    r"(?<!\w)" + re.escape(mention.source_text) + r"(?!\w)",
                    replacement,
                    result,
                    flags=re.IGNORECASE,
                )
        return result

    @staticmethod
    def validate_translation(
        translated: str, context: ContextResult, direction: str
    ) -> list[str]:
        normalized = _normal(translated)
        # A fixed safety fast-path phrase is the deterministic benchmark source
        # of truth. Do not reject that exact translation merely because a broad
        # terminology or entity rule prefers another lexical realization.
        if any(
            normalized == _normal(candidate.translated_text)
            for candidate in context.safety_candidates
        ):
            return []
        errors: list[str] = []
        for mention in context.canonical_mentions:
            expected = mention.en_standard if direction == "vi2en" else mention.vi_standard
            if expected and _normal(expected) not in normalized:
                errors.append(f"missing_term:{mention.canonical_id}:{expected}")
        for value in context.entities.get("numbers", []):
            canonical = str(value).replace(",", ".")
            if canonical not in translated.replace(",", "."):
                errors.append(f"missing_number:{value}")
        for value in context.entities.get("units", []):
            unit = _normal(str(value))
            if unit and unit not in normalized:
                errors.append(f"missing_unit:{value}")
        for value in context.entities.get("directions", []):
            source = _normal(str(value))
            expected = _DIRECTION_TRANSLATIONS.get(source, source)
            if expected and _normal(expected) not in normalized:
                errors.append(f"missing_direction:{value}:{expected}")
        if context.entities.get("negations"):
            target_negations = (
                ("not", "do not", "don't", "must not", "never", "no ")
                if direction == "vi2en"
                else ("không", "đừng", "chưa", "không được")
            )
            source_unsafe = direction == "vi2en" and "không an toàn" in context.normalized_text
            semantic_unsafe = source_unsafe and "unsafe" in normalized
            if not semantic_unsafe and not any(marker in normalized for marker in target_negations):
                errors.append("missing_negation")
        return errors
