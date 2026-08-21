"""Validated site-pack loading for project-specific vocabulary and safety rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SitePackError(ValueError):
    pass


@dataclass(slots=True)
class SitePack:
    site_id: str
    site_name: str
    project_type: str
    enabled_domains: set[str] = field(default_factory=set)
    local_terms: list[dict[str, Any]] = field(default_factory=list)
    equipment_ids: list[dict[str, Any]] = field(default_factory=list)
    zone_names: list[dict[str, Any]] = field(default_factory=list)
    company_abbreviations: list[dict[str, Any]] = field(default_factory=list)
    slang: list[dict[str, Any]] = field(default_factory=list)
    safety_overrides: list[dict[str, Any]] = field(default_factory=list)
    translation_memory: list[dict[str, Any]] = field(default_factory=list)


class SitePackLoader:
    REQUIRED = ("site_id", "site_name", "project_type")
    LIST_FIELDS = (
        "enabled_domains",
        "local_terms",
        "equipment_ids",
        "zone_names",
        "company_abbreviations",
        "slang",
        "safety_overrides",
        "translation_memory",
    )

    @classmethod
    def load(cls, path: str | Path) -> SitePack:
        source = Path(path)
        if not source.is_file():
            raise SitePackError(f"Site pack not found: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SitePackError(f"Invalid site pack JSON: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SitePack:
        if not isinstance(payload, dict):
            raise SitePackError("Site pack root must be a JSON object")
        missing = [key for key in cls.REQUIRED if not str(payload.get(key, "")).strip()]
        if missing:
            raise SitePackError(f"Missing required site-pack fields: {', '.join(missing)}")
        for key in cls.LIST_FIELDS:
            if key in payload and not isinstance(payload[key], list):
                raise SitePackError(f"Site-pack field '{key}' must be a list")

        lexical_fields = (
            "local_terms", "equipment_ids", "zone_names", "company_abbreviations", "slang"
        )
        surfaces: dict[str, tuple[str, str]] = {}
        for field_name in lexical_fields:
            for index, row in enumerate(payload.get(field_name, [])):
                if not isinstance(row, dict):
                    raise SitePackError(f"{field_name}[{index}] must be an object")
                vi = str(row.get("vi_standard", row.get("vi", ""))).strip()
                en = str(row.get("en_standard", row.get("en", ""))).strip()
                if not vi or not en:
                    raise SitePackError(f"{field_name}[{index}] requires vi/en text")
                canonical_id = str(row.get("canonical_id", f"{field_name}:{index}"))
                aliases = row.get("aliases", [])
                if not isinstance(aliases, list):
                    raise SitePackError(f"{field_name}[{index}].aliases must be a list")
                for surface in (vi, *[str(value) for value in aliases]):
                    normalized = " ".join(surface.casefold().split())
                    previous = surfaces.get(normalized)
                    current = (canonical_id, en.casefold())
                    if previous and previous != current:
                        raise SitePackError(
                            f"Alias collision for '{surface}' between {previous[0]} and {canonical_id}"
                        )
                    surfaces[normalized] = current

        memory: dict[tuple[str, str], str] = {}
        for index, row in enumerate(payload.get("translation_memory", [])):
            if not isinstance(row, dict):
                raise SitePackError(f"translation_memory[{index}] must be an object")
            vi, en = str(row.get("vi", "")).strip(), str(row.get("en", "")).strip()
            if not vi or not en:
                raise SitePackError(f"translation_memory[{index}] requires vi/en text")
            for direction, source, target in (("vi2en", vi, en), ("en2vi", en, vi)):
                key = (direction, " ".join(source.casefold().split()))
                if key in memory and memory[key] != target:
                    raise SitePackError(f"Translation-memory collision for '{source}'")
                memory[key] = target

        return SitePack(
            site_id=str(payload["site_id"]).strip(),
            site_name=str(payload["site_name"]).strip(),
            project_type=str(payload["project_type"]).strip(),
            enabled_domains=set(payload.get("enabled_domains", [])),
            local_terms=list(payload.get("local_terms", [])),
            equipment_ids=list(payload.get("equipment_ids", [])),
            zone_names=list(payload.get("zone_names", [])),
            company_abbreviations=list(payload.get("company_abbreviations", [])),
            slang=list(payload.get("slang", [])),
            safety_overrides=list(payload.get("safety_overrides", [])),
            translation_memory=list(payload.get("translation_memory", [])),
        )
