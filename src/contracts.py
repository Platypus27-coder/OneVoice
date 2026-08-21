"""Typed contracts shared by the OneVoice V2 runtime stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class CommitKind(str, Enum):
    WAIT = "WAIT"
    NORMAL = "COMMIT_NORMAL"
    SAFETY = "COMMIT_SAFETY"


@dataclass(slots=True)
class AudioFrame:
    samples: np.ndarray
    sample_rate: int
    sequence: int
    captured_at: float


@dataclass(slots=True)
class ASRHypothesis:
    text: str
    stable_prefix: str
    unstable_tail: str
    direction: str
    started_at: float
    updated_at: float
    endpoint: bool = False
    emotion: str = "neutral"
    event: str = "speech"


@dataclass(slots=True, frozen=True)
class CanonicalMention:
    canonical_id: str
    domain: str
    source_text: str
    vi_standard: str
    en_standard: str
    risk_level: str = "normal"


@dataclass(slots=True)
class ContextResult:
    source_text: str
    normalized_text: str
    canonical_mentions: list[CanonicalMention] = field(default_factory=list)
    domain: str | None = None
    intent: str | None = None
    entities: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "normal"
    safety_candidates: list[Any] = field(default_factory=list)
    translation_memory: str | None = None


@dataclass(slots=True)
class CommitDecision:
    kind: CommitKind
    text: str = ""
    reason: str = ""
    safety_match: Any | None = None
    decided_at: float = 0.0


@dataclass(slots=True)
class TranslationResult:
    source_text: str
    translated_text: str
    direction: str
    validated: bool
    validation_errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SynthesizedChunk:
    audio: np.ndarray
    sample_rate: int
    engine: str
    commit_id: int
    committed_at: float
    first_audio_at: float

