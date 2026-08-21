"""Stable-prefix alignment and safety-aware semantic commit decisions."""

from __future__ import annotations

import re
import time

from contracts import ASRHypothesis, CommitDecision, CommitKind, ContextResult


_OPEN_TAILS = {
    "không", "đừng", "chưa", "nếu", "nhưng", "trước khi", "sau khi",
    "không quá", "ít nhất", "nhiều nhất", "not", "do not", "if", "but",
    "before", "after", "at least", "at most",
}
_NUMBER_TAIL = re.compile(r"(?:^|\s)[+-]?\d+(?:[.,]\d+)?\s*$")
_DIRECTION_TAIL = re.compile(r"\b(trái|phải|lên|xuống|left|right|up|down)\s*$", re.I)


class StablePrefixAligner:
    def __init__(self):
        self._previous: list[str] = []

    def update(self, text: str) -> tuple[str, str]:
        current = text.split()
        size = 0
        for before, now in zip(self._previous, current):
            if before.casefold() != now.casefold():
                break
            size += 1
        stable = " ".join(current[:size])
        unstable = " ".join(current[size:])
        self._previous = current
        return stable, unstable

    def reset(self) -> None:
        self._previous = []


class RollingHypothesisAssembler:
    """Merge exact overlap between consecutive rolling-window transcripts."""

    def __init__(self):
        self._assembled: list[str] = []

    def update(self, window_text: str, endpoint: bool = False) -> str:
        current = window_text.split()
        if endpoint or not self._assembled:
            self._assembled = current
            return " ".join(current)
        overlap = 0
        maximum = min(len(self._assembled), len(current))
        for size in range(maximum, 0, -1):
            left = [word.casefold() for word in self._assembled[-size:]]
            right = [word.casefold() for word in current[:size]]
            if left == right:
                overlap = size
                break
        if overlap:
            self._assembled.extend(current[overlap:])
        elif [word.casefold() for word in current] != [
            word.casefold() for word in self._assembled[-len(current):]
        ]:
            # Do not append guessed text when windows cannot be aligned. A
            # contracted hypothesis makes the stable-prefix controller wait.
            self._assembled = current
        return " ".join(self._assembled)

    def reset(self) -> None:
        self._assembled = []


class SemanticCommitController:
    def __init__(self, safety_confirmations: int = 2):
        self.safety_confirmations = safety_confirmations
        self._emitted_words = 0
        self._last_safety_id: str | None = None
        self._safety_streak = 0

    def decide(
        self, hypothesis: ASRHypothesis, context: ContextResult
    ) -> CommitDecision:
        now = time.perf_counter()
        safety = context.safety_candidates[0] if context.safety_candidates else None
        if safety:
            if safety.safety_id == self._last_safety_id:
                self._safety_streak += 1
            else:
                self._last_safety_id = safety.safety_id
                self._safety_streak = 1
            if hypothesis.endpoint or self._safety_streak >= self.safety_confirmations:
                self._emitted_words = len(hypothesis.text.split())
                return CommitDecision(
                    kind=CommitKind.SAFETY,
                    text=hypothesis.text,
                    reason="confirmed_safety_phrase",
                    safety_match=safety,
                    decided_at=now,
                )
            return CommitDecision(
                CommitKind.WAIT,
                reason="safety_phrase_pending_confirmation",
                decided_at=now,
            )
        else:
            self._last_safety_id = None
            self._safety_streak = 0

        candidate = hypothesis.text if hypothesis.endpoint else hypothesis.stable_prefix
        words = candidate.split()
        if len(words) <= self._emitted_words:
            return CommitDecision(CommitKind.WAIT, reason="no_new_stable_text", decided_at=now)

        new_text = " ".join(words[self._emitted_words :]).strip()
        if not hypothesis.endpoint and self._has_open_semantics(candidate, context):
            return CommitDecision(CommitKind.WAIT, reason="open_semantic_dependency", decided_at=now)

        self._emitted_words = len(words)
        return CommitDecision(
            CommitKind.NORMAL,
            text=new_text,
            reason="endpoint" if hypothesis.endpoint else "stable_prefix",
            decided_at=now,
        )

    @staticmethod
    def _has_open_semantics(text: str, context: ContextResult) -> bool:
        lowered = " ".join(text.casefold().split())
        if any(lowered.endswith(marker) for marker in _OPEN_TAILS):
            return True
        # Inspect the committed candidate itself. Entities from the unstable
        # tail must never make an incomplete stable prefix look complete.
        if _NUMBER_TAIL.search(lowered):
            return True
        if _DIRECTION_TAIL.search(lowered):
            return True
        return False

    def reset(self) -> None:
        self._emitted_words = 0
        self._last_safety_id = None
        self._safety_streak = 0
