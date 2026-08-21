"""Dependency-light corpus metrics with explicit empty-prediction handling."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence


def normalize_metric_text(text: str) -> str:
    value = unicodedata.normalize("NFC", str(text)).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def corpus_error_rate(
    references: Iterable[str], hypotheses: Iterable[str], unit: str = "word"
) -> float:
    refs, hyps = list(references), list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError("references and hypotheses must have equal length")
    edits = 0
    total = 0
    for reference, hypothesis in zip(refs, hyps):
        ref_text = normalize_metric_text(reference)
        hyp_text = normalize_metric_text(hypothesis)
        if unit == "word":
            ref_units, hyp_units = ref_text.split(), hyp_text.split()
        elif unit == "char":
            ref_units, hyp_units = list(ref_text.replace(" ", "")), list(hyp_text.replace(" ", ""))
        else:
            raise ValueError("unit must be 'word' or 'char'")
        edits += edit_distance(ref_units, hyp_units)
        total += len(ref_units)
    if total == 0:
        return 0.0 if edits == 0 else 1.0
    return edits / total


def wer(reference: str, hypothesis: str) -> float:
    return corpus_error_rate([reference], [hypothesis], unit="word")


def cer(reference: str, hypothesis: str) -> float:
    return corpus_error_rate([reference], [hypothesis], unit="char")

