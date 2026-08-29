"""Canonical benchmark scopes for OneVoice release and historical reports."""

from __future__ import annotations

from collections.abc import Iterable


# Explicitly reviewed on 2026-08-29.  Do not replace this with a broad glob:
# the report tree intentionally contains rejected experiments and diagnostics.
RELEASE_REPORTS: frozenset[str] = frozenset(
    {
        "denoiser/passthrough/clean",
        "denoiser/passthrough/noisy",
        "en_asr_onnx_fp32_v1/full_clean",
        "en_asr_onnx_fp32_v1/full_noisy",
        "mt/candidate_vi2en_validator_v4/vi2en/test/context",
        "mt/candidate_vi2en_validator_v4/vi2en/minimal/context",
        "mt/candidate_vi2en_validator_v4/vi2en/safety/context",
        "mt/candidate_en2vi_validator_v2/en2vi/test/context",
        "mt/candidate_en2vi_validator_v2/en2vi/minimal/context",
        "mt/candidate_en2vi_validator_v2/en2vi/safety/context",
    }
)

RELEASE_LABELS: dict[str, str] = {
    "denoiser/passthrough/clean": "VI→EN ASR · GIPFormer official ONNX baseline · clean",
    "denoiser/passthrough/noisy": "VI→EN ASR · GIPFormer official ONNX baseline · noisy",
    "en_asr_onnx_fp32_v1/full_clean": "EN→VI ASR · SenseVoice fine-tuned ONNX FP32 · clean",
    "en_asr_onnx_fp32_v1/full_noisy": "EN→VI ASR · SenseVoice fine-tuned ONNX FP32 · noisy",
    "mt/candidate_vi2en_validator_v4/vi2en/test/context": "VI→EN MT · EnViT5 fine-tuned + context validator v4 · test",
    "mt/candidate_vi2en_validator_v4/vi2en/minimal/context": "VI→EN MT · EnViT5 fine-tuned + context validator v4 · minimal",
    "mt/candidate_vi2en_validator_v4/vi2en/safety/context": "VI→EN MT · EnViT5 fine-tuned + safety context v4 · safety",
    "mt/candidate_en2vi_validator_v2/en2vi/test/context": "EN→VI MT · EnViT5 fine-tuned + context validator v2 · test",
    "mt/candidate_en2vi_validator_v2/en2vi/minimal/context": "EN→VI MT · EnViT5 fine-tuned + context validator v2 · minimal",
    "mt/candidate_en2vi_validator_v2/en2vi/safety/context": "EN→VI MT · EnViT5 fine-tuned + safety context v2 · safety",
}


def select_release_rows(rows: Iterable[dict], profile: str = "all") -> list[dict]:
    """Select all history or only the explicitly promoted runtime reports."""

    if profile not in {"all", "release"}:
        raise ValueError(f"Unknown benchmark profile: {profile}")
    materialized = list(rows)
    if profile == "all":
        return materialized
    selected: list[dict] = []
    for row in materialized:
        report = str(row.get("report", ""))
        if report in RELEASE_REPORTS:
            item = dict(row)
            item["release_label"] = RELEASE_LABELS.get(report, report)
            selected.append(item)
    return selected


def release_scope_note() -> str:
    return (
        "Current runtime only: official GIPFormer ONNX baseline for VI→EN, "
        "fine-tuned SenseVoice ONNX FP32 for EN→VI, and promoted EnViT5 "
        "translation validators. Historical and rejected fine-tune runs are excluded."
    )
