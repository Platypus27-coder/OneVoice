"""Hard release-selection rules shared by the CLI runtime and tests."""

from __future__ import annotations

from pathlib import Path


class ReleasePolicyError(RuntimeError):
    """Raised before model loading when a rejected artifact is configured."""


REJECTED_GIPFORMER_MARKERS = (
    "head_ft",
    "icefall_ft",
    "gipformer_vi_construction",
)
RELEASE_GIPFORMER_DIRNAME = "gipformer"
RELEASE_SENSEVOICE_DIRNAME = "sensevoice_en_construction_v1_onnx_fp32"


def validate_release_config(config: dict, direction: str) -> None:
    """Refuse known rejected ASR candidates before importing model runtimes."""
    if direction not in {"vi2en", "en2vi"}:
        raise ReleasePolicyError(f"Unsupported release direction: {direction}")

    if direction == "vi2en":
        raw_path = str(config.get("asr", {}).get("gipformer_model_dir", "")).strip()
        name = Path(raw_path).name.casefold()
        lowered = raw_path.replace("\\", "/").casefold()
        rejected = next(
            (marker for marker in REJECTED_GIPFORMER_MARKERS if marker in lowered),
            None,
        )
        if rejected:
            raise ReleasePolicyError(
                f"Rejected GIPFormer fine-tune path ({rejected}): {raw_path}"
            )
        if name != RELEASE_GIPFORMER_DIRNAME:
            raise ReleasePolicyError(
                "VI→EN release runtime must use the selected official ONNX bundle "
                f"in a '{RELEASE_GIPFORMER_DIRNAME}' directory; got {raw_path!r}"
            )
        return

    cfg = config.get("sensevoice", {})
    raw_path = str(cfg.get("model_path", "")).strip()
    if Path(raw_path).name.casefold() != RELEASE_SENSEVOICE_DIRNAME:
        raise ReleasePolicyError(
            "EN→VI release runtime must use the verified SenseVoice FP32 bundle "
            f"in '{RELEASE_SENSEVOICE_DIRNAME}'; got {raw_path!r}"
        )
    if bool(cfg.get("quantize", False)):
        raise ReleasePolicyError(
            "SenseVoice INT8 was rejected by the quality gate; release runtime requires FP32"
        )
