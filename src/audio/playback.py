"""Convert generated audio to a device rate without changing speech speed."""

from __future__ import annotations

import math

import numpy as np


def prepare_playback_audio(
    audio: np.ndarray, source_rate: int, output_rate: int
) -> np.ndarray:
    for rate in (source_rate, output_rate):
        if not math.isfinite(float(rate)) or rate <= 0 or int(rate) != rate:
            raise ValueError("Audio sample rates must be positive finite integers")
    value = np.asarray(audio)
    if value.ndim not in (1, 2) or value.size == 0:
        raise ValueError("Playback requires non-empty mono or multi-channel audio")
    if value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError("Playback requires finite floating-point samples")
    value = value.astype(np.float32, copy=False)
    if source_rate == output_rate:
        return value
    from scipy.signal import resample_poly

    factor = math.gcd(int(source_rate), int(output_rate))
    return resample_poly(
        value, int(output_rate) // factor, int(source_rate) // factor, axis=0
    ).astype(np.float32)
