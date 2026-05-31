"""
Audio Utilities — Text segmentation, normalization, audio processing
=====================================================================
Core audio and text utilities for the OneVoice TTS pipeline.

Adapted from BetterBox-TTS/general/general_tool_audio.py
Original author: Dolly VN / ContextBoxAI (CC BY-NC 4.0)
Adapted by: Team Impact — OneVoice AI Challenge 2026
"""

import re
import math
import numpy as np
import librosa
from pathlib import Path
from typing import List, Optional

# ── Segment types ─────────────────────────────────────────────────────────────
SEGMENT_TEXT  = "text"
SEGMENT_PAUSE = "pause"

# ── Pause durations per punctuation (ms) ──────────────────────────────────────
# Adapted from BetterBox-TTS/general/general_tool_audio.py (Dolly VN, CC BY-NC 4.0)
_PUNCT_PAUSE_MS = {
    ".": 450, "!": 450, "?": 450, "。": 450, "！": 450, "？": 450,
    ",": 200, "，": 200, "、": 200,
    ";": 250, "；": 250,
    ":": 200, "：": 200,
    "/": 150, "…": 300, "-": 120, "—": 150, "–": 150,
}


def segment_text(text: str) -> List[dict]:
    """
    Split text into spoken clauses and pause segments for TTS synthesis.
    Each clause is synthesized separately, pauses are inserted as silence.

    Adapted from BetterBox-TTS (Dolly VN / ContextBoxAI, CC BY-NC 4.0).
    """
    if not text or not text.strip():
        return []

    punct_pattern = r'(\.{2,}|…+|[.!?,;:/—–\-，。？！、；：])'
    raw_parts = re.split(punct_pattern, text)

    segments: List[dict] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if re.fullmatch(punct_pattern, part):
            key = "…" if re.fullmatch(r'\.{2,}|…+', part) else part
            segments.append({
                "type": SEGMENT_PAUSE,
                "content": key,
                "pause_ms": _PUNCT_PAUSE_MS.get(key, 200),
            })
        else:
            segments.append({"type": SEGMENT_TEXT, "content": part})

    return segments


def clear_text(text: str) -> str:
    """
    Normalize raw text: lowercase, clean special chars.
    Adapted from BetterBox-TTS clearText() (Dolly VN, CC BY-NC 4.0).
    """
    text = text.casefold()
    text = re.sub(r'[!@#$%^&*()_+\-=\[\]{};\'\":\\|,.<>/?`~\.…]+', ', ', text)
    text = " ".join(text.split()).strip(", ")
    return text or text.strip()


def fix_silent_audio(
    audio: np.ndarray,
    sr: int,
    threshold_ms: int = 50,
    silence_db: float = -45.0,
) -> np.ndarray:
    """
    Remove excessive silence gaps from synthesized audio.
    Adapted from BetterBox-TTS fix_silent_and_speed_audio() (Dolly VN, CC BY-NC 4.0).
    """
    if len(audio) == 0:
        return audio

    frame_size = int(0.01 * sr)
    threshold_linear = 10 ** (silence_db / 20.0)
    frames = [audio[i:i+frame_size] for i in range(0, len(audio), frame_size)]

    is_silent = [np.sqrt(np.mean(f**2)) < threshold_linear if len(f) > 0 else True for f in frames]

    result = []
    i, n = 0, len(frames)
    while i < n:
        if is_silent[i]:
            j = i
            while j < n and is_silent[j]:
                j += 1
            seg = audio[i*frame_size: min(j*frame_size, len(audio))]
            dur_ms = len(seg) / sr * 1000
            keep_ms = min(dur_ms, threshold_ms)
            keep_samples = int(keep_ms / 1000 * sr)
            result.append(seg[:keep_samples])
            i = j
        else:
            result.append(audio[i*frame_size: min((i+1)*frame_size, len(audio))])
            i += 1

    return np.concatenate(result) if result else audio


def apply_pitch_shift(audio: np.ndarray, sr: int, pitch_ratio: float) -> np.ndarray:
    """
    Shift pitch of audio using pedalboard (high quality).
    pitch_ratio: 1.0=unchanged, >1.0=higher, <1.0=lower
    """
    if pitch_ratio == 1.0:
        return audio
    try:
        from pedalboard import Pedalboard, PitchShift
        n_semitones = 12.0 * math.log2(max(0.5, min(2.0, float(pitch_ratio))))
        board = Pedalboard([PitchShift(semitones=n_semitones)])
        audio_2d = audio.reshape(1, -1).astype(np.float32)
        return board(audio_2d, sr).flatten()
    except Exception as e:
        print(f"[Audio] ⚠ Pitch shift failed: {e}")
        return audio


def get_reference_wav(wavs_dir: str = "wavs") -> Optional[str]:
    """Get first available reference .wav file from wavs/ directory."""
    import random
    wavs = Path(wavs_dir)
    if not wavs.exists():
        return None
    priority = wavs / "reference_sound.wav"
    if priority.exists():
        return str(priority)
    files = list(wavs.glob("*.wav"))
    return str(random.choice(files)) if files else None
