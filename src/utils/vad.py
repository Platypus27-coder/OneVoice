"""
Voice Activity Detection — Silero VAD wrapper
==============================================
Detects speech segments in audio, removes silence and artifacts.

Adapted from BetterBox-TTS/general/noise_detect_VAD.py
Original author: Dolly VN / ContextBoxAI (CC BY-NC 4.0)
Adapted by: Team Impact — OneVoice AI Challenge 2026
"""

import torch
import numpy as np
import librosa

# Silero VAD (loaded lazily via torch.hub)
_VAD_MODEL = None
_VAD_UTILS = None


def _load_vad():
    """Load Silero VAD model (singleton, cached after first load)."""
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        try:
            _VAD_MODEL, _VAD_UTILS = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
                trust_repo=True,
            )
            print("[VAD] ✅ Silero VAD loaded.")
        except Exception as e:
            print(f"[VAD] ⚠ Could not load Silero VAD: {e}")
    return _VAD_MODEL, _VAD_UTILS


def vad_trim(audio: np.ndarray, sr: int, margin_s: float = 0.05) -> np.ndarray:
    """
    Trim silence from audio using Silero VAD.
    Only keeps segments with detected speech activity.

    Adapted from BetterBox-TTS noise_detect_VAD.py (Dolly VN / ContextBoxAI, CC BY-NC 4.0).

    Args:
        audio: float32 numpy array
        sr: sample rate of input audio
        margin_s: padding margin around speech (seconds)

    Returns:
        Trimmed audio keeping only speech segments.
    """
    if len(audio) == 0:
        return audio

    model, utils = _load_vad()
    if model is None:
        # Fallback: energy-based trim
        trimmed, _ = librosa.effects.trim(audio, top_db=30)
        return trimmed

    try:
        get_speech_timestamps = utils[0]
        # In newer silero-vad (v5+), collect_chunks is at index 4. In older versions it might be at 4 too, 
        # but the tuple length differs (5 vs 6).
        collect_chunks = [u for u in utils if callable(u) and getattr(u, "__name__", "") == "collect_chunks"]
        if collect_chunks:
            collect_chunks = collect_chunks[0]
        else:
            collect_chunks = utils[4]
            
        # Resample to 16kHz for VAD
        vad_sr = 16000
        wav_16k = librosa.resample(audio, orig_sr=sr, target_sr=vad_sr) if sr != vad_sr else audio
        wav_tensor = torch.tensor(wav_16k, dtype=torch.float32)

        timestamps = get_speech_timestamps(
            wav_tensor, model,
            sampling_rate=vad_sr,
            threshold=0.5,
            neg_threshold=0.30,
            min_speech_duration_ms=80,
            min_silence_duration_ms=100,
            speech_pad_ms=int(margin_s * 1000),
        )

        if not timestamps:
            return np.zeros(0, dtype=audio.dtype)

        speech_np = collect_chunks(timestamps, wav_tensor).numpy()

        if sr != vad_sr:
            speech_np = librosa.resample(speech_np, orig_sr=vad_sr, target_sr=sr)

        return speech_np.astype(np.float32)

    except Exception as e:
        print(f"[VAD] ⚠ Error: {e} — using energy trim fallback")
        trimmed, _ = librosa.effects.trim(audio, top_db=30)
        return trimmed
