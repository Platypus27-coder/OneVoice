"""Stateful VAD and rolling utterance events."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from contracts import AudioFrame


@dataclass(slots=True)
class RollingAudioEvent:
    audio: np.ndarray
    sample_rate: int
    started_at: float
    updated_at: float
    endpoint: bool
    denoise_ms: float


class RollingUtteranceSession:
    """Portable energy-VAD baseline with partial and endpoint events."""

    def __init__(self, audio_config: dict, pipeline_config: dict):
        self.sample_rate = int(audio_config["sample_rate"])
        self.frame_samples = int(audio_config.get("chunk_size", 512))
        self.frame_ms = self.frame_samples * 1000 / self.sample_rate
        self.threshold = float(audio_config.get("vad_energy_threshold", 0.015))
        self.min_speech_ms = float(audio_config.get("vad_min_speech_ms", 300))
        self.endpoint_ms = float(audio_config.get("vad_endpoint_ms", 500))
        self.max_utterance_ms = float(audio_config.get("max_utterance_ms", 15000))
        self.stride_ms = float(pipeline_config.get("rolling_stride_ms", 500))
        self.window_ms = float(pipeline_config.get("rolling_window_ms", 2000))
        pre_roll_count = max(
            1, round(float(audio_config.get("vad_pre_roll_ms", 192)) / self.frame_ms)
        )
        self._pre_roll: deque[tuple[AudioFrame, float]] = deque(maxlen=pre_roll_count)
        self._active = False
        self._frames: list[tuple[AudioFrame, float]] = []
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._last_emit_ms = 0.0
        self._last_sequence: int | None = None

    def accept(self, frame: AudioFrame, denoise_ms: float = 0.0) -> RollingAudioEvent | None:
        samples = np.asarray(frame.samples, dtype=np.float32)
        if int(frame.sample_rate) != self.sample_rate:
            raise ValueError(
                f"AudioFrame sample rate {frame.sample_rate} does not match session {self.sample_rate}"
            )
        if samples.ndim != 1:
            raise ValueError("AudioFrame samples must be mono float32")
        if samples.size != self.frame_samples:
            raise ValueError(
                f"AudioFrame has {samples.size} samples; expected {self.frame_samples}"
            )
        if self._last_sequence is not None and frame.sequence <= self._last_sequence:
            raise ValueError(
                f"AudioFrame sequence must increase ({frame.sequence} after {self._last_sequence})"
            )
        self._last_sequence = frame.sequence
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        voiced = rms >= self.threshold
        if not self._active:
            self._pre_roll.append((frame, denoise_ms))
            if not voiced:
                return None
            self._active = True
            self._frames = list(self._pre_roll)
            self._voiced_ms = self.frame_ms
            self._silence_ms = 0.0
            self._last_emit_ms = 0.0
            return None

        self._frames.append((frame, denoise_ms))
        if voiced:
            self._voiced_ms += self.frame_ms
            self._silence_ms = 0.0
        else:
            self._silence_ms += self.frame_ms
        duration_ms = len(self._frames) * self.frame_ms
        endpoint = self._silence_ms >= self.endpoint_ms or duration_ms >= self.max_utterance_ms
        ready = (
            self._voiced_ms >= self.min_speech_ms
            and duration_ms - self._last_emit_ms >= self.stride_ms
        )
        if endpoint:
            event = self._event(endpoint=True) if self._voiced_ms >= self.min_speech_ms else None
            self.reset()
            return event
        if ready:
            self._last_emit_ms = duration_ms
            return self._event(endpoint=False)
        return None

    def _event(self, endpoint: bool) -> RollingAudioEvent:
        selected = self._frames
        if not endpoint:
            window_frames = max(1, round(self.window_ms / self.frame_ms))
            selected = self._frames[-window_frames:]
        frames = [frame.samples for frame, _ in selected]
        return RollingAudioEvent(
            audio=np.concatenate(frames).astype(np.float32),
            sample_rate=self.sample_rate,
            started_at=self._frames[0][0].captured_at,
            updated_at=self._frames[-1][0].captured_at,
            endpoint=endpoint,
            denoise_ms=sum(value for _, value in selected),
        )

    def reset(self, *, clear_sequence: bool = False) -> None:
        """Reset utterance state while retaining frame ordering by default.

        Endpoint resets start the next utterance in the same capture stream, so
        their sequence numbers must remain monotonic. A new replay/capture
        session can explicitly clear the sequence with ``clear_sequence``.
        """
        self._active = False
        self._frames = []
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._last_emit_ms = 0.0
        if clear_sequence:
            self._last_sequence = None
        self._pre_roll.clear()
