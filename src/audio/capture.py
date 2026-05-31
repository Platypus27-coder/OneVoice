"""
Trạm 0: Audio Capture & Voice Activity Detection
=================================================
Captures audio from microphone in real-time and uses Silero VAD
to detect speech segments before pushing to the pipeline queue.

Reference: Silero VAD — https://github.com/snakers4/silero-vad (MIT License)
"""

import queue
import threading
import numpy as np
import sounddevice as sd
import torch

# ── Silero VAD ──────────────────────────────────────────────────────────────
VAD_MODEL, VAD_UTILS = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=True,
)
(get_speech_timestamps, _, read_audio, *_) = VAD_UTILS


class AudioCapture:
    """
    Captures microphone audio and segments it into speech chunks
    using Silero VAD. Pushes ready chunks to a shared Queue.
    """

    def __init__(self, audio_queue: queue.Queue, config: dict):
        self.q = audio_queue
        self.sample_rate = config["audio"]["sample_rate"]
        self.chunk_size = config["audio"]["chunk_size"]
        self.vad_threshold = config["audio"]["vad_threshold"]
        self.min_speech_ms = config["audio"]["vad_min_speech_ms"]
        self._running = False
        self._buffer = np.array([], dtype=np.float32)

    def _callback(self, indata: np.ndarray, frames: int, time, status):
        """Called by sounddevice for every audio chunk."""
        audio = indata[:, 0].astype(np.float32)
        self._buffer = np.concatenate([self._buffer, audio])

        # Run VAD every ~1 second of buffered audio
        if len(self._buffer) >= self.sample_rate:
            chunk = torch.from_numpy(self._buffer.copy())
            speech_ts = get_speech_timestamps(
                chunk,
                VAD_MODEL,
                threshold=self.vad_threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=self.min_speech_ms,
            )
            if speech_ts:
                # Speech detected — push chunk to pipeline
                if not self.q.full():
                    self.q.put(self._buffer.copy())
            self._buffer = np.array([], dtype=np.float32)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        print("[AudioCapture] ✅ Started (Silero VAD active)")

    def _stream_loop(self):
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._callback,
        ):
            while self._running:
                sd.sleep(100)

    def stop(self):
        self._running = False
        print("[AudioCapture] ⏹ Stopped")
