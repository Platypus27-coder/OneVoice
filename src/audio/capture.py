"""32 ms microphone frame capture for the OneVoice V2 streaming runtime."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from contracts import AudioFrame

try:
    import sounddevice as sd
except ImportError:
    sd = None


class AudioCapture:
    def __init__(self, audio_queue: queue.Queue, config: dict):
        self.q = audio_queue
        self.sample_rate = int(config["audio"]["sample_rate"])
        self.chunk_size = int(config["audio"].get("chunk_size", 512))
        self._running = False
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self.dropped_frames = 0
        self.error: BaseException | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        del frames, time_info
        if status:
            print(f"[AudioCapture] Warning: {status}")
        self._sequence += 1
        frame = AudioFrame(
            samples=np.ascontiguousarray(indata[:, 0], dtype=np.float32),
            sample_rate=self.sample_rate,
            sequence=self._sequence,
            captured_at=time.perf_counter(),
        )
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1
            print("[AudioCapture] Warning: input queue full; frame dropped")

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is required for microphone capture")
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        print(
            f"[AudioCapture] Started ({self.chunk_size * 1000 / self.sample_rate:.0f} ms frames)"
        )

    def _stream_loop(self) -> None:
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=self._callback,
            ):
                while self._running:
                    sd.sleep(50)
        except BaseException as exc:
            self.error = exc
            self._running = False

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1)
        print("[AudioCapture] Stopped")
