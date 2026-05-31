"""
Trạm 1: Streaming ASR — Whisper-Tiny (Multilingual)
=====================================================
Transcribes denoised audio chunks to text in a streaming fashion
to minimize latency. Supports Vietnamese and English detection.

Reference: Whisper — OpenAI (MIT License)
           https://github.com/openai/whisper
"""

import time
import queue
import numpy as np
import whisper


class WhisperASR:
    """
    Streaming ASR using Whisper-Tiny multilingual model.
    Processes audio in 3-second chunks and pushes transcribed text
    to the downstream translation queue.
    """

    def __init__(self, text_queue: queue.Queue, config: dict):
        self.q_out = text_queue
        self.model_name = config["asr"]["model_name"]
        self.language = config["asr"].get("language", None)  # None = auto-detect
        self.chunk_length_s = config["asr"]["chunk_length_s"]
        self.sample_rate = config["audio"]["sample_rate"]
        self._model = None

    def load(self):
        """Load Whisper-Tiny model (downloads on first run, ~150MB)."""
        print(f"[ASR] Loading {self.model_name}...")
        self._model = whisper.load_model("tiny")
        print("[ASR] ✅ Whisper-Tiny loaded.")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a single audio chunk.

        Args:
            audio: float32 numpy array at 16kHz

        Returns:
            Transcribed text string.
        """
        if self._model is None:
            raise RuntimeError("ASR not loaded. Call .load() first.")

        t0 = time.perf_counter()

        # Whisper expects float32 audio normalized to [-1, 1]
        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        result = self._model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            fp16=False,          # CPU inference
            without_timestamps=True,
        )
        text = result["text"].strip()
        detected_lang = result.get("language", "unknown")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[ASR] ⏱ {elapsed_ms:.0f}ms | Lang: {detected_lang} | \"{text}\"")

        return text, detected_lang

    def run(self, audio_queue: queue.Queue):
        """
        Worker loop: reads audio from queue, transcribes, pushes to text_queue.
        """
        print("[ASR Worker] ✅ Started")
        while True:
            try:
                audio_chunk = audio_queue.get(timeout=1)
                text, lang = self.transcribe(audio_chunk)
                if text:
                    self.q_out.put({"text": text, "lang": lang})
                audio_queue.task_done()
            except queue.Empty:
                continue
