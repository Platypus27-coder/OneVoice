"""
Trạm 3: Text-to-Speech Engine
==============================
Synthesizes translated text to natural-sounding speech.
Uses BetterBox-TTS for Vietnamese output and Tiny VITS for English output.
VALL-E X is available as an optional Premium Mode (voice cloning).

References:
  BetterBox-TTS — Dolly VN / ContextBoxAI (CC BY-NC 4.0)
    https://github.com/nowtranminh1-TTS/BetterBox-TTS
  VALL-E X — Plachtaa / Songting (MIT License)
    https://github.com/Plachtaa/VALL-E-X
"""

import time
import queue
import numpy as np
import sounddevice as sd


class TTSEngine:
    """
    Text-to-Speech engine supporting multiple backends:
      - "betterbox" : Vietnamese TTS (BetterBox-TTS / OmniVoice)
      - "vits"      : English TTS (Tiny VITS — lightweight ONNX)
      - "vallex"    : Premium cross-lingual voice cloning (VALL-E X)
    """

    def __init__(self, config: dict):
        self.cfg = config["tts"]
        self.sample_rate = config["audio"]["sample_rate"]
        self.default_engine = self.cfg.get("default_engine", "betterbox")
        self._betterbox = None
        self._vits_session = None

    def load(self):
        """Initialize TTS backends."""
        print(f"[TTS] Loading engine: {self.default_engine}")
        if self.default_engine == "betterbox":
            self._load_betterbox()
        elif self.default_engine == "vits":
            self._load_vits()
        print("[TTS] ✅ TTS Engine ready.")

    def _load_betterbox(self):
        """
        Load BetterBox-TTS (OmniVoice) for Vietnamese output.
        Imports from the reference BetterBox-TTS repository.
        Adjust sys.path if needed.
        """
        import sys, os
        betterbox_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", "BetterBox-TTS")
        )
        if betterbox_path not in sys.path:
            sys.path.insert(0, betterbox_path)
        # Import will be available once BetterBox-TTS dependencies are installed
        # from OmniVoice.omnivoice_inference.ttsOmni import OmniVoiceTTS
        # self._betterbox = OmniVoiceTTS(...)
        print(f"[TTS] BetterBox-TTS path: {betterbox_path}")
        print("[TTS] ⚠ BetterBox stub loaded (wire up OmniVoiceTTS when model is ready)")

    def _load_vits(self):
        """Load Tiny VITS ONNX model for English output."""
        import onnxruntime as ort
        model_path = self.cfg["vits"]["model_path"]
        try:
            self._vits_session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            print(f"[TTS] Tiny VITS ONNX loaded from: {model_path}")
        except Exception as e:
            print(f"[TTS] ⚠ VITS model not found ({e}). Using stub.")

    def synthesize(self, text: str, direction: str = "vi2en") -> np.ndarray:
        """
        Convert text to audio.

        Args:
            text: text to synthesize
            direction: "vi2en" (output is EN audio) or "en2vi" (output is VI audio)

        Returns:
            Audio as float32 numpy array at 16kHz.
        """
        t0 = time.perf_counter()

        # Route to correct TTS backend
        if direction == "en2vi":
            audio = self._synthesize_vi(text)   # BetterBox
        else:
            audio = self._synthesize_en(text)   # Tiny VITS

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[TTS] ⏱ {elapsed_ms:.0f}ms | synthesized: \"{text}\"")
        return audio

    def _synthesize_vi(self, text: str) -> np.ndarray:
        """Synthesize Vietnamese speech using BetterBox-TTS."""
        if self._betterbox is not None:
            return self._betterbox.generate(text)
        # Stub: return 0.5s of silence
        return np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)

    def _synthesize_en(self, text: str) -> np.ndarray:
        """Synthesize English speech using Tiny VITS."""
        if self._vits_session is not None:
            # Run VITS ONNX inference (depends on model input format)
            pass
        # Stub: return 0.5s of silence
        return np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)

    def play(self, audio: np.ndarray):
        """Play audio through the speaker."""
        sd.play(audio, samplerate=self.sample_rate)
        sd.wait()

    def run(self, text_queue: queue.Queue):
        """Worker loop: reads translated text, synthesizes, plays to speaker."""
        print("[TTS Worker] ✅ Started")
        while True:
            try:
                item = text_queue.get(timeout=1)
                text = item["text"]
                direction = item.get("direction", "vi2en")
                audio = self.synthesize(text, direction=direction)
                self.play(audio)
                text_queue.task_done()
            except queue.Empty:
                continue
