"""
Trạm 3: Text-to-Speech Engine (VI ↔ EN)
=========================================
Routes TTS output based on translation direction:
  - VI output (EN→VI direction): OmniVoice / BetterBox-TTS
  - EN output (VI→EN direction): Whisper-based English TTS / VITS-tiny

Premium Mode (Voice Cloning):
  - VALL-E X (activated when device has sufficient resources)
  - Preserves speaker voice identity across languages

References:
  BetterBox-TTS — Dolly VN / ContextBoxAI (CC BY-NC 4.0)
    https://github.com/nowtranminh1-TTS/BetterBox-TTS
  VALL-E X — Plachtaa / Songting (MIT License)
    https://github.com/Plachtaa/VALL-E-X
"""

import time
import queue
import sys
import os
import numpy as np
import sounddevice as sd


class TTSEngine:
    """
    Text-to-Speech router for VI↔EN pipeline.

    Routing:
      "vi2en" direction → output is English → English TTS (VITS/espeak)
      "en2vi" direction → output is Vietnamese → OmniVoice/BetterBox
    """

    def __init__(self, config: dict):
        self.cfg = config["tts"]
        self.sample_rate = config["audio"]["sample_rate"]
        self.default_engine = self.cfg.get("default_engine", "betterbox")
        self._omni = None          # OmniVoice for Vietnamese TTS
        self._en_tts = None        # English TTS engine
        self._vallex = None        # VALL-E X (Premium Mode)

    def load(self):
        """Initialize all TTS backends."""
        print(f"[TTS] Initializing engines...")
        self._load_omnivoice()
        self._load_english_tts()
        print("[TTS] ✅ TTS Engine ready.")

    def _load_omnivoice(self):
        """
        Load OmniVoice (BetterBox-TTS) for Vietnamese speech synthesis.
        Ported from BetterBox-TTS/OmniVoice/omnivoice_inference/ttsOmni.py
        """
        # Locate BetterBox-TTS repo relative to this file
        betterbox_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", "BetterBox-TTS")
        )
        omni_root = os.path.join(betterbox_root, "OmniVoice")

        for path in [betterbox_root, omni_root]:
            if path not in sys.path:
                sys.path.insert(0, path)

        try:
            from OmniVoice.omnivoice_inference.ttsOmni import Omni, generate_speech_omni
            model_path = self.cfg.get("betterbox", {}).get(
                "model_path", os.path.join(omni_root, "modelOmniLocal")
            )
            self._omni = Omni(model_path=model_path)
            self._omni.loadModelOmni()
            self._generate_speech_omni = generate_speech_omni
            print(f"[TTS] ✅ OmniVoice loaded from: {model_path}")
        except Exception as e:
            print(f"[TTS] ⚠ OmniVoice not available ({e}). Vietnamese TTS in stub mode.")
            self._omni = None

    def _load_english_tts(self):
        """
        Load lightweight English TTS.
        Priority: VITS-ONNX → pyttsx3 (offline fallback) → espeak
        """
        vits_path = self.cfg.get("vits", {}).get("model_path", "models/vits_en_tiny.onnx")
        if os.path.exists(vits_path):
            try:
                import onnxruntime as ort
                self._en_tts = ort.InferenceSession(
                    vits_path, providers=["CPUExecutionProvider"]
                )
                print(f"[TTS] ✅ VITS English TTS loaded from: {vits_path}")
                return
            except Exception as e:
                print(f"[TTS] ⚠ VITS load failed: {e}")

        # Fallback: pyttsx3 (offline, cross-platform)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            self._en_tts = engine
            print("[TTS] ✅ pyttsx3 English TTS loaded (fallback).")
        except Exception:
            print("[TTS] ⚠ No English TTS available — using silence stub.")

    def synthesize_vi(self, text: str) -> np.ndarray:
        """
        Synthesize Vietnamese speech using OmniVoice (BetterBox-TTS).
        Called for EN→VI direction (speaker heard Vietnamese output).
        """
        if self._omni is not None:
            try:
                t0 = time.perf_counter()
                # Use default reference audio if available
                ref_audio = self.cfg.get("betterbox", {}).get("reference_audio", None)
                result, status, _ = self._generate_speech_omni(
                    omni=self._omni,
                    text=text,
                    language="vi",
                    reference_audio=ref_audio,
                    speed=self.cfg.get("betterbox", {}).get("speed", 1.0),
                )
                if result is not None:
                    sr, audio = result
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    print(f"[TTS VI] ⏱ {elapsed_ms:.0f}ms | {status}")
                    return audio.astype(np.float32), sr
            except Exception as e:
                print(f"[TTS VI] ⚠ OmniVoice error: {e}")

        # Stub: silence
        return np.zeros(int(self.sample_rate * 0.5), dtype=np.float32), self.sample_rate

    def synthesize_en(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize English speech.
        Called for VI→EN direction (speaker heard English output).
        """
        t0 = time.perf_counter()

        # pyttsx3 path
        try:
            import pyttsx3
            if isinstance(self._en_tts, pyttsx3.Engine):
                import tempfile, soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                self._en_tts.save_to_file(text, tmp_path)
                self._en_tts.runAndWait()
                audio, sr = sf.read(tmp_path, dtype="float32")
                os.unlink(tmp_path)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(f"[TTS EN] ⏱ {elapsed_ms:.0f}ms | pyttsx3")
                return audio, sr
        except Exception as e:
            print(f"[TTS EN] ⚠ pyttsx3 error: {e}")

        # Stub: silence
        return np.zeros(int(self.sample_rate * 0.5), dtype=np.float32), self.sample_rate

    def synthesize(self, text: str, direction: str = "vi2en") -> tuple[np.ndarray, int]:
        """
        Route synthesis based on direction.

        Args:
            text: Text to speak (already translated)
            direction: "vi2en" → output EN speech | "en2vi" → output VI speech

        Returns:
            (audio_array, sample_rate)
        """
        if direction == "en2vi":
            # Output is Vietnamese
            return self.synthesize_vi(text)
        else:
            # Output is English
            return self.synthesize_en(text)

    def play(self, audio: np.ndarray, sample_rate: int = None):
        """Play synthesized audio through the speaker."""
        sr = sample_rate or self.sample_rate
        try:
            sd.play(audio, samplerate=sr)
            sd.wait()
        except Exception as e:
            print(f"[TTS] ⚠ Playback error: {e}")

    def run(self, text_queue: queue.Queue):
        """Worker loop: reads translated text, synthesizes, plays to speaker."""
        print("[TTS Worker] ✅ Started")
        while True:
            try:
                item = text_queue.get(timeout=1)
                text = item["text"]
                direction = item.get("direction", "vi2en")

                print(f"[TTS Worker] Synthesizing [{direction}]: \"{text}\"")
                audio, sr = self.synthesize(text, direction=direction)
                self.play(audio, sample_rate=sr)

                text_queue.task_done()
            except queue.Empty:
                continue
