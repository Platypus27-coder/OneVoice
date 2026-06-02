"""
OneVoice Edge — Main Pipeline Orchestrator (VI ↔ EN)
=====================================================
Connects all 4 stages using thread-safe queues for real-time,
low-latency Speech-to-Speech translation. Focused exclusively on
Vietnamese ↔ English translation.

Pipeline flow:
  Microphone
    → [Trạm 0: Denoise — GIPFormer ONNX]
    → [Trạm 1: ASR — GIPFormer (VI) / Whisper-Tiny (EN)]
    → [Trạm 2: MT — MarianMT (VI↔EN)]
    → [Trạm 3: TTS — OmniVoice (VI out) / pyttsx3 (EN out)]
    → Speaker

Direction modes:
  "vi2en" — Vietnamese speaker → English listener (default)
  "en2vi" — English speaker → Vietnamese listener

Target: < 1 second end-to-end latency, < 200MB RAM, 100% offline.
"""

import queue
import time
import threading
import yaml
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from audio.capture import AudioCapture
from audio.denoise import Denoiser
from asr.asr_manager import ASRManager
from translation.mt_engine import Translator
from tts.tts_engine import TTSEngine
from utils.text_normalizer import normalize
from utils.srt_generator import SRTGenerator


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class OneVoicePipeline:
    """
    End-to-end real-time VI↔EN speech translation pipeline.
    """

    def __init__(self, config_path: str = "config/config.yaml",
                 direction: str = "vi2en"):
        self.cfg = load_config(config_path)
        self.direction = direction  # "vi2en" or "en2vi"
        q_size = self.cfg["pipeline"]["queue_maxsize"]

        # ── Inter-stage Queues ──────────────────────────────────────────────
        self.q_audio_raw   = queue.Queue(maxsize=q_size)
        self.q_audio_clean = queue.Queue(maxsize=q_size)
        self.q_text_src    = queue.Queue(maxsize=q_size)
        self.q_text_tgt    = queue.Queue(maxsize=q_size)

        # ── Modules ─────────────────────────────────────────────────────────
        self.capture    = AudioCapture(self.q_audio_raw, self.cfg)
        self.denoiser   = Denoiser(num_threads=2)
        self.asr        = ASRManager(self.cfg)
        self.translator = Translator(self.cfg)
        self.tts        = TTSEngine(self.cfg)
        self.srt        = SRTGenerator(bilingual=True)

        # ── Latency tracking ────────────────────────────────────────────────
        self._latency_log: list = []

    def _denoise_worker(self):
        """Trạm 0: Reads raw audio, denoises, pushes to clean queue."""
        print("[Denoise Worker] ✅ Started")
        while True:
            try:
                raw = self.q_audio_raw.get(timeout=1)
                t0 = time.perf_counter()
                clean = self.denoiser.denoise(raw)
                elapsed = (time.perf_counter() - t0) * 1000
                if not self.q_audio_clean.full():
                    self.q_audio_clean.put((clean, elapsed))
                self.q_audio_raw.task_done()
            except queue.Empty:
                continue

    def _asr_worker(self):
        """Trạm 1: Transcribes audio, normalizes, pushes text."""
        print(f"[ASR Worker] ✅ Started (direction={self.direction})")
        while True:
            try:
                audio_tuple = self.q_audio_clean.get(timeout=1)
                audio, denoise_ms = audio_tuple

                t0 = time.perf_counter()
                result = self.asr.transcribe(audio, direction=self.direction)
                asr_ms = (time.perf_counter() - t0) * 1000

                if result["text"]:
                    # Normalize text before MT
                    normalized = normalize(result["text"], lang=result["lang"])
                    result["text"] = normalized
                    result["denoise_ms"] = denoise_ms
                    result["asr_ms"] = asr_ms
                    # Emotion and event are already in result from ASRManager
                    self.q_text_src.put(result)

                self.q_audio_clean.task_done()
            except queue.Empty:
                continue

    def _mt_worker(self):
        """Trạm 2: Translates text, pushes to TTS queue."""
        print("[MT Worker] ✅ Started (VI↔EN)")
        while True:
            try:
                item = self.q_text_src.get(timeout=1)
                t0 = time.perf_counter()
                translated = self.translator.translate(
                    item["text"], direction=item["direction"]
                )
                mt_ms = (time.perf_counter() - t0) * 1000

                if translated:
                    item["translated"] = translated
                    item["mt_ms"] = mt_ms
                    self.q_text_tgt.put(item)

                self.q_text_src.task_done()
            except queue.Empty:
                continue

    def _tts_worker(self):
        """Trạm 3: Synthesizes translated text, plays audio."""
        print("[TTS Worker] ✅ Started")
        while True:
            try:
                item = self.q_text_tgt.get(timeout=1)
                t0 = time.perf_counter()
                
                # Pass emotion to TTS if available
                emotion = item.get("emotion", "neutral")
                
                audio, sr = self.tts.synthesize(
                    item["translated"], 
                    direction=item["direction"],
                    emotion=emotion
                )
                tts_ms = (time.perf_counter() - t0) * 1000

                # Log full latency
                total_ms = (
                    item.get("denoise_ms", 0) +
                    item.get("asr_ms", 0) +
                    item.get("mt_ms", 0) +
                    tts_ms
                )
                self._log_latency(item, total_ms)

                # Play
                self.tts.play(audio, sample_rate=sr)

                # Generate SRT entry
                duration_s = len(audio) / sr
                self.srt.add_entry(item["text"], item["translated"], duration_s)

                self.q_text_tgt.task_done()
            except queue.Empty:
                continue

    def _log_latency(self, item: dict, total_ms: float):
        """Log per-utterance latency breakdown."""
        arrow = "VI→EN" if item.get("direction") == "vi2en" else "EN→VI"
        status = "✅" if total_ms < 1000 else "⚠️"
        print(
            f"\n{status} [{arrow}] Total: {total_ms:.0f}ms | "
            f"Denoise: {item.get('denoise_ms', 0):.0f}ms | "
            f"ASR: {item.get('asr_ms', 0):.0f}ms | "
            f"MT: {item.get('mt_ms', 0):.0f}ms | "
            f"TTS: remaining"
        )
        print(f"   \"{item['text']}\" → \"{item['translated']}\"\n")
        self._latency_log.append(total_ms)

    def load_models(self):
        """Load all models sequentially before starting workers."""
        print("\n🔄 Loading models (VI↔EN pipeline)...")
        t0 = time.perf_counter()

        self.denoiser.load()
        self.asr.load()
        self.translator.load()
        self.tts.load()

        elapsed = time.perf_counter() - t0
        print(f"\n✅ All models loaded in {elapsed:.1f}s\n")

    def start(self):
        """Start all workers and begin real-time translation."""
        self.load_models()

        direction_label = "🇻🇳 VI → EN 🇬🇧" if self.direction == "vi2en" else "🇬🇧 EN → VI 🇻🇳"

        workers = [
            threading.Thread(target=self.capture.start, daemon=True, name="AudioCapture"),
            threading.Thread(target=self._denoise_worker, daemon=True, name="Denoise"),
            threading.Thread(target=self._asr_worker, daemon=True, name="ASR"),
            threading.Thread(target=self._mt_worker, daemon=True, name="MT"),
            threading.Thread(target=self._tts_worker, daemon=True, name="TTS"),
        ]

        print(f"🎙️ OneVoice Edge is LIVE  —  {direction_label}")
        print(f"   Speak into the microphone. Press Ctrl+C to stop.\n")

        for t in workers:
            t.start()

        try:
            while True:
                time.sleep(5)
                if self._latency_log:
                    avg = sum(self._latency_log) / len(self._latency_log)
                    print(
                        f"[Pipeline] Queue sizes — "
                        f"raw:{self.q_audio_raw.qsize()} "
                        f"clean:{self.q_audio_clean.qsize()} "
                        f"src_text:{self.q_text_src.qsize()} "
                        f"tgt_text:{self.q_text_tgt.qsize()} | "
                        f"Avg latency: {avg:.0f}ms"
                    )
        except KeyboardInterrupt:
            self.capture.stop()
            # Save SRT on exit
            if self.srt.entry_count > 0:
                srt_path = f"output_{int(time.time())}.srt"
                self.srt.save(srt_path)
                print(f"\n📄 SRT subtitle saved: {srt_path}")
            print("\n⏹ OneVoice Edge stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OneVoice Edge — Real-time VI↔EN Speech Translation"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--direction",
        choices=["vi2en", "en2vi"],
        default="vi2en",
        help="Translation direction: vi2en (Vietnamese→English) or en2vi (English→Vietnamese)",
    )
    args = parser.parse_args()

    pipeline = OneVoicePipeline(config_path=args.config, direction=args.direction)
    pipeline.start()
