"""
OneVoice Edge — Main Pipeline Orchestrator
==========================================
Connects all 4 stages using thread-safe queues for real-time,
low-latency Speech-to-Speech translation.

Pipeline flow:
  Microphone → [Trạm 0: Denoise] → [Trạm 1: ASR] → [Trạm 2: MT] → [Trạm 3: TTS] → Speaker

Target latency: < 1 second end-to-end
RAM budget: < 200 MB
Network: None (100% offline)
"""

import queue
import time
import threading
import yaml
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from audio.capture import AudioCapture
from audio.denoise import Denoiser
from asr.whisper_asr import WhisperASR
from translation.mt_engine import Translator
from tts.tts_engine import TTSEngine


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class OneVoicePipeline:
    """
    End-to-end real-time speech translation pipeline.
    Each stage runs in its own daemon thread, communicating
    through bounded queues to prevent memory buildup.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)
        q_size = self.cfg["pipeline"]["queue_maxsize"]

        # ── Inter-stage Queues ──────────────────────────────────────────────
        self.q_audio_raw   = queue.Queue(maxsize=q_size)   # Raw mic audio
        self.q_audio_clean = queue.Queue(maxsize=q_size)   # Denoised audio
        self.q_text_src    = queue.Queue(maxsize=q_size)   # Transcribed text
        self.q_text_tgt    = queue.Queue(maxsize=q_size)   # Translated text

        # ── Modules ─────────────────────────────────────────────────────────
        self.capture   = AudioCapture(self.q_audio_raw, self.cfg)
        self.denoiser  = Denoiser(self.cfg["denoise"]["model_path"])
        self.asr       = WhisperASR(self.q_text_src, self.cfg)
        self.translator = Translator(self.cfg)
        self.tts       = TTSEngine(self.cfg)

    def _denoise_worker(self):
        """Reads raw audio, denoises, pushes to clean audio queue."""
        print("[Denoise Worker] ✅ Started")
        while True:
            try:
                raw = self.q_audio_raw.get(timeout=1)
                if self.cfg["denoise"]["enabled"]:
                    try:
                        clean = self.denoiser.denoise(raw)
                    except Exception:
                        clean = self.denoiser.passthrough(raw)
                else:
                    clean = raw
                if not self.q_audio_clean.full():
                    self.q_audio_clean.put(clean)
                self.q_audio_raw.task_done()
            except queue.Empty:
                continue

    def load_models(self):
        """Load all models before starting workers."""
        print("\n🔄 Loading models...")
        t0 = time.perf_counter()

        if self.cfg["denoise"]["enabled"]:
            try:
                self.denoiser.load()
            except Exception as e:
                print(f"[Denoiser] ⚠ Could not load model: {e} — will passthrough")

        self.asr.load()
        self.translator.load()
        self.tts.load()

        elapsed = time.perf_counter() - t0
        print(f"\n✅ All models loaded in {elapsed:.1f}s\n")

    def start(self):
        """Start all workers and begin real-time translation."""
        self.load_models()

        threads = [
            threading.Thread(target=self.capture.start, daemon=True, name="AudioCapture"),
            threading.Thread(target=self._denoise_worker, daemon=True, name="Denoise"),
            threading.Thread(target=self.asr.run,
                             args=(self.q_audio_clean,), daemon=True, name="ASR"),
            threading.Thread(target=self.translator.run,
                             args=(self.q_text_src, self.q_text_tgt), daemon=True, name="MT"),
            threading.Thread(target=self.tts.run,
                             args=(self.q_text_tgt,), daemon=True, name="TTS"),
        ]

        print("🎙️ OneVoice Edge is LIVE — speak into the microphone...")
        print("   Press Ctrl+C to stop.\n")

        for t in threads:
            t.start()

        try:
            while True:
                time.sleep(1)
                if self.cfg["pipeline"]["log_latency"]:
                    print(
                        f"[Pipeline] Queues — "
                        f"raw:{self.q_audio_raw.qsize()} "
                        f"clean:{self.q_audio_clean.qsize()} "
                        f"src_text:{self.q_text_src.qsize()} "
                        f"tgt_text:{self.q_text_tgt.qsize()}"
                    )
        except KeyboardInterrupt:
            print("\n⏹ Stopping OneVoice Edge pipeline.")
            self.capture.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OneVoice Edge — Real-time Speech Translation")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    pipeline = OneVoicePipeline(config_path=args.config)
    pipeline.start()
