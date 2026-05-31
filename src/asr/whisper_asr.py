"""
Trạm 1: Vietnamese ASR — GIPFormer ONNX (Primary) + Whisper-Tiny (Fallback)
=============================================================================
GIPFormer (gipformer-65M-rnnt) là mô hình ASR tiếng Việt state-of-the-art,
65M params, chạy nhanh qua sherpa-onnx với INT8 quantization.

Whisper-Tiny được dùng làm fallback để nhận diện tiếng Anh (EN→VI direction).

References:
  gipformer — G-Group AI Lab (MIT License)
    https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt
  Whisper — OpenAI (MIT License)
    https://github.com/openai/whisper
"""

import time
import queue
import numpy as np

try:
    import sherpa_onnx
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False

try:
    import whisper as openai_whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

from huggingface_hub import hf_hub_download

# ── GIPFormer model config ────────────────────────────────────────────────────
GIPFORMER_REPO   = "g-group-ai-lab/gipformer-65M-rnnt"
GIPFORMER_SAMPLE_RATE = 16000
GIPFORMER_FEATURE_DIM = 80
GIPFORMER_INT8_FILES  = {
    "encoder": "encoder-epoch-35-avg-6.int8.onnx",
    "decoder": "decoder-epoch-35-avg-6.int8.onnx",
    "joiner":  "joiner-epoch-35-avg-6.int8.onnx",
    "tokens":  "tokens.txt",
}


class GIPFormerASR:
    """
    Vietnamese ASR using GIPFormer (sherpa-onnx, INT8).
    Callable module wrapping gipformer/infer_onnx.py logic.
    """

    def __init__(self, num_threads: int = 2, decoding_method: str = "greedy_search"):
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self._recognizer = None

    def load(self):
        if not HAS_SHERPA:
            raise ImportError("sherpa-onnx not installed. Run: pip install sherpa-onnx")

        print("[GIPFormer ASR] Downloading INT8 model from HuggingFace...")
        paths = {}
        for key, filename in GIPFORMER_INT8_FILES.items():
            paths[key] = hf_hub_download(repo_id=GIPFORMER_REPO, filename=filename)
        print("[GIPFormer ASR] Model downloaded.")

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=paths["encoder"],
            decoder=paths["decoder"],
            joiner=paths["joiner"],
            tokens=paths["tokens"],
            num_threads=self.num_threads,
            sample_rate=GIPFORMER_SAMPLE_RATE,
            feature_dim=GIPFORMER_FEATURE_DIM,
            decoding_method=self.decoding_method,
        )
        print("[GIPFormer ASR] ✅ GIPFormer ready.")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if self._recognizer is None:
            raise RuntimeError("GIPFormer not loaded. Call .load() first.")

        # Convert stereo → mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        t0 = time.perf_counter()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_streams([stream])
        text = stream.result.text.strip()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        rtf = elapsed_ms / max((len(audio) / sample_rate * 1000), 1)
        print(f"[GIPFormer ASR] ⏱ {elapsed_ms:.0f}ms | RTF={rtf:.3f} | \"{text}\"")
        return text


class WhisperASR:
    """
    English ASR using Whisper-Tiny (fallback for EN→VI direction).
    """

    def __init__(self):
        self._model = None

    def load(self):
        if not HAS_WHISPER:
            raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")
        print("[Whisper ASR] Loading Whisper-Tiny...")
        self._model = openai_whisper.load_model("tiny")
        print("[Whisper ASR] ✅ Whisper-Tiny ready.")

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        if self._model is None:
            raise RuntimeError("Whisper not loaded. Call .load() first.")

        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        t0 = time.perf_counter()
        result = self._model.transcribe(audio, language=language, task="transcribe",
                                         fp16=False, without_timestamps=True)
        text = result["text"].strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[Whisper ASR] ⏱ {elapsed_ms:.0f}ms | \"{text}\"")
        return text


class ASRManager:
    """
    Manages both ASR engines and routes by translation direction.

      direction="vi2en" → GIPFormer (Vietnamese input)
      direction="en2vi" → Whisper-Tiny (English input)
    """

    def __init__(self, config: dict):
        self.cfg = config
        self._vi_asr = GIPFormerASR(
            num_threads=self.cfg["asr"].get("num_threads", 2)
        )
        self._en_asr = WhisperASR()
        self._loaded = False

    def load(self):
        self._vi_asr.load()
        self._en_asr.load()
        self._loaded = True

    def transcribe(self, audio: np.ndarray, direction: str = "vi2en") -> dict:
        """
        Transcribe audio based on translation direction.

        Returns:
            dict: {"text": str, "lang": str, "direction": str}
        """
        if not self._loaded:
            raise RuntimeError("ASRManager not loaded. Call .load() first.")

        if direction == "vi2en":
            text = self._vi_asr.transcribe(audio)
            return {"text": text, "lang": "vi", "direction": "vi2en"}
        else:
            text = self._en_asr.transcribe(audio, language="en")
            return {"text": text, "lang": "en", "direction": "en2vi"}

    def run(self, audio_queue: queue.Queue, text_queue: queue.Queue,
            direction: str = "vi2en"):
        """Worker loop for pipeline integration."""
        print(f"[ASR Worker] ✅ Started (direction={direction})")
        while True:
            try:
                audio_chunk = audio_queue.get(timeout=1)
                result = self.transcribe(audio_chunk, direction=direction)
                if result["text"]:
                    text_queue.put(result)
                audio_queue.task_done()
            except queue.Empty:
                continue


if __name__ == "__main__":
    import soundfile as sf
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

    # Test with gipformer sample audio
    audio_files = [
        "../../gipformer/data/audio1.wav",
        "../../gipformer/data/audio2.wav",
    ]

    asr = GIPFormerASR()
    asr.load()
    for f in audio_files:
        if os.path.exists(f):
            audio, sr = sf.read(f, dtype="float32")
            print(f"\nFile: {f}")
            print(f"Result: {asr.transcribe(audio, sr)}")
