"""
Trạm 1: ASR Manager — GIPFormer ONNX (VI) + SenseVoice (EN)
============================================================
GIPFormer (gipformer-65M-rnnt) là mô hình ASR tiếng Việt state-of-the-art,
chạy nhanh qua sherpa-onnx với INT8 quantization.

SenseVoiceSmall được dùng làm nhận diện tiếng Anh (EN→VI direction),
nhờ khả năng nhận diện siêu nhanh và trích xuất nhãn cảm xúc.
"""

import time
import queue
import numpy as np
from huggingface_hub import hf_hub_download

from .sensevoice_asr import SenseVoiceASR

try:
    import sherpa_onnx
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False

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

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
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
        return {"text": text, "emotion": "neutral", "event": "speech"}


class ASRManager:
    """
    Manages both ASR engines and routes by translation direction.

      direction="vi2en" → GIPFormer (Vietnamese input)
      direction="en2vi" → SenseVoice (English input with emotion detection)
    """

    def __init__(self, config: dict):
        self.cfg = config
        self._vi_asr = GIPFormerASR(
            num_threads=self.cfg["asr"].get("num_threads", 2)
        )
        self._en_asr = SenseVoiceASR(config)
        self._loaded = False

    def load(self):
        self._vi_asr.load()
        # SenseVoice load occurs during init, but we ensure it's ready here
        self._loaded = True

    def transcribe(self, audio: np.ndarray, direction: str = "vi2en") -> dict:
        """
        Transcribe audio based on translation direction.

        Returns:
            dict: {"text": str, "lang": str, "direction": str, "emotion": str, "event": str}
        """
        if not self._loaded:
            raise RuntimeError("ASRManager not loaded. Call .load() first.")

        if direction == "vi2en":
            result = self._vi_asr.transcribe(audio)
            return {
                "text": result["text"], 
                "emotion": result["emotion"], 
                "event": result["event"],
                "lang": "vi", 
                "direction": "vi2en"
            }
        else:
            result = self._en_asr.transcribe(audio, sample_rate=16000)
            return {
                "text": result["text"], 
                "emotion": result["emotion"], 
                "event": result["event"],
                "lang": "en", 
                "direction": "en2vi"
            }
